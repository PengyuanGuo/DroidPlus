#!/usr/bin/env python3
"""Read-only schema, cache, annotation and optional source-recording checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

if __package__:
    from .convert_droidplus_to_zarr import image_rgb, load_episode, resized_rgb, sha256
else:
    from convert_droidplus_to_zarr import image_rgb, load_episode, resized_rgb, sha256


def verify(path: Path, *, check_sources=False, full_images=False) -> dict:
    path = path.resolve()
    root = zarr.open_group(str(path), mode="r")
    errors, warnings = [], []
    result = {"zarr": str(path), "errors": errors, "warnings": warnings, "passed": False}

    def check(condition, message):
        if not condition:
            errors.append(message)

    check((path / ".zgroup").exists() and json.loads((path / ".zgroup").read_text()).get("zarr_format") == 2, "Container must be zarr v2")
    if "meta/episode_ends" not in root or "data/state" not in root:
        errors.append("Missing meta/episode_ends or data/state")
        return result
    ends = root["meta/episode_ends"][:]
    n = root["data/state"].shape[0]
    check(ends.dtype == np.int64 and ends.ndim == 1 and len(ends) > 0, "episode_ends must be nonempty int64 vector")
    check(bool(len(ends)) and bool(np.all(np.diff(np.r_[0, ends]) > 0)) and ends[-1] == n, "Episode ends must increase strictly and end at N")
    result.update({"frames": n, "episodes": len(ends), "image_check": "all frames" if full_images else "episode ends plus first/middle frames"})
    expected = {
        "agent_view": ((n, 224, 224, 3), "uint8"), "wrist_cam": ((n, 224, 224, 3), "uint8"),
        "state": ((n, 8), "float32"), "action": ((n, 8), "float32"),
        "prompt": ((n, 128), "uint8"), "phase_prompt": ((n, 128), "uint8"),
        "phase_label": ((n,), "int8"), "point_pixel": ((n, 2), "float32"),
        "timestamp": ((n,), "float64"), "source_seq": ((n,), "int64"),
        "point_needs_review": ((n,), "bool"),
    }
    for key, (shape, dtype) in expected.items():
        if f"data/{key}" not in root:
            errors.append(f"Missing data/{key}")
            continue
        array = root[f"data/{key}"]
        check(array.shape == shape and array.dtype == np.dtype(dtype), f"Wrong shape/dtype for data/{key}: {array.shape}, {array.dtype}")
    for key in root["data"].array_keys():
        check(root[f"data/{key}"].shape[0] == n, f"Unequal length: data/{key}")
    for camera in ("agent_view", "wrist_cam"):
        key = f"_cache/224x224/{camera}.zarr/images"
        check(key in root, f"Missing training cache {key}")
        if key in root:
            check(root[key].shape == (n, 3, 224, 224) and root[key].dtype == np.uint8, f"Wrong cache shape/dtype: {key}")
    if errors:
        return result
    arrays = {key: root[f"data/{key}"][:] for key in expected if key not in ("agent_view", "wrist_cam")}
    for key in ("state", "action", "point_pixel", "timestamp"):
        check(np.isfinite(arrays[key]).all(), f"Nonfinite values in {key}")
    for key in ("state", "action"):
        check(((arrays[key][:, 7] >= 0) & (arrays[key][:, 7] <= 1)).all(), f"Gripper outside [0,1]: {key}")
    check(((arrays["point_pixel"] >= 0) & (arrays["point_pixel"] <= 1)).all(), "Missing/sentinel/out-of-bounds points")
    for key in ("prompt", "phase_prompt"):
        try:
            for row in arrays[key]:
                check(bool(bytes(row).rstrip(b"\0").decode("utf-8").strip()), f"Empty text in {key}")
        except UnicodeDecodeError:
            errors.append(f"Invalid UTF-8 in {key}")
    starts = np.r_[0, ends[:-1]]
    for episode_index, (start, end) in enumerate(zip(starts, ends)):
        start, end = int(start), int(end)
        labels = arrays["phase_label"][start:end]
        boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        check(labels[boundaries[:-1]].tolist() == [0, 1, 0, 1, 0, 1], f"Episode {episode_index}: wrong six-phase sequence")
        check(np.all(np.diff(arrays["timestamp"][start:end]) > 0), f"Episode {episode_index}: timestamps not increasing")
        check(np.all(np.diff(arrays["source_seq"][start:end]) == 1), f"Episode {episode_index}: source frame IDs not consecutive")
        check(np.all(arrays["prompt"][start:end] == arrays["prompt"][start]), f"Episode {episode_index}: episode prompt varies")
        for left, right in zip(boundaries[:-1] + start, boundaries[1:] + start):
            check(np.all(arrays["phase_prompt"][left:right] == arrays["phase_prompt"][left]), f"Episode {episode_index}: phase prompt varies inside a segment")
            check(np.allclose(arrays["point_pixel"][left:right], arrays["point_pixel"][left], rtol=0, atol=1e-6), f"Episode {episode_index}: point varies inside a segment")
    # Read in batches to test caches without loading all image arrays into RAM.
    indices = np.unique(np.r_[0, n // 2, ends - 1]).astype(int)
    batches = [slice(i, min(i + 100, n)) for i in range(0, n, 100)] if full_images else [slice(i, i + 1) for i in indices]
    for selection in batches:
        for camera in ("agent_view", "wrist_cam"):
            try:
                image = root[f"data/{camera}"][selection]
                cache = root[f"_cache/224x224/{camera}.zarr/images"][selection]
                check(np.array_equal(image.transpose(0, 3, 1, 2), cache), f"Image/cache mismatch at {selection.start}: {camera}")
            except Exception as exc:
                errors.append(f"Unreadable image/cache chunk at {selection.start}: {exc}")
    flagged = arrays["point_needs_review"]
    result.update({"review_flagged_frames": int(flagged.sum()), "review_flagged_episodes": int(sum(flagged[s:e].any() for s, e in zip(starts, ends))), "agent_roi": root.attrs.get("agent_roi_xy_size")})
    if flagged.any():
        warnings.append("Point review flags remain; the training loader does not exclude flagged frames")
    manifest_path = path / "conversion_manifest.json"
    check(manifest_path.exists(), "Missing conversion_manifest.json")
    if check_sources and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        check(len(manifest["episodes"]) == len(ends), "Manifest episode count mismatch")
        for index, entry in enumerate(manifest["episodes"]):
            try:
                source = Path(entry["source_steps"])
                check(sha256(source) == entry["source_sha256"], f"Episode {index}: corrected source changed after conversion; regenerate zarr")
                check(sha256(Path(entry["annotation_report"])) == entry["annotation_report_sha256"], f"Episode {index}: annotation report changed after conversion; regenerate zarr")
                source_episode = load_episode(source, root.attrs["agent_roi_xy_size"], root.attrs["wrist_roi_xy_size"], entry["instruction"])
                start, end = entry["zarr_start"], entry["zarr_end"]
                check(start == int(starts[index]) and end == int(ends[index]), f"Episode {index}: manifest slice mismatch")
                for key, values in source_episode["data"].items():
                    check(np.array_equal(root[f"data/{key}"][start:end], values), f"Episode {index}: {key} differs from converted source")
                for local_index in (0, source_episode["n"] - 1):
                    for camera, key, crop in (("left", "agent_view", source_episode["left_roi"]), ("wrist", "wrist_cam", source_episode["wrist_roi"])):
                        expected_image = resized_rgb(image_rgb(source_episode["paths"][camera][local_index]), crop)
                        check(np.array_equal(root[f"data/{key}"][start + local_index], expected_image), f"Episode {index}: {key} crop/RGB differs from source at frame {local_index}")
            except (ValueError, OSError, KeyError, IndexError) as exc:
                errors.append(f"Episode {index}: source verification failed: {exc}")
        result["source_check"] = "all low-dimensional values, source hashes, first/last camera frames per episode"
    result["passed"] = not errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--full-images", action="store_true", help="Decode every image/cache chunk instead of sampling")
    parser.add_argument("--report", type=Path, help="Save JSON validation report")
    args = parser.parse_args()
    report = verify(args.zarr, check_sources=args.check_sources, full_images=args.full_images)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
