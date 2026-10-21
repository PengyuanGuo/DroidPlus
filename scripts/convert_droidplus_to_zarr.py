#!/usr/bin/env python3
"""Convert corrected DROID Plus JSONL/JPEG recordings to atomic-policy zarr v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parents[1]
SIZE = 224
TEXT_LEN = 128


def encode_text(text: str) -> np.ndarray:
    # Avoid cutting through a UTF-8 character at the byte limit.
    raw = text.encode("utf-8")[:TEXT_LEN].decode("utf-8", errors="ignore").encode("utf-8")
    result = np.zeros(TEXT_LEN, dtype=np.uint8)
    result[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return result


def image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_roi(roi, width: int, height: int) -> list[int]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 3 or any(type(v) is not int for v in roi):
        raise ValueError(f"ROI must be integer [x, y, size], got {roi!r}")
    x, y, size = roi
    if x < 0 or y < 0 or size <= 0 or x + size > width or y + size > height:
        raise ValueError(f"ROI {roi} is outside image {width}x{height}")
    return list(roi)


def center_roi(width: int, height: int) -> list[int]:
    size = min(width, height)
    return [(width - size) // 2, (height - size) // 2, size]


def transform_points(points: np.ndarray, source_roi, output_roi, width: int, height: int) -> np.ndarray:
    """Map normalized annotation-region xy to normalized output-crop xy."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Every frame must contain one finite point_pixel pair")
    if ((points < 0) | (points > 1)).any():
        raise ValueError("Invalid/missing point_pixel; correct [-1,-1] segments before conversion")
    if source_roi is None:
        raw = points * [width, height]
    else:
        ax, ay, size = validate_roi(source_roi, width, height)
        raw = points * size + [ax, ay]
    x, y, size = validate_roi(output_roi, width, height)
    mapped = (raw - [x, y]) / size
    outside = np.flatnonzero(((mapped < -1e-7) | (mapped > 1 + 1e-7)).any(axis=1))
    if len(outside):
        raise ValueError(f"{len(outside)} point(s) outside output ROI {output_roi}; first frame {int(outside[0])}. Choose a crop containing every target; points are not clipped.")
    return np.clip(mapped, 0, 1).astype(np.float32)


def resized_rgb(image: np.ndarray, roi) -> np.ndarray:
    x, y, size = validate_roi(roi, image.shape[1], image.shape[0])
    return cv2.resize(image[y:y + size, x:x + size], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode(path: Path, output_roi=None, wrist_roi=None, task_prompt=None, require_reviewed=False) -> dict:
    rows = []
    input_bytes = path.read_bytes()
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    for line_number, line in enumerate(input_bytes.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid frame JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a frame object")
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty episode: {path}")
    episode = path.parent
    report_path = episode / "molmo_point_annotations.json"
    report_bytes = report_path.read_bytes()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    report = json.loads(report_bytes)
    meta_path = episode / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n = len(rows)
    states, actions, points, labels, prompts, timestamps, seqs = [], [], [], [], [], [], []
    paths = {"left": [], "wrist": []}
    segments = []
    for frame, row in enumerate(rows):
        for kind, target in (("state", states), ("action", actions)):
            try:
                values = np.asarray(row[kind]["positions"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: frame {frame}: missing/invalid {kind}.positions") from exc
            if values.shape != (8,) or not np.isfinite(values).all() or not 0 <= values[7] <= 1:
                raise ValueError(f"{path}: frame {frame}: {kind}.positions must be 8 finite values with native gripper in [0,1]")
            values[7] = 1.0 - values[7]  # Native DROID 0=open -> training 1=open.
            target.append(values.astype(np.float32))
        phase, prompt = row.get("phase_index"), row.get("phase_prompt")
        if type(phase) is not int or phase not in (0, 1) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: frame {frame}: invalid phase index/prompt")
        if not segments or segments[-1]["phase_index"] != phase:
            segments.append({"phase_index": phase, "phase_prompt": prompt, "start_frame": frame, "end_frame": frame})
        if segments[-1]["phase_prompt"] != prompt:
            raise ValueError(f"{path}: frame {frame}: prompt changes within phase segment")
        segments[-1]["end_frame"] = frame + 1
        point = row.get("point_pixel")
        if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (int, float) for v in point):
            raise ValueError(f"{path}: frame {frame}: missing/invalid point_pixel")
        points.append(point)
        labels.append(phase)
        prompts.append(encode_text(prompt))
        seq, timestamp = row.get("seq"), row.get("t_wall_s")
        if type(seq) is not int or seq < 0 or (seqs and seq != seqs[-1] + 1):
            raise ValueError(f"{path}: frame {frame}: nonconsecutive/invalid seq {seq!r}")
        if type(timestamp) not in (int, float) or not np.isfinite(timestamp) or (timestamps and timestamp <= timestamps[-1]):
            raise ValueError(f"{path}: frame {frame}: invalid or non-increasing timestamp")
        seqs.append(seq)
        timestamps.append(timestamp)
        for camera in paths:
            relative = row.get("images", {}).get(camera)
            if not isinstance(relative, str) or not (episode / relative).is_file():
                raise ValueError(f"{path}: frame {frame}: missing {camera} image {relative!r}")
            paths[camera].append(episode / relative)
    if [s["phase_index"] for s in segments] != [0, 1, 0, 1, 0, 1]:
        raise ValueError(f"{path}: expected six consecutive phase segments 0,1,0,1,0,1")
    saved_segments = report.get("segments", [])
    if len(saved_segments) != len(segments):
        raise ValueError(f"{path}: annotation report segments do not match frames")
    reviews = np.zeros(n, dtype=bool)
    points_array = np.asarray(points, dtype=np.float64)
    for actual, saved in zip(segments, saved_segments):
        if any(saved.get(key) != actual[key] for key in ("phase_index", "phase_prompt", "start_frame", "end_frame")):
            raise ValueError(f"{path}: stale annotation report phase boundaries or prompts")
        start, end = actual["start_frame"], actual["end_frame"]
        reported_point = np.asarray(saved.get("point_pixel", []), dtype=np.float64)
        if reported_point.shape != (2,) or not np.allclose(points_array[start:end], reported_point, rtol=0, atol=1e-6):
            raise ValueError(f"{path}: per-frame points disagree with annotation report")
        reviews[start:end] = bool(saved.get("needs_review", False))
    if require_reviewed and reviews.any():
        raise ValueError(f"{path}: contains segments still marked needs_review")
    source_roi = report.get("roi_xy_size")
    left_shape = image_rgb(paths["left"][0]).shape
    wrist_shape = image_rgb(paths["wrist"][0]).shape
    height, width = left_shape[:2]
    left_roi = validate_roi(output_roi if output_roi is not None else (source_roi if source_roi is not None else center_roi(width, height)), width, height)
    wrist_crop = validate_roi(wrist_roi if wrist_roi is not None else center_roi(wrist_shape[1], wrist_shape[0]), wrist_shape[1], wrist_shape[0])
    mapped = transform_points(points_array, source_roi, left_roi, width, height)
    instruction = task_prompt if task_prompt is not None else (meta.get("instruction") or meta.get("task"))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{path}: missing episode instruction/task; supply --task-prompt")
    deltas = np.diff(timestamps)
    observed_fps = float(1 / np.median(deltas)) if len(deltas) else None
    raw_steps = episode / "steps.jsonl"
    source_matches = raw_steps.is_file() and report.get("source_sha256") == sha256(raw_steps)
    notes = []
    if reviews.any():
        notes.append("Some point annotations remain marked needs_review")
    if meta.get("step_count") != n:
        notes.append("Original meta.step_count is stale; converted length comes from corrected JSONL")
    if not source_matches:
        notes.append("Raw steps.jsonl hash differs from annotation-time source; corrected steps_molmo.jsonl is the conversion input")
    return {
        "path": path, "sha256": input_sha, "report_path": report_path, "report_sha256": report_sha,
        "n": n, "paths": paths, "left_shape": left_shape, "wrist_shape": wrist_shape,
        "left_roi": left_roi, "wrist_roi": wrist_crop, "source_roi": source_roi,
        "deltas": deltas,
        "data": {
            "state": np.stack(states), "action": np.stack(actions), "point_pixel": mapped,
            "phase_label": np.asarray(labels, dtype=np.int8), "phase_prompt": np.stack(prompts),
            "prompt": np.repeat(encode_text(instruction)[None], n, axis=0),
            "timestamp": np.asarray(timestamps, dtype=np.float64) - timestamps[0],
            "source_seq": np.asarray(seqs, dtype=np.int64), "point_needs_review": reviews,
        },
        "manifest": {
            "source_steps": str(path.resolve()), "source_sha256": input_sha,
            "annotation_report": str(report_path.resolve()), "annotation_report_sha256": report_sha,
            "length": n, "instruction": instruction, "annotation_roi_xy_size": source_roi,
            "agent_roi_xy_size": left_roi, "wrist_roi_xy_size": wrist_crop,
            "observed_fps_median": observed_fps,
            "nominal_record_rate_hz": (meta.get("control") or {}).get("record_rate_hz"),
            "needs_review_frames": int(reviews.sum()), "notes": notes,
            "segments": [{**s, "needs_review": bool(reviews[s["start_frame"]]), "human_verified": bool(saved.get("human_verified", False)), "point_pixel": mapped[s["start_frame"]].tolist()} for s, saved in zip(segments, saved_segments)],
        },
    }


def preflight(root: Path, *, roi=None, wrist_roi=None, max_episodes=0, task_prompt=None, require_reviewed=False) -> list[dict]:
    paths = [root / "steps_molmo.jsonl"] if (root / "steps_molmo.jsonl").is_file() else sorted(root.glob("teleop_*/episode_*/steps_molmo.jsonl"))
    if not paths:
        paths = sorted(root.glob("episode_*/steps_molmo.jsonl"))
    if max_episodes:
        paths = paths[:max_episodes]
    if not paths:
        raise ValueError(f"No corrected steps_molmo.jsonl files found under {root}")
    # Never silently omit an unannotated episode.
    episode_dirs = [root] if (root / "steps_molmo.jsonl").is_file() else sorted({p.parent for p in root.rglob("episode_*/steps.jsonl")})
    missing = [p for p in episode_dirs if not (p / "steps_molmo.jsonl").exists()]
    if missing and not max_episodes:
        raise ValueError(f"Episodes missing corrected annotations: {missing}")
    episodes = []
    fixed_roi, fixed_wrist = roi, wrist_roi
    for path in paths:
        try:
            episode = load_episode(path, fixed_roi, fixed_wrist, task_prompt, require_reviewed)
        except (ValueError, KeyError, OSError) as exc:
            raise ValueError(f"Cannot convert {path}: {exc}") from exc
        if not episodes:
            fixed_roi, fixed_wrist = episode["left_roi"], episode["wrist_roi"]
        elif episode["left_shape"] != episodes[0]["left_shape"] or episode["wrist_shape"] != episodes[0]["wrist_shape"]:
            raise ValueError("All episodes must have the same raw camera dimensions for fixed-ROI deployment")
        episodes.append(episode)
    return episodes


def create_store(path: Path, n: int, ends: np.ndarray, chunk_frames: int):
    try:
        import numcodecs
        import zarr
    except ImportError as exc:
        raise RuntimeError("Install conversion dependencies: uv pip install --python .venv-molmo/bin/python 'zarr>=2.12,<3' 'numcodecs>=0.10,<0.16'") from exc
    if int(zarr.__version__.split(".")[0]) >= 3:
        root = zarr.open_group(str(path), mode="w", zarr_format=2)
    else:
        root = zarr.open_group(str(path), mode="w")
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    def create(group, name, shape, dtype, chunks):
        method = group.create_array if hasattr(group, "create_array") else group.create_dataset
        return method(name, shape=shape, dtype=dtype, chunks=chunks, compressor=compressor)

    data = root.create_group("data")
    arrays = {}
    for key, shape, dtype in (
        ("state", (n, 8), "float32"), ("action", (n, 8), "float32"),
        ("point_pixel", (n, 2), "float32"), ("phase_label", (n,), "int8"),
        ("phase_prompt", (n, TEXT_LEN), "uint8"), ("prompt", (n, TEXT_LEN), "uint8"),
        ("timestamp", (n,), "float64"), ("source_seq", (n,), "int64"),
        ("point_needs_review", (n,), "bool"),
    ):
        arrays[key] = create(data, key, shape, dtype, (1000,) + shape[1:])
    for camera in ("agent_view", "wrist_cam"):
        arrays[camera] = create(data, camera, (n, SIZE, SIZE, 3), "uint8", (chunk_frames, SIZE, SIZE, 3))
    meta = root.create_group("meta")
    create(meta, "episode_ends", ends.shape, "int64", (1000,))[:] = ends
    cache = root.create_group("_cache").create_group("224x224")
    for camera in ("agent_view", "wrist_cam"):
        arrays[camera + "_cache"] = create(cache.create_group(camera + ".zarr"), "images", (n, 3, SIZE, SIZE), "uint8", (chunk_frames, 3, SIZE, SIZE))
    return root, arrays


def convert(episodes: list[dict], output: Path, *, chunk_frames=100, overwrite=False) -> dict:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to back it up and replace")
    if output.exists() and (not output.is_dir() or not (output / ".zgroup").is_file()):
        raise ValueError("Refusing to replace a path that is not a zarr v2 directory")
    for episode in episodes:
        source = episode["path"].parent.resolve()
        if output == source or output in source.parents:
            raise ValueError("Output cannot replace a source episode or its ancestors")
    ends = np.cumsum([e["n"] for e in episodes], dtype=np.int64)
    n = int(ends[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        root, arrays = create_store(staging, n, ends, chunk_frames)
        offset = 0
        for index, episode in enumerate(episodes):
            if sha256(episode["path"]) != episode["sha256"] or sha256(episode["report_path"]) != episode["report_sha256"]:
                raise ValueError(f"Annotations changed during conversion: {episode['path']}")
            end = offset + episode["n"]
            for key, value in episode["data"].items():
                arrays[key][offset:end] = value
            for start in range(0, episode["n"], chunk_frames):
                stop = min(start + chunk_frames, episode["n"])
                for camera, array_name, crop, shape in (
                    ("left", "agent_view", episode["left_roi"], episode["left_shape"]),
                    ("wrist", "wrist_cam", episode["wrist_roi"], episode["wrist_shape"]),
                ):
                    batch = []
                    for frame in range(start, stop):
                        image = image_rgb(episode["paths"][camera][frame])
                        if image.shape != shape:
                            raise ValueError(f"Camera dimensions changed: {episode['paths'][camera][frame]}")
                        batch.append(resized_rgb(image, crop))
                    batch = np.stack(batch)
                    arrays[array_name][offset + start:offset + stop] = batch
                    arrays[array_name + "_cache"][offset + start:offset + stop] = batch.transpose(0, 3, 1, 2)
            episode["manifest"]["zarr_start"] = offset
            episode["manifest"]["zarr_end"] = end
            offset = end
            LOG.info("Converted %d/%d: %s/%s (%d frames)", index + 1, len(episodes), episode["path"].parent.parent.name, episode["path"].parent.name, episode["n"])
        deltas = np.concatenate([e["deltas"] for e in episodes])
        observed_fps = float(1 / np.median(deltas)) if len(deltas) else None
        attributes = {
            "source": "droid_plus_native", "schema_version": 1,
            "training_loader": "atomic_policy.dataset.molmobot_pnp_color_phase_dataset.MolmoBotPnPColorPhaseDataset",
            "image_size": SIZE, "agent_roi_xy_size": episodes[0]["left_roi"],
            "wrist_roi_xy_size": episodes[0]["wrist_roi"],
            "agent_raw_image_hw": list(episodes[0]["left_shape"][:2]),
            "wrist_raw_image_hw": list(episodes[0]["wrist_shape"][:2]),
            "image_color_space": "RGB", "data_image_layout": "NHWC", "cache_image_layout": "NCHW",
            "point_coordinate_system": "normalized xy in agent_view square ROI; origin top-left",
            "phase_label_mapping": {"grasp": 0, "place": 1},
            "gripper_semantics": "state/action last value: opening fraction, 0=closed, 1=open; converted as 1-native_fraction",
            "action_semantics": "7 absolute commanded joint targets [rad] + gripper opening command [0,1]",
            "temporal_sampling": "Original recorded frames, unchanged order, no resampling",
            "frequency_hz": observed_fps, "observed_fps_median": observed_fps,
            "n_frames": n, "n_episodes": len(episodes),
            "needs_review_frames": sum(int(e["data"]["point_needs_review"].sum()) for e in episodes),
            "manifest": "conversion_manifest.json",
        }
        root.attrs.update(attributes)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "output": str(output), **attributes, "episodes": [e["manifest"] for e in episodes]}
        (staging / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        # Verify the schema consumed by both PandaTabletop and MolmoBot loaders.
        assert json.loads((staging / ".zgroup").read_text())["zarr_format"] == 2
        for array in arrays.values():
            if array.shape[0] != n:
                raise ValueError("Mismatched output array lengths")
        if not np.array_equal(root["meta/episode_ends"][:], ends):
            raise ValueError("Incorrect episode boundaries")
        for episode in episodes:
            if sha256(episode["path"]) != episode["sha256"] or sha256(episode["report_path"]) != episode["report_sha256"]:
                raise ValueError(f"Annotations changed during conversion: {episode['path']}")
        backup = None
        if output.exists():
            backup = output.with_name(output.name + ".backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            LOG.info("Previous output preserved at %s", backup)
        return manifest
    except Exception:
        LOG.error("Conversion incomplete; staging files retained at %s. Source recordings were not modified.", staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--input-root", type=Path, default=PROJECT / "output_1017_phase_sorting")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/droidplus_sorting_annotated.zarr")
    parser.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "SIZE"), help="Fixed agent crop; default annotation ROI or centered square from the first camera frame")
    parser.add_argument("--wrist-roi", nargs=3, type=int, metavar=("X", "Y", "SIZE"), help="Fixed wrist crop; default centered square")
    parser.add_argument("--task-prompt", help="Optional episode instruction override")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    parser.add_argument("--image-chunk", type=int, default=100, help="Image/cache frames per chunk and processing batch")
    parser.add_argument("--require-reviewed", action="store_true", help="Reject segments still marked needs_review; default preserves flags and warns")
    parser.add_argument("--dry-run", action="store_true", help="Check annotations, points, image paths and first image dimensions without writing zarr")
    parser.add_argument("--overwrite", action="store_true", help="Back up and replace an existing output zarr")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.max_episodes < 0 or args.image_chunk <= 0:
        parser.error("max-episodes must be nonnegative and image-chunk must be positive")
    try:
        episodes = preflight(args.dataset_root.resolve(), roi=args.roi, wrist_roi=args.wrist_roi, max_episodes=args.max_episodes, task_prompt=args.task_prompt, require_reviewed=args.require_reviewed)
        n = sum(e["n"] for e in episodes)
        reviews = sum(bool(e["data"]["point_needs_review"].any()) for e in episodes)
        stale = sum(any("hash differs" in note for note in e["manifest"]["notes"]) for e in episodes)
        LOG.info("Validated %d episodes, %d frames. Agent ROI=%s; wrist ROI=%s", len(episodes), n, episodes[0]["left_roi"], episodes[0]["wrist_roi"])
        if reviews:
            LOG.warning("%d episodes have point review flags. Flags are preserved; use --require-reviewed to reject these episodes.", reviews)
        if stale:
            LOG.warning("%d raw steps.jsonl files differ from annotation-time hashes. Using valid corrected steps_molmo.jsonl; see manifest notes.", stale)
        deltas = np.concatenate([e["deltas"] for e in episodes])
        if len(deltas):
            LOG.info("Observed median cadence: %.3f Hz; no resampling", 1 / np.median(deltas))
        if args.dry_run:
            return
        convert(episodes, args.output, chunk_frames=args.image_chunk, overwrite=args.overwrite)
        LOG.info("Saved %d frames to %s", n, args.output.resolve())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
