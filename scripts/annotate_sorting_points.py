#!/usr/bin/env python3
"""
Ground real sorting recordings with MolmoPoint; preserve original steps.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parents[1]


def pointing_prompts(phase: int, text: str) -> tuple[str, str]:
    # Four-phase collection prompts omit the destination; use this dataset's
    # sorting rule, never infer a destination from the object's current position.
    prefixes = {0: "pre-grasp", 1: "grasp", 2: "pre-place", 3: "place"}
    if phase in prefixes:
        match = re.fullmatch(rf"{prefixes[phase]} (banana|orange|apple|salt bottle)", text)
        if match:
            obj = match[1]
            if phase < 2:
                return f"Point to the {obj} on the table", f"Point to the {obj}"
            color = "white" if obj == "salt bottle" else "blue"
            return f"Point to the inside of the {color} bin", f"Point to the {color} bin"
    if phase == 0:
        match = re.fullmatch(r"grasp (banana|orange|apple|salt bottle)", text)
        if match:
            obj = match[1]
            return f"Point to the {obj} on the table", f"Point to the {obj}"
    if phase == 1:
        match = re.fullmatch(
            r"place (banana|orange|apple|salt bottle) to the (blue|white) bin", text,
        )
        if match:
            color = match[2]
            return f"Point to the inside of the {color} bin", f"Point to the {color} bin"
    raise ValueError(f"Unsupported phase/prompt: {phase!r}, {text!r}")


def read_episode(path: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    segments = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON (remove plain-text order notes)") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a frame object")
        phase = row.get("phase_index")
        prompt = row.get("phase_prompt")
        if type(phase) is not int or phase not in (0, 1, 2, 3) or not isinstance(prompt, str):
            raise ValueError(f"{path}:{number}: missing/invalid phase_index or phase_prompt")
        primary, fallback = pointing_prompts(phase, prompt)
        if not segments or segments[-1]["phase_index"] != phase:
            segments.append({
                "phase_index": phase, "phase_prompt": prompt,
                "start_frame": len(rows), "end_frame": len(rows),
                "molmo_prompt": primary, "fallback_prompt": fallback,
            })
        if segments[-1]["phase_prompt"] != prompt:
            raise ValueError(f"{path}:{number}: prompt changed without a phase boundary")
        segments[-1]["end_frame"] = len(rows) + 1
        relative = row.get("images", {}).get("left")
        if not isinstance(relative, str) or not (path.parent / relative).is_file():
            raise ValueError(f"{path}:{number}: missing left-camera image {relative!r}")
        rows.append(row)
    four_phase = bool(segments and segments[0]["phase_prompt"].startswith("pre-grasp "))
    if four_phase:
        labels = [segment["phase_index"] for segment in segments]
        if len(labels) % 4 or labels != [0, 1, 2, 3] * (len(labels) // 4):
            raise ValueError(f"{path}: expected complete four-phase cycles 0,1,2,3")
        for start in range(0, len(segments), 4):
            obj = segments[start]["phase_prompt"].removeprefix("pre-grasp ")
            expected = [f"{prefix} {obj}" for prefix in ("pre-grasp", "grasp", "pre-place", "place")]
            if [s["phase_prompt"] for s in segments[start:start + 4]] != expected:
                raise ValueError(f"{path}: four-phase object/prompt mismatch")
        return rows, segments
    if [segment["phase_index"] for segment in segments] != [0, 1, 0, 1, 0, 1]:
        raise ValueError(f"{path}: expected six consecutive phase segments 0,1,0,1,0,1")
    for grasp, place in zip(segments[::2], segments[1::2]):
        obj = grasp["phase_prompt"].removeprefix("grasp ")
        color = "white" if obj == "salt bottle" else "blue"
        if place["phase_prompt"] != f"place {obj} to the {color} bin":
            raise ValueError(f"{path}: grasp/place object or destination mismatch")
    return rows, segments


def sample_indices(start: int, end: int, count: int, skip: int) -> list[int]:
    """Spread unique samples across the first quarter; cap skip for short phases."""
    stop = min(end, start + max(1, int(np.ceil((end - start) * 0.25))))
    first = min(start + skip, stop - 1)
    return np.unique(np.linspace(first, stop - 1, min(count, stop - first), dtype=int)).tolist()


def read_rgb(path: Path, roi: list[int] | None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    if roi is not None:
        x, y, size = roi
        height, width = bgr.shape[:2]
        if x < 0 or y < 0 or size <= 0 or x + size > width or y + size > height:
            raise ValueError(f"ROI {roi} is outside {width}x{height} image: {path}")
        bgr = bgr[y:y + size, x:x + size]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ground_segment(grounder, episode: Path, rows: list[dict], segment: dict, args) -> dict:
    samples = sample_indices(segment["start_frame"], segment["end_frame"], args.samples, args.skip_first_frames)
    points = []
    attempts = []
    for index in samples:
        rgb = read_rgb(episode / rows[index]["images"]["left"], args.roi)
        height, width = rgb.shape[:2]
        attempt = {"frame_index": index, "seq": rows[index]["seq"], "errors": []}
        for prompt in (segment["molmo_prompt"], segment["fallback_prompt"]):
            try:
                pixel = np.asarray(grounder._point(rgb, prompt), dtype=np.float32)
                if pixel.shape != (2,) or not np.isfinite(pixel).all():
                    raise ValueError(f"Invalid point: {pixel}")
                if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                    raise ValueError(f"Point outside image: {pixel}")
                normalized = pixel / np.array([width, height], dtype=np.float32)
                points.append(normalized)
                attempt.update({"prompt_used": prompt, "point_pixel": normalized.tolist()})
                break
            except Exception as exc:
                message = str(exc).lower()
                # Stop on device failures; never turn a broken GPU into failed labels.
                if isinstance(exc, MemoryError) or any(word in message for word in (
                    "out of memory", "cuda error", "device-side assert", "meta tensor",
                )):
                    raise
                attempt["errors"].append({"prompt": prompt, "error": str(exc)})
        attempts.append(attempt)
    kept = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(kept) > 2:
        distances = np.linalg.norm(kept - np.median(kept, axis=0), axis=1)
        threshold = float(np.median(distances) + 2 * np.std(distances))
        kept = kept[distances <= threshold + 1e-8]
    required = min(args.min_valid_samples, len(samples))
    success = len(kept) >= required
    point = kept.mean(axis=0).tolist() if success else [-1.0, -1.0]
    return {
        **segment, "sample_indices": samples, "samples": attempts,
        "valid_predictions": len(points), "inliers": len(kept),
        "required_inliers": required, "point_pixel": point, "needs_review": not success,
    }


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_episode(path: Path, rows: list[dict], segments: list[dict], args) -> None:
    preview_dir = path.parent / "molmo_previews"
    preview_dir.mkdir(exist_ok=True)
    for number, segment in enumerate(segments):
        for row in rows[segment["start_frame"]:segment["end_frame"]]:
            row["point_pixel"] = segment["point_pixel"]
        # Preview uses the exact input crop and the final phase-level point.
        row = rows[segment["sample_indices"][0]]
        image = read_rgb(path.parent / row["images"]["left"], args.roi)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        x, y = segment["point_pixel"]
        if x >= 0:
            pixel = (min(width - 1, round(x * width)), min(height - 1, round(y * height)))
            cv2.drawMarker(image, pixel, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        text = segment["phase_prompt"] if x >= 0 else "FAILED: " + segment["phase_prompt"]
        cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if not cv2.imwrite(str(preview_dir / f"segment_{number:02d}.jpg"), image):
            raise RuntimeError("Failed to write point preview")
    report = {
        "source_steps": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint, "dtype": args.dtype, "load_in_4bit": args.load_in_4bit,
        "max_image_size": args.max_image_size, "roi_xy_size": args.roi,
        "point_coordinate_system": "normalized xy on cropped left RGB image" if args.roi else "normalized xy on full left RGB image",
        "sampling": "first 25% of each consecutive phase segment",
        "needs_review": any(s["needs_review"] for s in segments), "segments": segments,
    }
    atomic_text(path.parent / "steps_molmo.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    atomic_text(path.parent / "molmo_point_annotations.json", json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "output_1017_phase_sorting")
    parser.add_argument("--atomic-policy-root", type=Path, default=PROJECT / "atomic_policy-main")
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "SIZE"), help="Fixed square crop in raw image pixels")
    geometry.add_argument("--full-frame", action="store_true", help="Point in full camera coordinates; useful for an initial test")
    parser.add_argument("--checkpoint", default="allenai/MolmoPoint-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=448)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-first-frames", type=int, default=2)
    parser.add_argument("--min-valid-samples", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    parser.add_argument("--dry-run", action="store_true", help="Validate labels, image samples and ROI without loading Molmo or saving annotations")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously generated annotation files; originals are never modified")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if min(args.samples, args.min_valid_samples, args.max_image_size) <= 0 or args.skip_first_frames < 0 or args.max_episodes < 0:
        parser.error("samples, min-valid-samples, max-image-size must be positive; skip/max-episodes cannot be negative")
    if args.min_valid_samples > args.samples:
        parser.error("min-valid-samples cannot exceed samples")
    root = args.dataset_root.resolve()
    paths = [root / "steps.jsonl"] if (root / "steps.jsonl").is_file() else sorted(root.rglob("episode_*/steps.jsonl"))
    if args.max_episodes:
        paths = paths[:args.max_episodes]
    if not paths:
        parser.error(f"No episode steps.jsonl files under {root}")
    total_frames = 0
    total_samples = 0
    # Validate the entire selection before loading the model or writing files.
    for path in paths:
        rows, segments = read_episode(path)
        if not args.dry_run and not args.overwrite:
            for output in ("steps_molmo.jsonl", "molmo_point_annotations.json"):
                if (path.parent / output).exists():
                    parser.error(f"{path.parent / output} exists; use --overwrite to regenerate")
        for segment in segments:
            for index in sample_indices(segment["start_frame"], segment["end_frame"], args.samples, args.skip_first_frames):
                read_rgb(path.parent / rows[index]["images"]["left"], args.roi)
                total_samples += 1
        total_frames += len(rows)
    LOG.info("Validated %d episodes, %d frames, %d image samples", len(paths), total_frames, total_samples)
    if args.dry_run:
        return
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("A working CUDA GPU is required by the repository's MolmoPoint grounder")
    sys.path.insert(0, str(args.atomic_policy_root.resolve()))
    from atomic_policy.vlm.grounder_molmo import MolmoPointGrounder

    grounder = MolmoPointGrounder(
        checkpoint=args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype),
        load_in_4bit=args.load_in_4bit, max_image_size=args.max_image_size,
    )
    failed_segments = 0
    for episode_index, path in enumerate(paths):
        rows, segments = read_episode(path)
        results = []
        for number, segment in enumerate(segments):
            LOG.info("Episode %d/%d %s segment %d: %s", episode_index + 1, len(paths), path.parent.name, number, segment["phase_prompt"])
            result = ground_segment(grounder, path.parent, rows, segment, args)
            failed_segments += result["needs_review"]
            results.append(result)
            LOG.info("  point=%s, inliers=%d, needs_review=%s", result["point_pixel"], result["inliers"], result["needs_review"])
        write_episode(path, rows, results, args)
    LOG.info("Finished: %d episodes, %d failed segments. Inspect molmo_previews/ in each episode.", len(paths), failed_segments)
    if failed_segments:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
