#!/usr/bin/env python3
"""Replace one consecutive phase segment's points using a chosen left-camera image."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .annotate_sorting_points import PROJECT, atomic_text, pointing_prompts, read_episode, read_rgb
else:
    from annotate_sorting_points import PROJECT, atomic_text, pointing_prompts, read_episode, read_rgb

LOG = logging.getLogger(__name__)


def prepare_correction(image: Path, expected_segment: int | None = None) -> dict:
    image = image.resolve(strict=True)
    requested_image = image
    episode = image.parent.parent
    source = episode / "steps.jsonl"
    annotated = episode / "steps_molmo.jsonl"
    report_path = episode / "molmo_point_annotations.json"
    rows, source_segments = read_episode(source)
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if image.parent.name == "molmo_previews":
        match = re.fullmatch(r"segment_(\d{2})\.jpg", image.name)
        if not match or not 0 <= int(match[1]) < len(source_segments):
            raise ValueError("Preview must be named segment_00.jpg through segment_05.jpg")
        preview_segment = int(match[1])
        if expected_segment is not None and expected_segment != preview_segment:
            raise ValueError(f"Preview is segment {preview_segment:02d}, not requested segment {expected_segment:02d}")
        expected_segment = preview_segment
        # Use the unmarked original behind the preview, preserving the recorded crop.
        # The painted cross/text must not become part of the model's input.
        indices = report["segments"][preview_segment].get("sample_indices", [])
        if not indices or type(indices[0]) is not int or not 0 <= indices[0] < len(rows):
            raise ValueError("Report is missing a valid preview source frame")
        image = (episode / rows[indices[0]]["images"]["left"]).resolve(strict=True)
    matches = [i for i, row in enumerate(rows) if (episode / row["images"]["left"]).resolve() == image]
    if len(matches) != 1:
        raise ValueError("Image must match exactly one left-camera image reference in this episode")
    frame_index = matches[0]
    segment_index = next(i for i, s in enumerate(source_segments) if s["start_frame"] <= frame_index < s["end_frame"])
    if expected_segment is not None and segment_index != expected_segment:
        raise ValueError(f"Image belongs to segment {segment_index:02d}, not requested segment {expected_segment:02d}")
    source_bytes = source.read_bytes()
    annotated_bytes = annotated.read_bytes()
    if report.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("Source steps.jsonl changed since annotation; regenerate annotations before correction")
    if len(report["segments"]) != len(source_segments):
        raise ValueError("Report phase segments do not match current steps.jsonl")
    for saved, actual in zip(report["segments"], source_segments):
        for field in ("phase_index", "phase_prompt", "start_frame", "end_frame"):
            if saved[field] != actual[field]:
                raise ValueError(f"Report segment {field} disagrees with current steps.jsonl")
    data_lines = [line for line in annotated_bytes.decode().splitlines() if line.strip()]
    if len(data_lines) != len(rows):
        raise ValueError("steps_molmo.jsonl frame count does not match source")
    for index, (line, original) in enumerate(zip(data_lines, rows)):
        row = json.loads(line)
        if "point_pixel" not in row:
            raise ValueError(f"Annotated frame {index} is missing point_pixel")
        row.pop("point_pixel")
        if row != original:
            raise ValueError(f"Annotated frame {index} differs from source beyond point_pixel")
    roi = report.get("roi_xy_size")
    if roi is not None and (not isinstance(roi, list) or len(roi) != 3 or any(type(v) is not int for v in roi)):
        raise ValueError("Invalid ROI in annotation report")
    rgb = read_rgb(image, roi)
    preview_path = episode / "molmo_previews" / f"segment_{segment_index:02d}.jpg"
    expected_files = {
        source: source_bytes, annotated: annotated_bytes, report_path: report_bytes,
        preview_path: preview_path.read_bytes() if preview_path.exists() else None,
    }
    return {
        "image": image, "requested_image": requested_image, "episode": episode, "source": source, "annotated": annotated,
        "report_path": report_path, "preview_path": preview_path, "report": report,
        "segment_index": segment_index, "frame_index": frame_index, "rows": rows,
        "segment": source_segments[segment_index], "rgb": rgb, "expected_files": expected_files,
    }


def predict_point(grounder, rgb: np.ndarray, prompts: list[str]) -> dict:
    height, width = rgb.shape[:2]
    errors = []
    for prompt in dict.fromkeys(prompts):
        try:
            pixel = np.asarray(grounder._point(rgb, prompt), dtype=np.float64)
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise ValueError(f"Invalid point {pixel}")
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                raise ValueError(f"Point outside image: {pixel}")
            return {
                "pixel_xy": pixel.tolist(),
                "point_pixel": (pixel / [width, height]).tolist(),
                "prompt_used": prompt, "errors": errors,
                "primary_prompt": prompts[0], "fallback_prompt": prompts[1] if len(prompts) > 1 else "",
            }
        except Exception as exc:
            if isinstance(exc, MemoryError) or any(word in str(exc).lower() for word in (
                "out of memory", "cuda error", "device-side assert", "meta tensor",
            )):
                raise
            errors.append({"prompt": prompt, "error": str(exc)})
    raise ValueError(f"All correction prompts failed; existing files are unchanged. Errors: {errors}")


def correction_prompts(segment: dict, primary: str | None, fallback: str | None) -> list[str]:
    default_primary, default_fallback = pointing_prompts(segment["phase_index"], segment["phase_prompt"])
    chosen_primary = default_primary if primary is None else primary.strip()
    if not chosen_primary:
        raise ValueError("Primary prompt cannot be empty")
    # A custom target (e.g. blue bottle) must not fall back to the old target.
    chosen_fallback = fallback.strip() if fallback is not None else (default_fallback if primary is None else "")
    return list(dict.fromkeys(p for p in (chosen_primary, chosen_fallback) if p))


def apply_correction(
    context: dict, prediction: dict, settings: dict, *,
    method: str = "single_frame_molmo_correction", human_verified: bool = False,
) -> Path:
    """Update only the selected segment and its preview, retaining backup/history."""
    point = np.asarray(prediction["point_pixel"], dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or not ((point >= 0).all() and (point < 1).all()):
        raise ValueError("Correction point must be finite normalized xy in [0, 1)")
    point_list = point.tolist()
    index = context["segment_index"]
    segment = context["segment"]
    start, end = segment["start_frame"], segment["end_frame"]
    report = copy.deepcopy(context["report"])
    old_segment = copy.deepcopy(report["segments"][index])
    sample = {
        "frame_index": context["frame_index"],
        "seq": context["rows"][context["frame_index"]]["seq"],
        "errors": prediction["errors"], "prompt_used": prediction["prompt_used"],
        "point_pixel": point_list,
    }
    primary, fallback = pointing_prompts(segment["phase_index"], segment["phase_prompt"])
    if human_verified:
        primary = old_segment.get("molmo_prompt", primary)
        fallback = old_segment.get("fallback_prompt", fallback)
    primary = prediction.get("primary_prompt", primary)
    fallback = prediction.get("fallback_prompt", fallback)
    replacement = {
        **segment, "molmo_prompt": primary, "fallback_prompt": fallback,
        "method": method, "sample_indices": [context["frame_index"]],
        "samples": [sample], "valid_predictions": 1, "inliers": 1, "required_inliers": 1,
        "point_pixel": point_list, "needs_review": not human_verified,
        "review_reasons": [] if human_verified else ["single_frame_correction_requires_visual_review"],
        "model_settings": settings,
    }
    if human_verified:
        replacement["human_verified"] = True
    report["segments"][index] = replacement
    report["needs_review"] = any(s["needs_review"] for s in report["segments"])
    # Keep every unselected row byte-for-byte, including spacing and line endings.
    original = context["expected_files"][context["annotated"]]
    lines = original.decode().splitlines(keepends=True)
    frame = 0
    point_pattern = re.compile(r'"point_pixel"\s*:\s*\[[^\]]*\]')
    for line_number, line in enumerate(lines):
        if not line.strip():
            continue
        if start <= frame < end:
            if len(point_pattern.findall(line)) != 1:
                raise ValueError("Expected exactly one point_pixel array per annotated frame")
            updated = point_pattern.sub(lambda _: '"point_pixel": ' + json.dumps(point_list), line, count=1)
            before, after = json.loads(line), json.loads(updated)
            before.pop("point_pixel")
            assert after.pop("point_pixel") == point_list and before == after
            lines[line_number] = updated
        frame += 1
    if frame != len(context["rows"]):
        raise ValueError("Unexpected annotation row count")
    preview = cv2.cvtColor(context["rgb"], cv2.COLOR_RGB2BGR)
    height, width = preview.shape[:2]
    pixel = (min(width - 1, round(point[0] * width)), min(height - 1, round(point[1] * height)))
    cv2.drawMarker(preview, pixel, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.putText(preview, segment["phase_prompt"], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, encoded = cv2.imencode(".jpg", preview)
    if not ok:
        raise RuntimeError("Could not encode corrected preview")
    # Inference can take a while: reject concurrent edits instead of overwriting them.
    for path, expected in context["expected_files"].items():
        current = path.read_bytes() if path.exists() else None
        if current != expected:
            raise ValueError(f"File changed during correction: {path}; no files overwritten")
    backup_dir = context["episode"] / "molmo_correction_backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    backup = backup_dir / f"segment_{index:02d}_{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}.tar.gz"
    with tarfile.open(backup, "w:gz") as archive:
        for path in (context["annotated"], context["report_path"], context["preview_path"]):
            if path.exists():
                archive.add(path, arcname=str(path.relative_to(context["episode"])))
    with tarfile.open(backup, "r:gz") as archive:
        for path in (context["annotated"], context["report_path"], context["preview_path"]):
            expected = context["expected_files"][path]
            if expected is not None:
                assert archive.extractfile(str(path.relative_to(context["episode"]))).read() == expected
    report.setdefault("corrections", []).append({
        "time_utc": timestamp.isoformat(), "segment_index": index,
        "image": str(context["image"]), "frame_index": context["frame_index"],
        "requested_image": str(context.get("requested_image", context["image"])),
        "previous_segment": old_segment, "new_point_pixel": point_list,
        "prediction": prediction, "model_settings": settings, "backup": str(backup),
        "method": method, "human_verified": human_verified,
    })
    new_report = json.dumps(report, indent=2, allow_nan=False) + "\n"
    context["preview_path"].parent.mkdir(exist_ok=True)
    temporary_preview = context["preview_path"].with_suffix(".jpg.tmp")
    try:
        atomic_text(context["annotated"], "".join(lines))
        atomic_text(context["report_path"], new_report)
        temporary_preview.write_bytes(encoded.tobytes())
        temporary_preview.replace(context["preview_path"])
    except Exception:
        # Roll back the episode if one of the related output writes fails.
        for path in (context["annotated"], context["report_path"], context["preview_path"]):
            expected = context["expected_files"][path]
            if expected is not None:
                path.write_bytes(expected)
        raise
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Raw left-camera JPEG or molmo_previews/segment_NN.jpg (resolved to its unmarked source)")
    parser.add_argument("--segment", type=int, choices=range(6), help="Optional segment number 0..5 as a check (not phase_index)")
    parser.add_argument("--prompt", help="Custom Molmo instruction; disables the old fallback unless --fallback-prompt is supplied")
    parser.add_argument("--fallback-prompt", help="Custom fallback; pass an empty string to disable it")
    parser.add_argument("--atomic-policy-root", type=Path, default=PROJECT / "atomic_policy-main")
    parser.add_argument("--checkpoint", help="Default: inherit existing annotation checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), help="Default: inherit annotation dtype")
    parser.add_argument("--max-image-size", type=int, help="Default: inherit annotation resolution")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None, help="Default: inherit annotation quantization")
    parser.add_argument("--dry-run", action="store_true", help="Locate/validate segment without loading model or writing files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        context = prepare_correction(args.image, args.segment)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    report = context["report"]
    settings = {
        "checkpoint": args.checkpoint or report["checkpoint"],
        "dtype": args.dtype or report["dtype"],
        "max_image_size": args.max_image_size if args.max_image_size is not None else report["max_image_size"],
        "load_in_4bit": args.load_in_4bit if args.load_in_4bit is not None else report["load_in_4bit"],
        "roi_xy_size": report.get("roi_xy_size"),
    }
    if settings["dtype"] not in ("bfloat16", "float32") or settings["max_image_size"] <= 0:
        parser.error("Invalid model dtype or max-image-size")
    try:
        prompts = correction_prompts(context["segment"], args.prompt, args.fallback_prompt)
    except ValueError as exc:
        parser.error(str(exc))
    start, end = context["segment"]["start_frame"], context["segment"]["end_frame"]
    LOG.info("Image matches frame %d, segment %02d (%s), rows [%d:%d]: %d frames", context["frame_index"], context["segment_index"], context["segment"]["phase_prompt"], start, end, end - start)
    LOG.info("Unmarked model input image: %s", context["image"])
    LOG.info("ROI: %s; prompts: %s; preview: %s", settings["roi_xy_size"], prompts, context["preview_path"])
    if args.dry_run:
        return
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("A working CUDA GPU is required for MolmoPoint")
    # Verify GPU kernels before loading the large model (Blackwell needs CUDA 12.8+ wheels).
    (torch.ones(1, device=args.device) + 1).cpu()
    sys.path.insert(0, str(args.atomic_policy_root.resolve()))
    from atomic_policy.vlm.grounder_molmo import MolmoPointGrounder

    grounder = MolmoPointGrounder(
        checkpoint=settings["checkpoint"], device=args.device, dtype=getattr(torch, settings["dtype"]),
        load_in_4bit=settings["load_in_4bit"], max_image_size=settings["max_image_size"],
    )
    prediction = predict_point(grounder, context["rgb"], prompts)
    LOG.info("Predicted pixel xy=%s; normalized xy=%s", prediction["pixel_xy"], prediction["point_pixel"])
    backup = apply_correction(context, prediction, settings)
    LOG.info("Updated only segment %02d (%d frames). Backup: %s", context["segment_index"], end - start, backup)
    LOG.info("Inspect %s. The new single-frame prediction is flagged needs_review=true until visually reviewed.", context["preview_path"])


if __name__ == "__main__":
    main()
