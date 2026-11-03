#!/usr/bin/env python3
"""Per-frame wrist-camera target points for annotated teleop episodes (contact-anchored hindsight labels).

Each target window (e.g. pre-grasp + grasp -> letter cube; pre-place + place -> slot) gets the 3D fingertip
centre at its contact (gripper close / open command) and that point is projected into every wrist frame with the
calibrated wrist camera (scripts/calibrate_wrist_camera.py) and joint-angle FK. Objects ride with the gripper after
the close. Nothing is tracked, so occlusion by the fingers or the held cube does not matter.

Adds to every row of steps_molmo.jsonl (the converter input; other fields untouched, first run keeps
steps_molmo.jsonl.wrist_orig):
    wrist_point_pixel    [x, y] normalized on the raw wrist image, [-1, -1] when not visible
    wrist_point_visible  true if the target is in front of the camera and inside the wrist image
    wrist_point_held     true while the target object is in the gripper
and a "wrist" block (calibration, targets, QA) to molmo_point_annotations.json.
QA: --sam2-check compares each object window with the SAM 2 tracks cached by the calibration and marks windows
whose median disagreement exceeds --review-px as needs_review.

Usage:
    python scripts/annotate_wrist_points.py --dataset-root DATASET [--calibration DATASET/wrist_camera_calibration.json]
        [--previews 3] [--sam2-check] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_preview(ep: Path, rows, lab, out: Path, fps: float, stride: int = 1) -> None:
    import cv2
    W, H = 672, 376
    writer = None
    for f in range(0, len(rows), stride):
        wrist = cv2.imread(str(ep / rows[f]["images"]["wrist"]))
        left = cv2.imread(str(ep / rows[f]["images"]["left"]))
        x, y = rows[f]["point_pixel"]
        cv2.circle(left, (int(x * left.shape[1]), int(y * left.shape[0])), 7, (0, 0, 255), 2)
        if lab["visible"][f]:
            u, v = lab["uv"][f] * [W, H]
            color = (255, 160, 0) if lab["held"][f] else (0, 0, 255)
            cv2.circle(wrist, (int(u), int(v)), 9, color, 2)
            cv2.circle(wrist, (int(u), int(v)), 2, color, -1)
        txt = f"f{f} {rows[f]['phase_prompt']}" + (" [held]" if lab["held"][f] else "") + ("" if lab["visible"][f] else " [not visible]")
        frame = np.hstack([cv2.resize(left, (W, H)), cv2.resize(wrist, (W, H))])  # left | wrist
        cv2.putText(frame, txt, (8, 24), 0, 0.7, (0, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer is not None:
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None, help="default: DATASET/wrist_camera_calibration.json")
    ap.add_argument("--atomic-policy-root", type=Path, default=PROJECT.parent / "atomic_policy")
    ap.add_argument("--place-drop", type=float, default=0.0, help="lower place targets by this much (m) below the release fingertip")
    ap.add_argument("--previews", type=int, default=0, help="write wrist_previews.mp4 for the first N episodes")
    ap.add_argument("--sam2-check", action="store_true", help="QA object windows against cached SAM 2 tracks")
    ap.add_argument("--review-px", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(args.atomic_policy_root.resolve()))
    from atomic_policy.wrist.geometry import WristCamera, panda_fk_batch
    from atomic_policy.wrist.targets import wrist_labels

    root = args.dataset_root.resolve()
    cal_path = (args.calibration or root / "wrist_camera_calibration.json").resolve()
    cam = WristCamera.load(cal_path)
    tracks = {}
    if args.sam2_check:
        cache = root / "wrist_calibration_tracks.json"
        tracks = json.loads(cache.read_text()) if cache.exists() else {}
    episodes = sorted(p.parent for p in root.glob("*/episode_*/steps_molmo.jsonl"))
    totals = {"episodes": 0, "frames": 0, "visible": 0, "windows": 0, "review": 0, "checked": 0}
    qa_errors = []
    for n_ep, ep in enumerate(episodes):
        path = ep / "steps_molmo.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        q = np.array([r["state"]["positions"][:7] for r in rows])
        cmd = np.array([r["action"]["positions"][7] for r in rows]) > 0.5
        phases = [r["phase_index"] for r in rows]
        T = panda_fk_batch(q)
        lab = wrist_labels(cam, T, cmd, phases, args.place_drop)
        win_info = []
        for i, w in enumerate(lab["windows"]):
            info = {"segments": w.segments, "phases": [lab["segments"][k][0] for k in w.segments],
                    "start_frame": w.start, "end_frame": w.end, "role": w.role, "contact_frame": w.contact,
                    "release_frame": w.release, "target_base_m": [round(v, 5) for v in w.point_base],
                    "visible_frames": int(lab["visible"][w.start:w.end].sum()), "needs_review": False}
            key = f"{ep.relative_to(root)}|{w.contact}"
            if w.role == "object" and key in tracks:
                tr = tracks[key]
                errs = [np.hypot(*(lab["uv"][f] * [cam.width, cam.height] - tr[str(f)][:2])) for f in range(w.start, w.contact - 2)
                        if tr.get(str(f)) and tr[str(f)][2] >= 80 and lab["visible"][f]]
                if errs:
                    info["sam2_check_median_px"] = float(np.median(errs)); qa_errors.extend(errs); totals["checked"] += 1
                    info["needs_review"] = bool(np.median(errs) > args.review_px)
            totals["review"] += info["needs_review"]
            win_info.append(info)
        for f, r in enumerate(rows):
            vis = bool(lab["visible"][f])
            r["wrist_point_pixel"] = [round(float(v), 6) for v in lab["uv"][f]] if vis else [-1.0, -1.0]
            r["wrist_point_visible"] = vis
            r["wrist_point_held"] = bool(lab["held"][f])
        totals["episodes"] += 1; totals["frames"] += len(rows); totals["visible"] += int(lab["visible"].sum()); totals["windows"] += len(win_info)
        if not args.dry_run:
            orig = path.with_name(path.name + ".wrist_orig")
            if not orig.exists():
                shutil.copy2(path, orig)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            ann_path = ep / "molmo_point_annotations.json"
            ann = json.loads(ann_path.read_text())
            ann["wrist"] = {"method": "contact-anchored hindsight projection (atomic_policy.wrist.targets)",
                            "calibration": str(cal_path), "calibration_sha256": sha256(cal_path),
                            "place_drop_m": args.place_drop, "point_coordinate_system": "normalized xy on full wrist RGB image",
                            "steps_sha256": sha256(path), "windows": win_info}
            ann_path.write_text(json.dumps(ann, indent=2) + "\n")
            if n_ep < args.previews:
                fps = 1.0 / np.median(np.diff([r["t_wall_s"] for r in rows]))
                write_preview(ep, rows, lab, ep / "wrist_previews.mp4", fps)
        bad = [i for i, w in enumerate(win_info) if w["needs_review"]]
        print(f"{ep.relative_to(root)}: {len(win_info)} targets, wrist point visible on {lab['visible'].mean():.0%} of frames"
              + (f", REVIEW windows {bad}" if bad else ""), flush=True)
    print(f"\n{totals['episodes']} episodes, {totals['frames']} frames, visible {totals['visible'] / max(1, totals['frames']):.0%}, "
          f"{totals['windows']} target windows, {totals['review']} need review" + (" (dry run, nothing written)" if args.dry_run else ""))
    if qa_errors:
        e = np.array(qa_errors)
        print(f"SAM 2 check on {totals['checked']} object windows: median {np.median(e):.1f}px, p90 {np.percentile(e, 90):.1f}px")


if __name__ == "__main__":
    main()
