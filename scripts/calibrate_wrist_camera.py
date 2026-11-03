#!/usr/bin/env python3
"""Calibrate the wrist camera from annotated teleop episodes (no calibration board).

For every object grasp (gripper command closes), the grasped object sits between the fingertips at a fixed wrist
pixel (--seed-pixel). SAM 2 is seeded there a moment after the close and tracks the object backward to the start of
its approach. Its 3D centre is the fingertip centre at the close, so these tracks fix the camera intrinsics, the
camera pose on the flange and the fingertip offset (atomic_policy.wrist.calibration.fit_wrist_camera).

Writes <dataset>/wrist_camera_calibration.json (WristCamera + fit statistics + target heights per phase, which
deployment uses to lift a MolmoPoint pixel to 3D). SAM 2 tracks are cached in <dataset>/wrist_calibration_tracks.json.

Usage (DroidPlus/.venv-molmo, SAM 2 installed):
    python scripts/calibrate_wrist_camera.py --dataset-root data/annotated/test_spacemouse_crossword_mars_4phase_wrist_cam \
        --atomic-policy-root ../atomic_policy [--per-session 2] [--sam2-model large]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def load_episode(ep: Path):
    rows = [json.loads(l) for l in (ep / "steps_molmo.jsonl").read_text().splitlines() if l.strip()]
    q = np.array([r["state"]["positions"][:7] for r in rows])
    cmd = np.array([r["action"]["positions"][7] for r in rows]) > 0.5
    phases = [r["phase_index"] for r in rows]
    return rows, q, cmd, phases


def sam2_backward(pred, ep: Path, rows, start: int, seed: int, seed_px, torch):
    """Object centroid (px) for frames start..seed, tracked backward from the grasp seed."""
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, f in enumerate(range(start, seed + 1)):
            os.symlink((ep / rows[f]["images"]["wrist"]).resolve(), f"{tmp}/{i:05d}.jpg")
        state = pred.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
        w, h = state["video_width"], state["video_height"]
        pts = np.array([[seed_px[0] * w, seed_px[1] * h], [0.53 * w, 0.62 * h], [0.75 * w, 0.62 * h]], np.float32)
        pred.add_new_points_or_box(state, frame_idx=seed - start, obj_id=1, points=pts, labels=np.array([1, 0, 0], np.int32))
        out = {}
        for i, _, logits in pred.propagate_in_video(state, start_frame_idx=seed - start, reverse=True):
            m = (logits[0, 0] > 0).cpu().numpy()
            ys, xs = np.nonzero(m)
            out[start + i] = [float(xs.mean()), float(ys.mean()), int(m.sum())] if len(xs) else None
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--atomic-policy-root", type=Path, default=PROJECT.parent / "atomic_policy")
    ap.add_argument("--per-session", type=int, default=2, help="episodes per session used for the fit (0 = all)")
    ap.add_argument("--sam2-model", default="large", choices=["tiny", "small", "base_plus", "large"])
    ap.add_argument("--sam2-checkpoints", default="/workspace/pengyuan/checkpoints/sam2")
    ap.add_argument("--seed-pixel", nargs=2, type=float, default=[0.62, 0.45], help="held object in the wrist image (normalized)")
    ap.add_argument("--seed-delay", type=int, default=15, help="frames after the close command to seed SAM 2")
    ap.add_argument("--min-area", type=int, default=80, help="drop frames whose SAM 2 mask is smaller (px)")
    ap.add_argument("--gpu-mem-gb", type=float, default=0, help="cap this process's GPU memory (0 = no cap)")
    ap.add_argument("--tcp-z", type=float, default=0.15,
                    help="fingertip distance along the flange axis (m); unobservable from xyz_yaw teleop, so fixed. <=0 = fit it")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(args.atomic_policy_root.resolve()))
    from atomic_policy.wrist.calibration import ObjectTrack, fit_wrist_camera, track_errors
    from atomic_policy.wrist.geometry import WristCamera, panda_fk_batch
    from atomic_policy.wrist.targets import phase_segments, target_windows

    root = args.dataset_root.resolve()
    output = args.output or root / "wrist_camera_calibration.json"
    episodes = sorted(p.parent for p in root.glob("*/episode_*/steps_molmo.jsonl"))
    by_session = defaultdict(list)
    for ep in episodes:
        by_session[ep.parent.name].append(ep)
    chosen = [ep for eps in by_session.values() for ep in (eps if args.per_session <= 0 else eps[: args.per_session])]
    print(f"{len(episodes)} annotated episodes in {len(by_session)} sessions; fitting on {len(chosen)}")

    cache_path = root / "wrist_calibration_tracks.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    probe = WristCamera(330, 330, 336, 188, [0, 0, 0], [0, 0, 0], [0, 0, 0.13])  # contacts only; tcp unused here
    pred = torch = None
    tracks, heights = [], defaultdict(list)
    for ep in chosen:
        rows, q, cmd, phases = load_episode(ep)
        T = panda_fk_batch(q)
        segs = phase_segments(phases)
        for w in target_windows(cmd, segs, probe, T):
            if w.role != "object":
                continue
            key = f"{ep.relative_to(root)}|{w.contact}"
            if key not in cache:
                if pred is None:
                    import torch
                    if args.gpu_mem_gb > 0:
                        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
                    from sam2.build_sam import build_sam2_video_predictor
                    cfg = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}[args.sam2_model]
                    pred = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{cfg}.yaml",
                                                      f"{args.sam2_checkpoints}/sam2.1_hiera_{args.sam2_model}.pt", device="cuda")
                stop = (w.release if w.release is not None else w.end) - 1
                seed = min(w.contact + args.seed_delay, stop)
                cache[key] = {str(k): v for k, v in sam2_backward(pred, ep, rows, w.start, seed, args.seed_pixel, torch).items()}
                cache_path.write_text(json.dumps(cache))
                print(f"  tracked {key} ({seed - w.start + 1} frames)", flush=True)
            tr = cache[key]
            frames = [f for f in range(w.start, w.contact - 2) if tr.get(str(f)) and tr[str(f)][2] >= args.min_area]
            if len(frames) >= 10:
                tracks.append(ObjectTrack(ep.parent.name, T[w.contact], T[frames], np.array([tr[str(f)][:2] for f in frames])))

    sessions = sorted({t.group for t in tracks})
    held = sessions[-1]
    tcp_z = args.tcp_z if args.tcp_z > 0 else None
    cam_cv, stats_cv = fit_wrist_camera([t for t in tracks if t.group != held], tcp_z=tcp_z)
    e_held = track_errors(cam_cv, [t for t in tracks if t.group == held])
    print(f"held-out session {held}: median {np.median(e_held):.1f}px p90 {np.percentile(e_held, 90):.1f}px "
          f"(fit on the other {len(sessions) - 1}: median {stats_cv['median_px']:.1f}px)")
    cam, stats = fit_wrist_camera(tracks, tcp_z=tcp_z)
    print(f"all sessions: {stats['frames']} frames from {stats['objects']} grasps, median {stats['median_px']:.1f}px, "
          f"p90 {stats['p90_px']:.1f}px; per session {json.dumps({k[-6:]: round(v, 1) for k, v in stats['per_group_median_px'].items()})}")
    print(f"fx fy cx cy = {cam.fx:.1f} {cam.fy:.1f} {cam.cx:.1f} {cam.cy:.1f}; camera centre in link8 {np.round(stats['camera_centre_in_link8_m'], 3).tolist()} m; fingertip d = {cam.tcp_in_link8[2]:.3f} m")

    # target heights per phase (deployment lifts a MolmoPoint pixel to this plane), from all annotated episodes
    for ep in episodes:
        rows, q, cmd, phases = load_episode(ep)
        T = panda_fk_batch(q)
        segs = phase_segments(phases)
        for w in target_windows(cmd, segs, cam, T):
            for k in w.segments:
                heights[segs[k][0]].append((w.role, w.point_base[2]))
    targets = {str(p): {"role": max({r for r, _ in v}, key=[r for r, _ in v].count),
                        "z_median_m": float(np.median([z for _, z in v])), "z_std_m": float(np.std([z for _, z in v])), "n": len(v)}
               for p, v in sorted(heights.items())}
    for p, t in targets.items():
        print(f"  phase {p}: {t['role']:6s} target height {t['z_median_m'] * 100:.1f} cm (std {t['z_std_m'] * 100:.1f} cm, n={t['n']})")
    cam.info = {**stats, "held_out_session": held, "held_out_median_px": float(np.median(e_held)),
                "held_out_p90_px": float(np.percentile(e_held, 90)), "targets_by_phase": targets,
                "dataset": str(root), "episodes_used": [str(ep.relative_to(root)) for ep in chosen],
                "seed_pixel": args.seed_pixel, "sam2_model": args.sam2_model}
    cam.save(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
