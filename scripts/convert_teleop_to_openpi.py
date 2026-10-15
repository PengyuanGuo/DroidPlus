# SPDX-License-Identifier: Apache-2.0
"""
Convert DroidPlus teleop runs into an openpi-ready LeRobot (v2.1) dataset in
DROID naming, for fine-tuning pi05_droid_jointpos.

Run with the openpi venv (it pins the lerobot version openpi reads):

    HF_LEROBOT_HOME=/path/to/datasets \
    /scratch/.../openpi/.venv/bin/python scripts/convert_teleop_to_openpi.py \
        --repo-id droidplus/chicken_blue_bowl \
        output/teleop_20261010_173324 output/teleop_20261010_175353 output/teleop_20261010_180904

Per episode:
- skip episodes whose meta.json is not ``success: true`` (or is ``valid: false``)
- trim the start until the follower has caught up with the SO-101 leader
  (max |commanded - measured| joint gap < ``--settle-thresh`` rad)
- resample onto an exact ``--fps`` grid from ``t_wall_s``: joints and joint
  targets are linearly interpolated, the binary gripper command is held, and
  each frame takes the nearest-in-time camera image
- write DROID keys: exterior_image_1_left (left), wrist_image_left (wrist),
  joint_position (7), gripper_position (1), actions (7 absolute joint targets
  + gripper command). openpi's DeltaActions transform makes the joints relative.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

IMAGE_HW = (180, 320)  # DROID RLDS resolution, as in openpi's DROID converter


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_image(path: Path, cache: dict) -> np.ndarray:
    if path not in cache:
        if len(cache) > 64:
            cache.clear()
        img = Image.open(path).convert("RGB").resize((IMAGE_HW[1], IMAGE_HW[0]), Image.BICUBIC)
        cache[path] = np.asarray(img, dtype=np.uint8)
    return cache[path]


def _nearest_index(t: np.ndarray, tk: np.ndarray) -> np.ndarray:
    hi = np.clip(np.searchsorted(t, tk), 1, len(t) - 1)
    lo = hi - 1
    return np.where(np.abs(t[hi] - tk) < np.abs(tk - t[lo]), hi, lo)


def resample_episode(ep_dir: Path, fps: float, settle_thresh: float) -> dict:
    steps = _read_jsonl(ep_dir / "steps.jsonl")
    t = np.array([s["t_wall_s"] for s in steps], dtype=np.float64)
    state = np.array([s["state"]["positions"][:8] for s in steps], dtype=np.float64)
    action = np.array([s["action"]["positions"][:8] for s in steps], dtype=np.float64)
    left = [ep_dir / s["images"]["left"] for s in steps]
    wrist = [ep_dir / s["images"]["wrist"] for s in steps]
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{ep_dir}: non-increasing t_wall_s")

    gap = np.abs(action[:, :7] - state[:, :7]).max(axis=1)
    settled = np.flatnonzero(gap < settle_thresh)
    if settled.size == 0:
        raise ValueError(f"{ep_dir}: commanded/measured joint gap never drops below {settle_thresh} rad")
    i0 = int(settled[0])

    tk = t[i0] + np.arange(int(np.floor((t[-1] - t[i0]) * fps)) + 1) / fps
    interp = lambda x: np.stack([np.interp(tk, t, x[:, j]) for j in range(x.shape[1])], axis=1)  # noqa: E731
    joint_position = interp(state[:, :7])
    gripper_position = np.interp(tk, t, state[:, 7])[:, None]
    joint_target = interp(action[:, :7])
    gripper_cmd = action[np.searchsorted(t, tk, side="right") - 1, 7][:, None]  # hold
    img_idx = _nearest_index(t, tk)

    return {
        "joint_position": joint_position.astype(np.float32),
        "gripper_position": gripper_position.astype(np.float32),
        "actions": np.concatenate([joint_target, gripper_cmd], axis=1).astype(np.float32),
        "left": [left[i] for i in img_idx],
        "wrist": [wrist[i] for i in img_idx],
        "trimmed_frames": i0,
        "raw_hz": float(1.0 / np.diff(t).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="Teleop run dirs containing episode_*/")
    p.add_argument("--repo-id", required=True, help="Dataset id; written to $HF_LEROBOT_HOME/<repo-id>")
    p.add_argument("--prompt", default=None, help="Override every episode's instruction")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--settle-thresh", type=float, default=0.15, help="rad")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    image_feat = {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=int(args.fps),
        features={
            "exterior_image_1_left": image_feat,
            "wrist_image_left": image_feat,
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    report = {"source_runs": [str(r.resolve()) for r in args.runs], "fps": args.fps,
              "settle_thresh_rad": args.settle_thresh, "included": [], "skipped": []}
    cache: dict = {}
    for run in args.runs:
        for ep_dir in sorted(run.glob("episode_*")):
            meta = json.loads((ep_dir / "meta.json").read_text())
            if meta.get("success") is not True or meta.get("valid") is False:
                report["skipped"].append({"episode": str(ep_dir), "success": meta.get("success"),
                                          "valid": meta.get("valid")})
                print(f"skip {ep_dir} (success={meta.get('success')}, valid={meta.get('valid')})")
                continue
            prompt = args.prompt or meta["instruction"]
            ep = resample_episode(ep_dir, args.fps, args.settle_thresh)
            for k in range(len(ep["actions"])):
                dataset.add_frame({
                    "exterior_image_1_left": _load_image(ep["left"][k], cache),
                    "wrist_image_left": _load_image(ep["wrist"][k], cache),
                    "joint_position": ep["joint_position"][k],
                    "gripper_position": ep["gripper_position"][k],
                    "actions": ep["actions"][k],
                    "task": prompt,
                })
            dataset.save_episode()
            report["included"].append({"episode": str(ep_dir), "frames": len(ep["actions"]),
                                       "trimmed_raw_frames": ep["trimmed_frames"],
                                       "raw_hz": round(ep["raw_hz"], 2), "prompt": prompt})
            print(f"{ep_dir}: raw {ep['raw_hz']:.1f} Hz, trimmed {ep['trimmed_frames']} raw frames "
                  f"-> {len(ep['actions'])} frames @ {args.fps:g} Hz")

    (out / "meta" / "droidplus_conversion.json").write_text(json.dumps(report, indent=2))
    n = sum(e["frames"] for e in report["included"])
    print(f"\nWrote {len(report['included'])} episodes / {n} frames to {out} "
          f"(skipped {len(report['skipped'])})")


if __name__ == "__main__":
    main()
