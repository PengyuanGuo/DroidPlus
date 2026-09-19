# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Compute end-effector poses from recorded joint positions using Pinocchio + URDF.

This fetches the URDF from franky_service, builds a Pinocchio model, and runs FK
for all timesteps in a recorded episode's steps.jsonl.

Usage:
  python analysis/end_effector_pose.py runs/recordings/.../episode_000
  python analysis/end_effector_pose.py runs/recordings/.../episode_000 --plot
  python analysis/end_effector_pose.py runs/recordings/.../episode_000 --out poses.csv

Requires: pinocchio (pip install pin)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from droid_plus.constants import FRANKY_SERVICE_URL

try:
    import pinocchio as pin  # type: ignore
except ImportError:
    pin = None  # type: ignore[assignment]

VERBOSE = False

# ---------------------------------------------------------------------------
# URDF fetching
# ---------------------------------------------------------------------------

# def get_urdf(service_url: str = FRANKY_SERVICE_URL) -> str:
#     """Fetch the URDF XML string from franky_service."""
#     response = requests.get(f"{service_url}/urdf", timeout=10)
#     response.raise_for_status()
#     return response.json()["urdf"]
def get_urdf(service_url: str = FRANKY_SERVICE_URL) -> str:
    """Fetch the URDF XML string from franky_service, or a local file."""
    local = os.environ.get("FRANKA_URDF_PATH")
    if local:
        with open(local) as f:
            return f.read()
    response = requests.get(f"{service_url}/urdf", timeout=10)
    response.raise_for_status()
    return response.json()["urdf"]

@lru_cache(maxsize=1)
def _cached_urdf(service_url: str) -> str:
    return get_urdf(service_url)


# ---------------------------------------------------------------------------
# Pinocchio model building
# ---------------------------------------------------------------------------

def build_model_from_urdf(urdf_xml: str) -> tuple[Any, Any, str]:
    """
    Build a Pinocchio model from URDF XML string.

    Returns:
        (model, data, ee_frame_name)
    """
    if pin is None:
        raise ImportError("pinocchio is required. Install with: pip install pin")

    # Pinocchio needs a file path; write URDF to a temp file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_xml)
        urdf_path = f.name

    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
    finally:
        os.unlink(urdf_path)

    # Diagnostic: print model info.
    if VERBOSE:
        print(f"  Pinocchio model: nq={model.nq}, nv={model.nv}, njoints={model.njoints}, nframes={model.nframes}")
    if model.njoints > 0:
        joint_names = [model.names[i] for i in range(model.njoints)]
        if VERBOSE:
            print(f"  Joints: {joint_names}")
    if model.nq == 0:
        if VERBOSE:
            print("  WARNING: model.nq == 0 means all joints are fixed! FK will be constant.")

    # List all frames for debugging.
    all_frame_names = [model.frames[i].name for i in range(model.nframes)]
    if VERBOSE:
        print(f"  Available frames: {all_frame_names}")

    # Find the end-effector frame.
    # Priority order: common Franka EE names, then link8 variants, then last link before fixed frames.
    ee_candidates = [
        "panda_link8", "panda_hand", "panda_hand_tcp", "panda_grasptarget",
        "link8", "flange", "tool0", "ee_link", "end_effector",
    ]
    ee_frame_name = None
    for name in ee_candidates:
        if model.existFrame(name):
            ee_frame_name = name
            break

    if ee_frame_name is None:
        # Fallback: find the frame attached to the last moving joint (joint7).
        # Look for frames whose parent joint is joint7 or whose name contains "7" or "8".
        for i in range(model.nframes):
            frame = model.frames[i]
            fname = frame.name
            # Skip accelerometer/sensor frames.
            if "accelerometer" in fname.lower() or "sensor" in fname.lower():
                continue
            # Prefer frames with "7" or "8" in name (end of kinematic chain).
            if "8" in fname or (fname.endswith("7") and "link" in fname.lower()):
                ee_frame_name = fname
                break
        # Ultimate fallback: last non-sensor frame.
        if ee_frame_name is None:
            for i in range(model.nframes - 1, -1, -1):
                fname = model.frames[i].name
                if "accelerometer" not in fname.lower() and "sensor" not in fname.lower():
                    ee_frame_name = fname
                    break

    if ee_frame_name is None:
        if model.nframes > 0:
            ee_frame_name = model.frames[-1].name
        else:
            raise RuntimeError("No frames found in URDF model")

    return model, data, ee_frame_name


def compute_ee_pose(model: Any, data: Any, ee_frame_name: str, q: np.ndarray) -> dict[str, Any]:
    """
    Compute end-effector pose for a single joint configuration.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        ee_frame_name: Name of the end-effector frame
        q: Joint positions (7,)

    Returns:
        {"position": [x, y, z], "quaternion": [x, y, z, w], "rpy": [r, p, y]}
    """
    if pin is None:
        raise ImportError("pinocchio is required")

    q_pin = np.asarray(q, dtype=np.float64).flatten()
    # Pinocchio expects nq-dim config; for Franka arm it's 7.
    if q_pin.shape[0] < model.nq:
        # Pad with zeros if needed (shouldn't happen for 7-DOF arm).
        q_pin = np.concatenate([q_pin, np.zeros(model.nq - q_pin.shape[0])])
    elif q_pin.shape[0] > model.nq:
        q_pin = q_pin[: model.nq]

    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    frame_id = model.getFrameId(ee_frame_name)
    oMf = data.oMf[frame_id]  # SE3 transform

    pos = oMf.translation.copy()
    rot = oMf.rotation.copy()

    # Quaternion (x, y, z, w) from rotation matrix.
    quat = pin.Quaternion(rot)
    quat_xyzw = np.array([quat.x, quat.y, quat.z, quat.w])

    # RPY (roll, pitch, yaw) from rotation matrix.
    rpy = pin.rpy.matrixToRpy(rot)

    return {
        "position": pos.tolist(),
        "quaternion": quat_xyzw.tolist(),
        "rpy": rpy.tolist(),
    }


# ---------------------------------------------------------------------------
# Episode loading (shared with tracking.py)
# ---------------------------------------------------------------------------

def _iter_steps(jsonl_path: Path) -> Iterable[dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _find_episode_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "steps.jsonl").exists():
        return [run_dir]
    eps = sorted([p for p in run_dir.glob("episode_*") if p.is_dir()])
    return [p for p in eps if (p / "steps.jsonl").exists()]


def load_joint_positions(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load timesteps and joint positions from an episode.

    Returns:
        (t_wall_s, state_positions, action_positions)
        Each positions array is (N, 7) for the 7 arm joints.
    """
    steps_path = episode_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing {steps_path}")

    t_wall_s = []
    state_pos = []
    action_pos = []

    for step in _iter_steps(steps_path):
        t_wall_s.append(float(step.get("t_wall_s", 0.0)))
        st = step.get("state") or {}
        ac = step.get("action") or {}
        # First 7 are arm joints, 8th is gripper.
        sp = [float(x) for x in (st.get("positions") or [])][:7]
        ap = [float(x) for x in (ac.get("positions") or [])][:7]
        # Pad if needed.
        while len(sp) < 7:
            sp.append(0.0)
        while len(ap) < 7:
            ap.append(0.0)
        state_pos.append(sp)
        action_pos.append(ap)

    return (
        np.array(t_wall_s, dtype=np.float64),
        np.array(state_pos, dtype=np.float64),
        np.array(action_pos, dtype=np.float64),
    )


def compute_ee_trajectory(
    episode_dir: Path,
    service_url: str = FRANKY_SERVICE_URL,
    use_action: bool = False,
) -> dict[str, Any]:
    """
    Compute end-effector poses for all timesteps in an episode.

    Args:
        episode_dir: Path to episode directory containing steps.jsonl
        service_url: franky_service URL for fetching URDF
        use_action: If True, compute FK for action (commanded) positions instead of state

    Returns:
        {
            "t_wall_s": [...],
            "t_rel_s": [...],
            "position": [[x, y, z], ...],
            "quaternion": [[x, y, z, w], ...],
            "rpy": [[r, p, y], ...],
            "ee_frame": "panda_link8",
        }
    """
    urdf_xml = _cached_urdf(service_url)
    model, data, ee_frame = build_model_from_urdf(urdf_xml)

    t_wall_s, state_pos, action_pos = load_joint_positions(episode_dir)
    joint_pos = action_pos if use_action else state_pos

    positions = []
    quaternions = []
    rpys = []

    for i in range(joint_pos.shape[0]):
        q = joint_pos[i]
        ee = compute_ee_pose(model, data, ee_frame, q)
        positions.append(ee["position"])
        quaternions.append(ee["quaternion"])
        rpys.append(ee["rpy"])

    t0 = float(t_wall_s[0]) if len(t_wall_s) > 0 else 0.0
    t_rel_s = (t_wall_s - t0).tolist()

    return {
        "t_wall_s": t_wall_s.tolist(),
        "t_rel_s": t_rel_s,
        "position": positions,
        "quaternion": quaternions,
        "rpy": rpys,
        "ee_frame": ee_frame,
    }


# ---------------------------------------------------------------------------
# Batch processing for run folders
# ---------------------------------------------------------------------------

def compute_and_save_ee_trajectories(
    run_dir: str | Path,
    service_url: str = FRANKY_SERVICE_URL,
    use_action: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Compute and save end-effector trajectories for all episodes in a run folder.

    For each episode directory containing steps.jsonl, this computes the EE trajectory
    using forward kinematics and saves it as ee_trajectory.json in the episode folder.

    Args:
        run_dir: Path to run directory containing episode_*/ subdirectories
        service_url: franky_service URL for fetching URDF
        use_action: If True, compute FK for action (commanded) positions instead of state
        overwrite: If True, recompute even if ee_trajectory.json already exists
        verbose: If True, print progress messages

    Returns:
        Dict mapping episode directory name to the path of the saved ee_trajectory.json
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    episode_dirs = _find_episode_dirs(run_dir)
    if not episode_dirs:
        if verbose:
            print(f"No episodes found in {run_dir}")
        return {}

    if verbose:
        print(f"Found {len(episode_dirs)} episode(s) in {run_dir}")

    saved_files: dict[str, Path] = {}

    for episode_dir in episode_dirs:
        out_path = episode_dir / "ee_trajectory.json"

        # Skip if already exists and not overwriting
        if out_path.exists() and not overwrite:
            if verbose:
                print(f"  Skipping {episode_dir.name}: ee_trajectory.json already exists")
            saved_files[episode_dir.name] = out_path
            continue

        try:
            if verbose:
                print(f"  Computing EE trajectory for {episode_dir.name}...")

            traj = compute_ee_trajectory(
                episode_dir,
                service_url=service_url,
                use_action=use_action,
            )

            # Save to JSON
            out_path.write_text(json.dumps(traj, indent=2), encoding="utf-8")
            saved_files[episode_dir.name] = out_path

            if verbose:
                n_steps = len(traj.get("t_wall_s", []))
                print(f"    Saved {out_path} ({n_steps} timesteps)")

        except Exception as e:
            if verbose:
                print(f"    ERROR processing {episode_dir.name}: {e}")
            continue

    return saved_files


def compute_and_save_ee_trajectory_single(
    episode_dir: str | Path,
    service_url: str = FRANKY_SERVICE_URL,
    use_action: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path | None:
    """
    Compute and save end-effector trajectory for a single episode.

    Args:
        episode_dir: Path to episode directory containing steps.jsonl
        service_url: franky_service URL for fetching URDF
        use_action: If True, compute FK for action (commanded) positions instead of state
        overwrite: If True, recompute even if ee_trajectory.json already exists
        verbose: If True, print progress messages

    Returns:
        Path to saved ee_trajectory.json, or None if skipped/failed
    """
    episode_dir = Path(episode_dir).resolve()
    out_path = episode_dir / "ee_trajectory.json"

    if out_path.exists() and not overwrite:
        if verbose:
            print(f"Skipping {episode_dir.name}: ee_trajectory.json already exists")
        return out_path

    try:
        if verbose:
            print(f"Computing EE trajectory for {episode_dir.name}...")

        traj = compute_ee_trajectory(
            episode_dir,
            service_url=service_url,
            use_action=use_action,
        )

        out_path.write_text(json.dumps(traj, indent=2), encoding="utf-8")

        if verbose:
            n_steps = len(traj.get("t_wall_s", []))
            print(f"  Saved {out_path} ({n_steps} timesteps)")

        return out_path

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ee_trajectory(traj: dict[str, Any], title: str = "", show: bool = True, out_path: Path | None = None) -> None:
    """Plot end-effector position and orientation over time."""
    import matplotlib.pyplot as plt

    t = np.array(traj["t_rel_s"])
    pos = np.array(traj["position"])
    rpy = np.array(traj["rpy"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    ax_pos = axes[0]
    ax_pos.plot(t, pos[:, 0], label="x", lw=1.5)
    ax_pos.plot(t, pos[:, 1], label="y", lw=1.5)
    ax_pos.plot(t, pos[:, 2], label="z", lw=1.5)
    ax_pos.set_ylabel("position (m)")
    ax_pos.set_title((title + " " if title else "") + f"End-effector trajectory ({traj['ee_frame']})")
    ax_pos.legend(loc="upper right")
    ax_pos.grid(True, which="major", alpha=0.25)
    ax_pos.minorticks_on()
    ax_pos.grid(True, which="minor", alpha=0.1, linestyle=":")

    ax_rpy = axes[1]
    rad2deg = 180.0 / np.pi
    ax_rpy.plot(t, rpy[:, 0] * rad2deg, label="roll", lw=1.5)
    ax_rpy.plot(t, rpy[:, 1] * rad2deg, label="pitch", lw=1.5)
    ax_rpy.plot(t, rpy[:, 2] * rad2deg, label="yaw", lw=1.5)
    ax_rpy.set_ylabel("orientation (deg)")
    ax_rpy.set_xlabel("t since first step (s)")
    ax_rpy.legend(loc="upper right")
    ax_rpy.grid(True, which="major", alpha=0.25)
    ax_rpy.minorticks_on()
    ax_rpy.grid(True, which="minor", alpha=0.1, linestyle=":")

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Compute end-effector poses from recorded joint positions.")
    p.add_argument("run_dir", help="Episode dir (contains steps.jsonl) or run dir (contains episode_*/)")
    p.add_argument("--episode", type=int, default=None, help="Episode index (when run_dir contains episode_*/)")
    p.add_argument("--service-url", default=FRANKY_SERVICE_URL, help="franky_service URL for URDF")
    p.add_argument("--use-action", action="store_true", help="Use action (commanded) positions instead of state")
    p.add_argument("--plot", action="store_true", help="Plot the end-effector trajectory")
    p.add_argument("--no-show", action="store_true", help="Don't open interactive plot window")
    p.add_argument("--out", default=None, help="Output file path (.csv or .json)")
    p.add_argument("--out-plot", default=None, help="Output path for plot image (e.g. ee_traj.png)")
    args = p.parse_args()

    run_dir = Path(os.path.expanduser(args.run_dir)).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    eps = _find_episode_dirs(run_dir)
    if not eps:
        raise RuntimeError(f"No episodes found under {run_dir}")

    # Select episode.
    if (run_dir / "steps.jsonl").exists():
        episode_dir = run_dir
    else:
        if args.episode is None:
            episode_dir = eps[-1]  # Default to latest.
        else:
            matches = [e for e in eps if e.name.endswith(f"_{args.episode:03d}") or e.name == f"episode_{args.episode}"]
            if not matches:
                raise RuntimeError(f"Episode {args.episode} not found in {run_dir}")
            episode_dir = matches[0]

    print(f"Computing EE trajectory for: {episode_dir}")
    traj = compute_ee_trajectory(episode_dir, service_url=args.service_url, use_action=args.use_action)
    print(f"  EE frame: {traj['ee_frame']}")
    print(f"  Timesteps: {len(traj['t_wall_s'])}")

    # Print summary stats.
    pos = np.array(traj["position"])
    print("  Position range:")
    for i, axis in enumerate(["x", "y", "z"]):
        print(f"    {axis}: [{pos[:, i].min():.4f}, {pos[:, i].max():.4f}] m")

    # Save output.
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".csv":
            import csv
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t_wall_s", "t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw", "roll", "pitch", "yaw"])
                for i in range(len(traj["t_wall_s"])):
                    row = [
                        traj["t_wall_s"][i],
                        traj["t_rel_s"][i],
                        *traj["position"][i],
                        *traj["quaternion"][i],
                        *traj["rpy"][i],
                    ]
                    writer.writerow(row)
            print(f"Wrote {out_path}")
        else:
            out_path.write_text(json.dumps(traj, indent=2), encoding="utf-8")
            print(f"Wrote {out_path}")

    # Plot.
    if args.plot or args.out_plot:
        title = episode_dir.name
        out_plot = Path(args.out_plot) if args.out_plot else None
        plot_ee_trajectory(traj, title=title, show=not args.no_show, out_path=out_plot)


if __name__ == "__main__":
    main()
