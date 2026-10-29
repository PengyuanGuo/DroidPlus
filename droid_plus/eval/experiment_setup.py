# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared setup helpers for experiment entry points.

Both ``scripts/run_experiment_cli.py`` and ``scripts/run_experiment_ui.py``
go through the same bootstrap sequence: wait for cameras, home the robot,
pick a run directory, resolve policy host/port, build the policy client,
and (optionally) compute EE trajectories at the end. This module owns that
shared sequence so the two entry points can stay thin.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from droid_plus.constants import CAMERA_SERIAL_NAMES, POLICIES

if TYPE_CHECKING:
    from droid_plus.robot import DroidPlus

# ANSI colors reused by both entry points.
_YELLOW = "\033[93m"
_RESET = "\033[0m"


# ── Cameras ──────────────────────────────────────────────────────────────────

def wait_for_cameras(droid: "DroidPlus", timeout_s: float = 15.0, poll_s: float = 0.5) -> None:
    """Poll camera_service until the cameras this run uses report ``has_frame=True``.

    Cameras the service knows but the run leaves unset (e.g. ``--wrist-camera-id ''``)
    are not waited for. With no camera ids set, all service cameras are waited for.
    """
    used = {str(getattr(droid, f"{role}_camera_id", "") or "") for role in ("left", "wrist", "right")} - {""}
    t0 = time.time()
    while True:
        try:
            health = droid.camera.health()
        except Exception:
            elapsed = time.time() - t0
            print(f"\r\033[KCameras: waiting for camera_service ({elapsed:.0f}s)...", end="", flush=True)
            if elapsed > timeout_s:
                print(f"\n{_YELLOW}Camera service unreachable after {timeout_s:.0f}s — continuing anyway.{_RESET}")
                break
            time.sleep(poll_s)
            continue

        cam_statuses = health.get("cameras", {})
        if used:
            cam_statuses = {cid: info for cid, info in cam_statuses.items() if str(cid) in used}
        parts = []
        errors = []
        all_ok = True
        for cid, info in cam_statuses.items():
            name = CAMERA_SERIAL_NAMES.get(cid, cid)
            ok = info.get("has_frame", False)
            err = info.get("last_error")
            if ok:
                parts.append(f"\033[92m{name}=OK\033[0m")
            else:
                parts.append(f"\033[91m{name}=WAIT\033[0m")
                all_ok = False
                if err:
                    errors.append(f"{name}: {err}")

        elapsed = time.time() - t0
        status = " | ".join(parts) if parts else "no cameras detected"
        err_str = f"  [{'; '.join(errors)}]" if errors else ""
        print(f"\r\033[KCameras: {status}{err_str} ({elapsed:.1f}s)", end="", flush=True)

        if all_ok and parts:
            print()
            break
        if elapsed > timeout_s:
            print(f"\n{_YELLOW}Not all cameras live after {timeout_s:.0f}s — continuing anyway.{_RESET}")
            break
        time.sleep(poll_s)


# ── Robot ────────────────────────────────────────────────────────────────────

def init_droid_and_home(droid: "DroidPlus", dry_run: bool = False) -> None:
    """Connect gripper, stop, move home, open gripper. Best-effort on gripper ops.

    When ``dry_run`` is True, skip every command that would actuate the robot.
    """
    if dry_run:
        print(f"{_YELLOW}[dry-run] skipping robot init (stop/home/open gripper).{_RESET}")
        return

    try:
        droid.connect_gripper()
        droid.activate_gripper()
    except Exception:
        pass

    try:
        droid.stop()
    except Exception:
        pass

    droid.move_to_home()
    time.sleep(2)
    droid.open_gripper(wait=True)
    droid.stop()


# ── Run directory ────────────────────────────────────────────────────────────

def make_base_run_dir(tag: str, policy_name: str, parent: str = "output") -> str:
    """Build ``{tag}_{policy}_{timestamp}`` under ``parent/`` and create it."""
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    policy_str = f"_{policy_name}" if policy_name else ""
    tag_str = tag if tag else ""
    run_name = f"{tag_str}{policy_str}_{ts}"
    path = os.path.join(parent, run_name)
    os.makedirs(path, exist_ok=True)
    return path


# ── Policy resolution ────────────────────────────────────────────────────────

def resolve_policy(
    policy_name: str,
    host_override: str | None = None,
    port_override: int | None = None,
) -> tuple[str, int, bool]:
    """Resolve ``(host, port, from_defaults)`` for a named policy.

    Raises ``SystemExit`` with the list of known policies if the name is
    unknown and no overrides are supplied.
    """
    cfg = POLICIES.get(policy_name)
    if cfg is None and (host_override is None or port_override is None):
        available = ", ".join(POLICIES.keys())
        raise SystemExit(
            f"Unknown policy '{policy_name}'. Available: {available}. "
            "Or supply host/port overrides."
        )
    from_defaults = host_override is None and port_override is None
    host = host_override if host_override is not None else cfg["host"]  # type: ignore[index]
    port = port_override if port_override is not None else cfg["port"]  # type: ignore[index]
    return str(host), int(port), from_defaults


def build_policy_client(host: str, port: int, open_loop_horizon: int) -> Any:
    """Instantiate a ``Pi0DroidJointposClient``. Raises ``ImportError`` if unavailable."""
    try:
        from droid_plus.policies import Pi0DroidJointposClient
    except ImportError as e:
        raise ImportError(
            "openpi_client is required to run experiments. "
            "Install it with: pip install openpi_client\n"
            "Or install franky with policy support: pip install franky[policy]"
        ) from e
    return Pi0DroidJointposClient(
        remote_host=host,
        remote_port=port,
        open_loop_horizon=int(open_loop_horizon),
    )


# ── Post-run analysis ────────────────────────────────────────────────────────

def maybe_compute_ee_trajectories(run_dir: str | None, *, overwrite: bool = False, verbose: bool = True) -> None:
    """Compute and save EE trajectories for every episode under ``run_dir``."""
    if not run_dir:
        return
    print("\nComputing end-effector trajectories for recorded episodes...")
    try:
        from droid_plus.analysis.end_effector_pose import compute_and_save_ee_trajectories
        compute_and_save_ee_trajectories(run_dir=run_dir, overwrite=overwrite, verbose=verbose)
    except Exception as e:
        print(f"Warning: Failed to compute EE trajectories: {e}")
