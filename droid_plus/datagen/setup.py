# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Teleop session bootstrap: FK model, SO-101 leader connection, run dir.

Parallels ``droid_plus.eval.experiment_setup`` but for data-generation flows.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from droid_plus.analysis.end_effector_pose import build_model_from_urdf, get_urdf
from droid_plus.constants import FRANKY_SERVICE_URL

if TYPE_CHECKING:
    from droid_plus.robot import DroidPlus


_YELLOW = "\033[93m"
_RESET = "\033[0m"


# ── FK model ─────────────────────────────────────────────────────────────────

def build_fk_model(franky_service_url: str = FRANKY_SERVICE_URL) -> tuple[Any, Any, str]:
    """Fetch the URDF from franky_service and build a Pinocchio FK model.

    Returns ``(pin_model, pin_data, ee_frame)`` ready for
    :func:`droid_plus.datagen.safety.enforce_min_z`.
    """
    print("Fetching URDF for FK safety checks...")
    urdf_xml = get_urdf(franky_service_url)
    pin_model, pin_data, ee_frame = build_model_from_urdf(urdf_xml)
    print(f"FK model ready (ee_frame={ee_frame!r})")
    return pin_model, pin_data, ee_frame


# ── Gripper ──────────────────────────────────────────────────────────────────

def init_gripper(droid: "DroidPlus", *, backend: str = "robotiq") -> bool:
    """Best-effort connect + activate of the gripper.

    ``backend`` is informational (``robotiq`` or ``franka``); the actual HTTP
    target is whatever ``droid.gripper`` was constructed with.

    Returns ``True`` if initialization succeeded, else ``False`` (so the
    caller can decide whether to proceed without gripper control).
    """
    try:
        print(f"Initializing gripper (backend={backend})...")
        if backend == "franka":
            # Homing often exceeds the default 5s HTTP timeout.
            droid.gripper.timeout_s = max(float(droid.gripper.timeout_s), 15.0)
            droid.gripper.activate_timeout_s = max(
                float(getattr(droid.gripper, "activate_timeout_s", 60.0)),
                60.0,
            )
            print("Connecting to Franka Hand via franky_service...")
        conn = droid.connect_gripper()
        if backend == "franka" and not conn.get("connected", True):
            raise RuntimeError(
                f"Franka Hand not connected on franky_service: {conn}. "
                "Redeploy franky_service on the NUC with FRANKA_GRIPPER=1."
            )
        if backend == "franka":
            print("Homing Franka Hand (often 10–30s, please wait)...")
        droid.activate_gripper()
        print("Gripper initialized successfully")
        return True
    except Exception as e:
        print(f"{_YELLOW}Warning: Could not initialize gripper: {e}{_RESET}")
        print("Continuing without gripper control...")
        return False


# ── SO-101 ───────────────────────────────────────────────────────────────────

def connect_so101(port: str = "/dev/ttyACM0", *, id: str = "main_leader", settle_s: float = 2.0) -> Any:
    """Connect to the SO-101 leader arm.

    ``settle_s`` sleeps between gripper init and SO-101 handshake — the
    gripper USB initialization can otherwise interfere with the SO-101 serial
    handshake when they share a hub. Callers that don't init the gripper can
    pass ``settle_s=0.0``.
    """
    if settle_s > 0:
        print(f"Waiting {settle_s}s before initializing SO-101...")
        time.sleep(settle_s)
    # Lazy import: lerobot is an optional extra.
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    print(f"Connecting to SO-101 on {port}...")
    teleop = SO101Leader(SO101LeaderConfig(port=port, id=id))
    teleop.connect()
    print("SO-101 connected successfully")
    return teleop


# ── Run directory ────────────────────────────────────────────────────────────

def make_teleop_run_dir(task: str = "", *, parent: str = "output") -> str:
    """Build ``{task_}teleop_{timestamp}`` under ``parent/`` and create it."""
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    task_str = f"{task}_" if task else ""
    run_name = f"{task_str}teleop_{ts}"
    path = os.path.join(parent, run_name)
    os.makedirs(path, exist_ok=True)
    return path


# ── Rate helpers ─────────────────────────────────────────────────────────────

def compute_record_every_n(rate_hz: float, record_rate_hz: float) -> int:
    """Control ticks per recorded frame: ``max(1, round(rate / record_rate))``."""
    return max(1, round(float(rate_hz) / float(record_rate_hz)))
