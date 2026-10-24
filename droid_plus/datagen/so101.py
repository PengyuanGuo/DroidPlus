# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SO-101 leader-arm adapter: extract joint/gripper values from LeRobot action
dicts, and map SO-101 (5 DoF) targets onto Franka (7 DoF) joint space.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from droid_plus.services.franky_client import HOME_POSITION

# Per-step action-dict keys emitted by LeRobot's SO-101 teleoperator.
SO101_ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]

# Possible gripper keys (LeRobot version drift) — checked in order.
SO101_GRIPPER_KEYS = [
    "gripper.pos",
    "finger.pos",
    "hand.pos",
    "gripper",
    "finger",
    "hand",
]


def action_to_so101_joints_deg(action: dict[str, Any]) -> np.ndarray:
    """Extract SO-101 joint positions (deg) from a LeRobot action dict."""
    return np.array([float(action[k]) for k in SO101_ACTION_KEYS], dtype=float)


def so101_to_franka(so101_joints_rad: np.ndarray) -> np.ndarray:
    """Map SO-101 (5 DoF, rad) → Franka (7 DoF, rad) joint targets.

    Drives a subset of the Franka joints from SO-101 inputs; unmapped joints
    are held at the Franka home configuration. SO-101 joints are clipped to
    conservative ranges before mapping.
    """
    q = np.asarray(so101_joints_rad, dtype=float)
    if q.shape != (5,):
        raise ValueError("Expected 5 joint values for SO-101 (J1..J5).")

    so_101_clipped = q.copy()
    so_101_clipped[0] = np.clip(so_101_clipped[0], -np.pi / 4, np.pi / 4)
    so_101_clipped[1] = np.clip(so_101_clipped[1], np.deg2rad(-60), np.deg2rad(90))
    so_101_clipped[2] = np.clip(so_101_clipped[2], np.deg2rad(-80), np.deg2rad(80))
    # Wrist joints intentionally unclamped.

    q_home = np.array(HOME_POSITION, dtype=float)
    q_franka = q_home.copy()
    q_franka[0] = -so_101_clipped[0]
    q_franka[1] = so_101_clipped[1]
    q_franka[3] = -np.pi / 2 - so_101_clipped[2]
    q_franka[5] = -so_101_clipped[3] + np.pi
    q_franka[6] = -so_101_clipped[4]

    return q_franka


def extract_so101_gripper_deg(action: dict[str, Any]) -> float | None:
    """Extract LeRobot's gripper value (0–100), or ``None``.

    The historical function name is retained for callers; the gripper motor
    uses RANGE_0_100 independently of the arm's degree setting.
    """
    for key in SO101_GRIPPER_KEYS:
        if key in action:
            return float(action[key])
    # Substring-fallback: accommodate LeRobot key renames.
    for key in action.keys():
        kl = key.lower()
        if "gripper" in kl or "finger" in kl or "hand" in kl:
            return float(action[key])
    return None


def so101_gripper_to_robotiq(so101_gripper_deg: float) -> int:
    """SO-101 gripper angle (deg) → Robotiq bits.

    0° (SO-101) → 255 (Robotiq closed); 90° → 0 (fully open). Linear, clipped.
    """
    deg = float(np.clip(so101_gripper_deg, 0.0, 90.0))
    robotiq_pos = int(255 * (1.0 - deg / 90.0))
    return int(np.clip(robotiq_pos, 0, 255))
