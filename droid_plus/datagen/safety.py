# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Table-collision safety: FK-based clamp that prevents commanded Franka joint
configurations from driving the end-effector below a minimum z height.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from droid_plus.analysis.end_effector_pose import compute_ee_pose

# Minimum allowed EE Z height (metres, in robot base frame). Tune to sit just
# above the table surface.
DEFAULT_MIN_EE_Z = 0.23


def enforce_min_z(
    q_franka: np.ndarray,
    q_prev_safe: np.ndarray,
    model: Any,
    data: Any,
    ee_frame: str,
    min_z: float = DEFAULT_MIN_EE_Z,
) -> tuple[np.ndarray, float]:
    """Clamp a candidate Franka joint target to keep the EE above ``min_z``.

    Strategy:
      1. If ``q_franka`` keeps the EE above ``min_z``, pass it through.
      2. Otherwise revert the two "lifting" joints (j1, j3) to the last known
         safe values.
      3. If still too low, revert the full configuration to ``q_prev_safe``.

    If ``model``/``data``/``ee_frame`` are unavailable (FK disabled), returns
    ``q_franka`` unchanged with ``z=nan``.
    """
    if model is None or data is None or not ee_frame:
        return q_franka, float("nan")

    ee = compute_ee_pose(model, data, ee_frame, q_franka)
    z = ee["position"][2]

    if z >= min_z:
        return q_franka, z

    q_clamped = q_franka.copy()
    q_clamped[1] = q_prev_safe[1]
    q_clamped[3] = q_prev_safe[3]

    ee2 = compute_ee_pose(model, data, ee_frame, q_clamped)
    z2 = ee2["position"][2]
    if z2 >= min_z:
        print(f"[safety] EE z={z:.3f}m < {min_z:.3f}m — clamped joints 1,3")
        return q_clamped, z2

    print(f"[safety] EE z={z2:.3f}m still < {min_z:.3f}m — reverting to last safe config")
    ee3 = compute_ee_pose(model, data, ee_frame, q_prev_safe)
    return q_prev_safe.copy(), ee3["position"][2]
