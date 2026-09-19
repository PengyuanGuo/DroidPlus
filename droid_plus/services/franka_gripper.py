# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Franka Hand helpers + Robotiq-compatible bit mapping.

Teleop / GripperClient speak in Robotiq-style position bits:
  0   = fully open
  255 = fully closed

Franka Hand commands use finger width in metres (typically max ≈ 0.08 m).
"""
from __future__ import annotations

# Official Franka Hand max opening is 80 mm. Actual hardware max_width is
# read from the device when available and overrides this default.
DEFAULT_FRANKA_HAND_MAX_WIDTH_M = 0.08

# Speed mapping for Robotiq-style speed bits → m/s.
# Franka Hand max continuous speed is ~0.1 m/s (libfranka hard limit).
DEFAULT_SPEED_MIN_M_S = 0.05
DEFAULT_SPEED_MAX_M_S = 0.10

# Force mapping for Robotiq-style force bits → Newtons (grasp only).
DEFAULT_FORCE_MIN_N = 5.0
DEFAULT_FORCE_MAX_N = 70.0


def bits_to_width_m(bits: int, *, max_width_m: float = DEFAULT_FRANKA_HAND_MAX_WIDTH_M) -> float:
    """Robotiq bits (0=open … 255=closed) → Franka finger width (m)."""
    b = max(0, min(255, int(bits)))
    max_w = max(1e-6, float(max_width_m))
    return max_w * (1.0 - (b / 255.0))


def width_m_to_bits(width_m: float, *, max_width_m: float = DEFAULT_FRANKA_HAND_MAX_WIDTH_M) -> int:
    """Franka finger width (m) → Robotiq bits (0=open … 255=closed)."""
    max_w = max(1e-6, float(max_width_m))
    w = max(0.0, min(max_w, float(width_m)))
    return int(round(255.0 * (1.0 - (w / max_w))))


def bits_to_speed_m_s(
    speed_bits: int,
    *,
    min_m_s: float = DEFAULT_SPEED_MIN_M_S,
    max_m_s: float = DEFAULT_SPEED_MAX_M_S,
) -> float:
    """Robotiq speed bits (0–255) → Franka gripper speed (m/s)."""
    b = max(0, min(255, int(speed_bits)))
    return float(min_m_s) + (float(max_m_s) - float(min_m_s)) * (b / 255.0)


def bits_to_force_n(
    force_bits: int,
    *,
    min_n: float = DEFAULT_FORCE_MIN_N,
    max_n: float = DEFAULT_FORCE_MAX_N,
) -> float:
    """Robotiq force bits (0–255) → Franka grasp force (N)."""
    b = max(0, min(255, int(force_bits)))
    return float(min_n) + (float(max_n) - float(min_n)) * (b / 255.0)
