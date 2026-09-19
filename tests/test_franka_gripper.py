# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from droid_plus.services.franka_gripper import (
    bits_to_width_m,
    width_m_to_bits,
)


def test_bits_width_roundtrip_endpoints() -> None:
    assert bits_to_width_m(0, max_width_m=0.08) == 0.08
    assert bits_to_width_m(255, max_width_m=0.08) == 0.0
    assert width_m_to_bits(0.08, max_width_m=0.08) == 0
    assert width_m_to_bits(0.0, max_width_m=0.08) == 255


def test_bits_width_midpoint() -> None:
    w = bits_to_width_m(128, max_width_m=0.08)
    assert abs(w - 0.04) < 1e-6
    assert width_m_to_bits(w, max_width_m=0.08) == 128
