# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import time
from typing import Any, List

import numpy as np
import requests

from droid_plus.constants import FRANKY_SERVICE_URL

HOME_POSITION: List[float] = [0.0, -0.40, 0.0, -1.9, 0.0, 1.5, math.pi / 4]
# GRIPPER_MOUNT_POSITION: List[float] = [ 0.0, -0.40, 0.0, -1.9,  0.0,  2.8, 0.0 ]

ALT_POSITION_1 = [ 0.0,  0.36324462, 0.0,  -1.97732031,  0.0,  2.44544444, 0.57125038]
ALT_POSITION_2 = [ 0.0, 0.05, 0.0, -1.89402282,  0.0,  2.52416387, 0.6005711 ]
ALT_POSITION_3 = [ 0.4, 0.12361217, 0.0, -1.89402282,  0.0,  2.52416387, 0.6005711 ]
ALT_POSITION_4 = [ -0.4, 0.12361217, 0.0, -1.89402282,  0.0,  2.52416387, 0.6005711 ]

class FrankyClient:
    """
    Thin HTTP client for `franky_service.py`.

    Endpoints used:
      - POST /stop
      - POST /target_joint_state
      - GET  /target_joint_state
      - GET  /joint_state
      - GET  /health
      - GET/POST /command_timeout
    """

    def __init__(self, base_url: str | None = None, *, timeout_s: float = 5.0):
        self.base_url = (base_url or FRANKY_SERVICE_URL).rstrip("/")
        self.timeout_s = float(timeout_s)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def stop(self) -> dict[str, Any]:
        r = requests.post(self._url("/stop"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def set_target_joint_state(
        self,
        positions: Any,
        velocities: Any | None = None,
        *,
        seq: int | None = None,
    ) -> dict[str, Any]:
        if velocities is None:
            velocities = [0.0] * 7

        # Accept numpy arrays/lists/tuples
        pos = [float(x) for x in np.asarray(positions, dtype=float).tolist()]
        vel = [float(x) for x in np.asarray(velocities, dtype=float).tolist()]
        if len(pos) != 7 or len(vel) != 7:
            raise ValueError("positions and velocities must have length 7")

        payload = {"positions": pos, "velocities": vel, "seq": seq}
        r = requests.post(self._url("/target_joint_state"), json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def get_target_joint_state(self) -> dict[str, Any]:
        r = requests.get(self._url("/target_joint_state"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def get_current_joint_state(self) -> dict[str, Any]:
        r = requests.get(self._url("/joint_state"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict[str, Any]:
        r = requests.get(self._url("/health"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def get_command_timeout(self) -> dict[str, Any]:
        r = requests.get(self._url("/command_timeout"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def set_command_timeout(self, command_timeout_s: float) -> dict[str, Any]:
        r = requests.post(
            self._url("/command_timeout"),
            json={"command_timeout_s": float(command_timeout_s)},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()


def _default_client() -> FrankyClient:
    return FrankyClient()


def stop():
    """Send stop command to the service."""
    return _default_client().stop()


def set_target_joint_state(positions, velocities=None, seq: int | None = None):
    """Set target joint positions/velocities via /target_joint_state."""
    return _default_client().set_target_joint_state(positions, velocities, seq=seq)


def get_target_joint_state():
    """Get the latest accepted target from /target_joint_state."""
    return _default_client().get_target_joint_state()


def get_current_joint_state():
    """Get current measured joint state from /joint_state."""
    return _default_client().get_current_joint_state()


if __name__ == "__main__":
    # Example: move to home
    print("Stopping first...")
    print(stop())

    print("Sending HOME_POSITION target...")
    resp = set_target_joint_state(HOME_POSITION)
    print("Target response:", resp)
    time.sleep(2)
    print(stop())
