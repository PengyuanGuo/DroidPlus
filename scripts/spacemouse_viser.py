#!/usr/bin/env python3
"""SpaceMouse teleop simulator in viser: drives a virtual arm, never commands the robot.

Uses the same SpaceMouse reader, Cartesian controller, limits and height floor as
``run_spacemouse_cli.py``. Practise and verify axes here before live control.

    Usage:
    python scripts/spacemouse_viser.py                     # start at HOME_POSITION
    python scripts/spacemouse_viser.py --start nuc         # start at the robot's measured joints (read-only)
    python scripts/spacemouse_viser.py --button-mode deoxys --gripper-button 0 --enable-button 1
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from droid_plus.analysis.end_effector_pose import build_model_from_urdf
from droid_plus.constants import FRANKY_SERVICE_URL
from droid_plus.datagen.safety import DEFAULT_MIN_EE_Z
from droid_plus.datagen.spacemouse import CartesianController, add_controller_arguments, connect_controller
from droid_plus.datagen.spacemouse_viz import TeleopVisualizer
from droid_plus.services.franky_client import HOME_POSITION

DEFAULT_URDF = os.path.join(os.path.dirname(__file__), "..", "models", "franka_panda", "panda.urdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", choices=["home", "nuc"], default="home",
                        help="Initial virtual pose: HOME_POSITION or the robot's measured joints (GET only)")
    parser.add_argument("--franky-service-url", default=FRANKY_SERVICE_URL)
    parser.add_argument("--urdf", default=os.environ.get("FRANKA_URDF_PATH") or os.path.normpath(DEFAULT_URDF))
    parser.add_argument("--min-z", type=float, default=DEFAULT_MIN_EE_Z)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--viser-port", type=int, default=8080)
    add_controller_arguments(parser)
    args = parser.parse_args()
    args.controller = "spacemouse"

    with open(args.urdf) as f:
        model, data, ee_frame = build_model_from_urdf(f.read())
    if args.start == "nuc":
        import requests

        r = requests.get(f"{args.franky_service_url}/joint_state", timeout=5)
        r.raise_for_status()
        q0 = np.asarray(r.json()["positions"], dtype=float)
        print(f"Starting from measured joints {np.round(q0, 3)}")
    else:
        q0 = np.array(HOME_POSITION, dtype=float)

    viz = TeleopVisualizer(model, data, ee_frame, args.min_z, port=args.viser_port)
    mouse = connect_controller(args)
    ctrl = CartesianController(model, data, ee_frame, q0, mouse.config, args.min_z)
    print(f"Simulating ({mouse.config.button_mode} mode). Center the cap and release buttons to arm. Ctrl-C quits.")
    dt = 1.0 / args.rate_hz
    previous = time.monotonic()
    last_print = 0.0
    try:
        while True:
            tick = time.monotonic()
            try:
                twist, enabled, grip = mouse.read()
                q, gripper_percent = ctrl.step(twist, enabled, grip, ctrl.q, tick - previous)
            except RuntimeError as exc:
                if "read failed" in str(exc):
                    raise
                # Live teleop would end the episode here; the simulator re-arms instead.
                print(f"\n[sim] {exc}; re-arming")
                ctrl.armed = False
                twist, enabled, q, gripper_percent = np.zeros(6), False, ctrl.q.copy(), None
            previous = tick
            if gripper_percent is not None:
                print(f"\n[sim] gripper {'CLOSE' if gripper_percent == 0.0 else 'OPEN'}")
            viz.update(target=q, twist=twist, enabled=enabled, gripper_closed=ctrl.closed, status=ctrl.status)
            if tick - last_print >= 0.5:
                last_print = tick
                print(f"\r[sim] {ctrl.status:45s}", end="", flush=True)
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print()
    finally:
        mouse.disconnect()
        viz.close()


if __name__ == "__main__":
    main()
