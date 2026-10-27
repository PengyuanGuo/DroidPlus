#!/usr/bin/env python3
"""SO-101 drawer/cube collection: six named phases, mandatory left/wrist images.

Uses DroidPlus's existing teleoperation UI, robot clients, and recording loop.
Phase names are written to every step during collection, not added afterward.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_teleop_ui as ui

PHASE_NAMES = (
    "grasp_handle_open",
    "open_drawer_and_home",
    "pick_cube_and_lift",
    "carry_place_and_home",
    "grasp_handle_close",
    "close_drawer_and_home",
)


def main():
    ui.configure_phase_profile(PHASE_NAMES, save_prompts=True, required_cameras=("left", "wrist"))
    # Explicit defaults avoid silently recording state-only episodes when a new
    # terminal has not sourced the camera environment variables. User flags win.
    defaults = [
        "--gripper", "franka",
        "--left-camera-id", "31262010",
        "--wrist-camera-id", "14633207",
        "--task", "drawer_cube_6phase",
        "--output-dir", "output_drawer_6phase",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    ui.main()


if __name__ == "__main__":
    main()
