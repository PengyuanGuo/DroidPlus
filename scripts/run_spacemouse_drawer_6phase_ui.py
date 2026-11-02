#!/usr/bin/env python3
"""SpaceMouse drawer/cube collection with the shared six-phase recording UI."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_teleop_drawer_6phase_ui as drawer_ui


def main() -> None:
    if any(arg == '--controller' or arg.startswith('--controller=') for arg in sys.argv[1:]):
        raise SystemExit('This entry point always uses SpaceMouse; omit --controller.')
    # The shared profile supplies the camera IDs, Franka Hand, and phase prompts.
    # Keep SpaceMouse recordings separate; explicit user flags still win.
    sys.argv[1:1] = [
        '--controller', 'spacemouse',
        '--task', 'drawer_cube_6phase_spacemouse',
        '--output-dir', 'output/spacemouse_drawer_6phase',
    ]
    drawer_ui.main()


if __name__ == '__main__':
    main()
