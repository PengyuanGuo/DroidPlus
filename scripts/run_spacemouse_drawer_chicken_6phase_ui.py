#!/usr/bin/env python3
"""Collect drawer-search demonstrations with SpaceMouse and six labelled phases.

The chicken leg is hidden in one of two drawers. Open one drawer and return home;
during open-and-home, check that the drawer opened and whether the leg is inside.
Empty (left open) or not opened: pre-open again (other drawer, or retry). Leg seen:
grasp it and place it in the blue bin. The episode ends at place.

Phases: 0 pre-open, 1 open-and-home, 2 pre-grasp, 3 grasp, 4 pre-place, 5 place.
Leg in the first drawer:              0 1 2 3 4 5
Leg in the second drawer (or retry):  0 1 0 1 2 3 4 5
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_teleop_ui as ui

PHASE_NAMES = ('pre-open', 'open-and-home', 'pre-grasp', 'grasp', 'pre-place', 'place')
PHASE_TRANSITIONS = {
    0: (1,),     # pre-open -> open-and-home
    1: (0, 2),   # empty or not opened -> pre-open (other drawer / retry); leg seen -> pre-grasp
    2: (3,),     # pre-grasp -> grasp
    3: (4,),     # grasp -> pre-place
    4: (5,),     # pre-place -> place
}
FINAL_PHASES = (5,)
INSTRUCTION = ('Find the chicken leg hidden in one of the two drawers: open one drawer at a time and '
               'check inside, then pick up the chicken leg and place it in the blue bin.')
FLOW_GUIDANCE = ('Per drawer: 0 pre-open, 1 open-and-home (check the drawer opened and look for the leg). '
                 'Empty or not opened: select 0 again (other drawer or retry). Leg seen: 2 pre-grasp, '
                 '3 grasp, 4 pre-place, 5 place, then stop the episode.')


def main() -> None:
    if any(arg == '--controller' or arg.startswith('--controller=') for arg in sys.argv[1:]):
        raise SystemExit('This entry point always uses SpaceMouse; omit --controller.')
    ui.configure_phase_profile(
        PHASE_NAMES, save_prompts=True, required_cameras=('left', 'wrist'),
        phase_transitions=PHASE_TRANSITIONS, final_phases=FINAL_PHASES, flow_guidance=FLOW_GUIDANCE,
    )
    defaults = [
        '--controller', 'spacemouse',
        '--gripper', 'franka',
        '--rotation', 'yaw',  # XYZ + yaw; the gripper keeps its starting tilt
        '--left-camera-id', '31262010',
        '--wrist-camera-id', '14633207',
        '--task', 'drawer_chicken_search_6phase',
        '--instruction', INSTRUCTION,
        '--output-dir', 'output/spacemouse_drawer_chicken_6phase',
    ]
    if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        print(__doc__ + '\n')
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    ui.main()


if __name__ == '__main__':
    main()
