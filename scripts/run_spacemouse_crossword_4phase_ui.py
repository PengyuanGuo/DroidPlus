#!/usr/bin/env python3
"""Collect MARS crossword demonstrations with SpaceMouse and four labelled phases.

By default one episode spells MARS: repeat phases 0..3 for M, A, R, then S.
Use --episode-mode letter to record one letter per episode instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_teleop_ui as ui

PHASE_NAMES = ('pre-grasp', 'grasp', 'pre-place', 'place')
WORD = 'MARS'


def main() -> None:
    if any(arg == '--controller' or arg.startswith('--controller=') for arg in sys.argv[1:]):
        raise SystemExit('This entry point always uses SpaceMouse; omit --controller.')
    profile = argparse.ArgumentParser(add_help=False)
    profile.add_argument('--episode-mode', choices=('word', 'letter'), default='word')
    options, remaining = profile.parse_known_args()
    repetitions = len(WORD) if options.episode_mode == 'word' else 1
    instruction = ('Arrange the letter tiles to spell MARS in the crossword.'
                   if options.episode_mode == 'word'
                   else 'Place one letter tile in its position in the crossword word MARS.')
    ui.configure_phase_profile(
        PHASE_NAMES, save_prompts=True, required_cameras=('left', 'wrist'),
        phase_repetitions=repetitions,
    )
    defaults = [
        '--controller', 'spacemouse',
        '--gripper', 'franka',
        '--left-camera-id', '31262010',
        '--wrist-camera-id', '14633207',
        '--task', 'crossword_mars_4phase',
        '--instruction', instruction,
        '--output-dir', 'output/spacemouse_crossword_mars_4phase',
    ]
    if any(arg in ('-h', '--help') for arg in remaining):
        print(__doc__ + '\n')
    sys.argv = [sys.argv[0], *defaults, *remaining]
    ui.main()


if __name__ == '__main__':
    main()
