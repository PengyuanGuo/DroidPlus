"""Hardware-free drawer/chicken-leg search profile: launcher, branching review, UI."""
import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException

from droid_plus.datagen.teleop_runner import TeleopSessionConfig
from droid_plus.eval.episode_runner import EpisodeConfig, EpisodeResult
from droid_plus.logging.phase_labels import review_phase_labels
from scripts import run_spacemouse_drawer_chicken_6phase_ui as chicken
from scripts import run_teleop_ui as ui

FIRST_DRAWER = [0, 1, 2, 3, 4, 5]
SECOND_DRAWER = [0, 1, 0, 1, 2, 3, 4, 5]


@pytest.fixture
def profile(monkeypatch):
    for name in ('PHASE_NAMES', 'SAVE_PHASE_PROMPTS', 'REQUIRED_CAMERAS', 'PHASE_REPETITIONS',
                 'PHASE_TRANSITIONS', 'FINAL_PHASES', 'PHASE_FLOW_GUIDANCE'):
        monkeypatch.setattr(ui, name, getattr(ui, name))
    ui.configure_phase_profile(
        chicken.PHASE_NAMES, save_prompts=True, required_cameras=('left', 'wrist'),
        phase_transitions=chicken.PHASE_TRANSITIONS, final_phases=chicken.FINAL_PHASES,
        flow_guidance=chicken.FLOW_GUIDANCE,
    )


def test_launcher_defaults_and_overrides(profile, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['chicken', '--output-dir', 'custom-output'])
    main = Mock()
    monkeypatch.setattr(ui, 'main', main)
    chicken.main()
    main.assert_called_once()
    assert ui.PHASE_NAMES == ('pre-open', 'open-and-home', 'pre-grasp', 'grasp', 'pre-place', 'place')
    assert ui.PHASE_TRANSITIONS == chicken.PHASE_TRANSITIONS and ui.FINAL_PHASES == (5,)
    assert sys.argv[sys.argv.index('--controller') + 1] == 'spacemouse'
    assert sys.argv[sys.argv.index('--rotation') + 1] == 'yaw'
    assert 'chicken leg' in sys.argv[sys.argv.index('--instruction') + 1]
    assert sys.argv[-2:] == ['--output-dir', 'custom-output']


@pytest.mark.parametrize('phases, needs_review', [
    (FIRST_DRAWER, False),
    (SECOND_DRAWER, False),
    ([0, 1, 0, 1, 0, 1, 2, 3, 4, 5], False),  # a failed open retried
    ([0, 1], True),                     # ended without finding the leg
    ([0, 1, 0, 1], True),
    ([0, 1, 2, 3], True),               # grasped but not placed
    ([0, 1, 3, 4, 5], True),            # skipped pre-grasp
    ([0, 2, 3, 4, 5], True),            # skipped open-and-home
    ([1, 2, 3, 4, 5], True),            # did not start at pre-open
])
def test_branching_review(phases, needs_review):
    labels = [phase for phase in phases for _ in range(8)]
    review = review_phase_labels(labels, min_frames=8, transitions=chicken.PHASE_TRANSITIONS,
                                 final_phases=chicken.FINAL_PHASES)
    assert review['needs_review'] is needs_review, review['issues']


def test_invalid_profile_rejected(monkeypatch):
    for name in ('PHASE_NAMES', 'PHASE_TRANSITIONS', 'FINAL_PHASES'):
        monkeypatch.setattr(ui, name, getattr(ui, name))
    with pytest.raises(ValueError):
        ui.configure_phase_profile(chicken.PHASE_NAMES, phase_transitions={0: (7,)}, final_phases=(6,))
    with pytest.raises(ValueError):
        ui.configure_phase_profile(chicken.PHASE_NAMES, phase_transitions=chicken.PHASE_TRANSITIONS)


def make_state(tmp_path):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    droid = Mock(left_camera_id='31262010', wrist_camera_id='14633207', right_camera_id='')
    droid.get_left_image.return_value = rgb
    droid.get_wrist_image.return_value = rgb
    return ui.AppState(
        droid=droid, teleop=Mock(), gripper_initialized=False,
        pin_model=None, pin_data=None, ee_frame='tool',
        session=TeleopSessionConfig(policy_name='teleop_spacemouse',
                                   controller_info={'controller': 'spacemouse', 'rotation': 'full'}),
        so101_port='SpaceMouse (auto)', base_run_dir=str(tmp_path),
        default_instruction=chicken.INSTRUCTION, default_task='drawer_chicken_search_6phase', status='running',
    )


@pytest.mark.parametrize('phases, needs_review', [(SECOND_DRAWER, False), ([0, 1, 0, 1], True)])
def test_ui_records_branching_episode(profile, tmp_path, monkeypatch, phases, needs_review):
    state = make_state(tmp_path)
    state.current_config = EpisodeConfig(task=state.default_task, instruction=state.default_instruction)
    rgb = state.droid.get_left_image()

    def record_episode(**kwargs):
        recorder = kwargs['recorder']
        for seq, phase in enumerate([phase for phase in phases for _ in range(8)]):
            recorder.record_step(seq, left_rgb=rgb, wrist_rgb=rgb,
                                 state_positions=np.zeros(8), state_velocities=np.zeros(8),
                                 action_positions=np.zeros(8), action_velocities=np.zeros(8), phase_index=phase)
        result = EpisodeResult(0, len(phases) * 8, 1, 20, [], recorder, False, True)
        result.actual_record_rate_hz = 15.0
        return result

    monkeypatch.setattr(ui, 'run_spacemouse_episode', Mock(side_effect=record_episode))
    monkeypatch.setattr(ui, 'run_teleop_episode', Mock(side_effect=AssertionError('SO-101 runner')))
    monkeypatch.setattr(ui, 'compute_and_save_ee_trajectory_single', Mock())
    ui._run_episode_thread(state)
    episode = tmp_path / 'episode_000'
    meta = json.loads((episode / 'meta.json').read_text())
    review = meta['phase_annotation']
    assert review['needs_review'] is needs_review, review['issues']
    assert 'expected_phase_sequence' not in review
    assert meta['phase_transitions']['1'] == [0, 2] and meta['final_phases'] == [5]
    assert meta['phase_prompt_mapping']['5'] == 'place'
    rows = [json.loads(line) for line in (episode / 'steps.jsonl').read_text().splitlines()]
    assert [r['phase_prompt'] for r in rows[::8]] == [chicken.PHASE_NAMES[p] for p in phases]


def test_ui_guidance_and_phase_keys(profile, tmp_path, monkeypatch):
    state = make_state(tmp_path)
    monkeypatch.setattr(ui, '_state', state)
    html = ui.root().body.decode()
    assert 'Leg seen: 2 pre-grasp, 3 grasp, 4 pre-place, 5 place' in html
    for phase in range(6):
        assert ui.episode_phase({'phase_index': phase})['phase_index'] == phase
    with pytest.raises(HTTPException):
        ui.episode_phase({'phase_index': 6})
