"""Hardware-free MARS profile, UI, and recorded phase-cycle checks."""
import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException

from droid_plus.datagen.teleop_runner import TeleopSessionConfig
from droid_plus.eval.episode_runner import EpisodeConfig, EpisodeResult
from scripts import run_spacemouse_crossword_4phase_ui as crossword
from scripts import run_teleop_ui as ui


@pytest.fixture
def profile(monkeypatch):
    for name in ('PHASE_NAMES', 'SAVE_PHASE_PROMPTS', 'REQUIRED_CAMERAS', 'PHASE_REPETITIONS'):
        monkeypatch.setattr(ui, name, getattr(ui, name))
    ui.configure_phase_profile(crossword.PHASE_NAMES, save_prompts=True,
                               required_cameras=('left', 'wrist'), phase_repetitions=4)


@pytest.mark.parametrize('mode, repetitions', [('word', 4), ('letter', 1)])
def test_launcher_defaults_and_overrides(profile, monkeypatch, mode, repetitions):
    monkeypatch.setattr(sys, 'argv', ['crossword', '--episode-mode', mode, '--linear-speed', '0.15',
                                     '--output-dir', 'custom-output'])
    main = Mock()
    monkeypatch.setattr(ui, 'main', main)
    crossword.main()
    main.assert_called_once()
    assert ui.PHASE_NAMES == ('pre-grasp', 'grasp', 'pre-place', 'place')
    assert ui.PHASE_REPETITIONS == repetitions
    assert ui.SAVE_PHASE_PROMPTS and ui.REQUIRED_CAMERAS == ('left', 'wrist')
    assert sys.argv[sys.argv.index('--controller') + 1] == 'spacemouse'
    assert 'MARS' in sys.argv[sys.argv.index('--instruction') + 1]
    assert sys.argv[-4:] == ['--linear-speed', '0.15', '--output-dir', 'custom-output']


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
        default_instruction='Arrange the letter tiles to spell MARS in the crossword.',
        default_task='crossword_mars_4phase', status='running',
    )


@pytest.mark.parametrize('phases, needs_review', [
    (list(range(4)) * 4, False),
    (list(range(4)) * 3, True),
    (list(range(4)) * 4 + [0], True),
    ([0, 2, 3] + list(range(4)) * 3, True),
])
def test_ui_records_mars_cycles(profile, tmp_path, monkeypatch, phases, needs_review):
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

    mouse_runner = Mock(side_effect=record_episode)
    arm_runner = Mock(side_effect=AssertionError('SO-101 runner must not be called'))
    monkeypatch.setattr(ui, 'run_spacemouse_episode', mouse_runner)
    monkeypatch.setattr(ui, 'run_teleop_episode', arm_runner)
    monkeypatch.setattr(ui, 'compute_and_save_ee_trajectory_single', Mock())
    ui._run_episode_thread(state)
    mouse_runner.assert_called_once()
    arm_runner.assert_not_called()
    assert state.status == 'idle' and len(state.history) == 1
    episode = tmp_path / 'episode_000'
    meta = json.loads((episode / 'meta.json').read_text())
    assert meta['policy']['name'] == 'teleop_spacemouse'
    assert meta['instruction'] == state.default_instruction
    assert meta['task'] == state.default_task
    assert meta['phase_annotation']['expected_phase_sequence'] == list(range(4)) * 4
    assert meta['phase_annotation']['needs_review'] is needs_review
    rows = [json.loads(line) for line in (episode / 'steps.jsonl').read_text().splitlines()]
    assert len(rows) == len(phases) * 8 == meta['step_count']
    for row in rows:
        assert row['phase_prompt'] == crossword.PHASE_NAMES[row['phase_index']]
        assert list(row)[-2:] == ['phase_index', 'phase_prompt']
        assert all((episode / image).is_file() for image in row['images'].values())


def test_ui_defaults_and_phase_controls(profile, tmp_path, monkeypatch):
    state = make_state(tmp_path)
    monkeypatch.setattr(ui, '_state', state)
    html = ui.root().body.decode()
    assert 'teleop_service_state_v1:crossword_mars_4phase' in html
    assert 'Repeat phases 0–3 in order 4 times' in html
    assert '__STORAGE_KEY_JSON__' not in html
    assert ui.get_config()['default_instruction'] == state.default_instruction
    for phase in range(4):
        assert ui.episode_phase({'phase_index': phase})['phase_index'] == phase
    with pytest.raises(HTTPException):
        ui.episode_phase({'phase_index': 4})
    state.status = 'idle'
    thread = Mock()
    monkeypatch.setattr(ui.threading, 'Thread', Mock(return_value=thread))
    ui.episode_start({})
    assert state.current_config.instruction == state.default_instruction
    assert state.current_config.task == state.default_task
    assert state.phases.get() == 0
    thread.start.assert_called_once()
