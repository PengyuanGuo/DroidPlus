"""Metadata survives SpaceMouse failures and the wait for operator labels."""
import json
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from droid_plus.datagen import spacemouse_runner as runner
from droid_plus.datagen.teleop_runner import finalize_teleop_episode_recording
from droid_plus.eval.episode_runner import EpisodeConfig
from droid_plus.logging import EpisodeRecorder


@pytest.mark.parametrize('failure', [None, 'input', 'camera', 'interrupt'])
def test_metadata_lifecycle(tmp_path, monkeypatch, failure):
    recorder = EpisodeRecorder(str(tmp_path), 0, cameras=['left'])
    meta_path = tmp_path / 'episode_000' / 'meta.json'
    recorded = threading.Event()
    droid = Mock(left_camera_id='left', wrist_camera_id='', right_camera_id='')
    droid.get_current_joint_state.return_value = {'positions': np.zeros(7)}

    def camera(**kwargs):
        if failure == 'camera':
            raise RuntimeError('camera unavailable')
        return np.zeros((2, 2, 3), dtype=np.uint8)

    droid.get_left_image.side_effect = camera
    original_record = recorder.record_step

    def record(**kwargs):
        original_record(**kwargs)
        recorded.set()

    recorder.record_step = record
    controller = Mock(q=np.zeros(7), status='ready')
    controller.step.return_value = (np.zeros(7), None)

    def make_controller(*args):
        assert json.loads(meta_path.read_text())['recording_status'] == 'recording'
        return controller

    monkeypatch.setattr(runner, 'CartesianController', make_controller)
    mouse = Mock()
    calls = 0

    def read():
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == 'camera':
                # Deterministically wait until the worker has saved its failure.
                return np.zeros(6), False, False
            assert recorded.wait(2)
            if failure == 'input':
                raise RuntimeError('input disconnected')
            if failure == 'interrupt':
                raise KeyboardInterrupt()
        return np.zeros(6), False, False

    mouse.read.side_effect = read
    session = runner.TeleopSessionConfig(
        dry_run=True, rate_hz=100, record_rate_hz=100, policy_name='teleop_spacemouse',
    )
    config = EpisodeConfig(task='test', instruction='collect a demonstration', action_step_limit=3)
    kwargs = dict(config=config, session=session, droid=droid, teleop=mouse,
                  gripper_initialized=False, pin_model=None, pin_data=None, ee_frame='tool',
                  recorder=recorder, should_stop=lambda: False)
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else RuntimeError):
            runner.run_spacemouse_episode(**kwargs)
        meta = json.loads(meta_path.read_text())
        assert meta['recording_status'] == 'interrupted'
        assert meta['recording_error']
        if failure == 'camera':
            assert 'left camera left' in meta['recording_error']
            assert 'RuntimeError: camera unavailable' in meta['recording_error']
        assert meta['valid'] is False
        assert recorder._closed
    else:
        result = runner.run_spacemouse_episode(**kwargs)
        meta = json.loads(meta_path.read_text())
        assert meta['recording_status'] == 'pending_labels'
        assert not recorder._closed
        finalize_teleop_episode_recording(
            result=result, config=config, session=session, episode_valid=True, episode_success=True,
        )
        meta = json.loads(meta_path.read_text())
        assert meta['recording_status'] == 'complete'
        assert meta['valid'] is True and meta['success'] is True
        assert recorder._closed
    rows = [json.loads(line) for line in (tmp_path / 'episode_000' / 'steps.jsonl').read_text().splitlines()]
    assert meta['step_count'] == len(rows)
    assert meta['policy']['name'] == 'teleop_spacemouse'
    assert meta['instruction'] == config.instruction
    assert all((tmp_path / 'episode_000' / row['images']['left']).exists() for row in rows)
