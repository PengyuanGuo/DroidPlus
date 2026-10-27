"""Standalone CLI/UI episodes must not suppress OPEN while the Hand is busy."""
from unittest.mock import Mock, call, patch
import numpy as np
import pytest

from droid_plus.eval.episode_runner import EpisodeConfig, SessionConfig, run_episode


def run(gripper_targets, dry_run=False):
    droid = Mock()
    droid.gripper.gripper_state.return_value = {'position_frac': .7, 'busy': True}
    droid.get_left_image.return_value = np.zeros((10, 10, 3), np.uint8)
    droid.get_wrist_image.return_value = np.zeros((10, 10, 3), np.uint8)
    droid.get_current_joint_state.return_value = {'positions': [0.] * 7}
    policy = Mock()
    policy.infer_chunk.return_value = np.c_[np.zeros((len(gripper_targets), 7)), gripper_targets]
    with patch('droid_plus.eval.episode_runner.RateLimiter'), patch('droid_plus.eval.episode_runner.tqdm'):
        run_episode(config=EpisodeConfig(instruction='put banana in bin', action_step_limit=len(gripper_targets)),
                    session=SessionConfig(record=False, dry_run=dry_run), droid=droid, policy=policy,
                    episode_idx=0, recorder=None, should_stop=lambda: False)
    return droid


def test_busy_hand_receives_close_then_open_without_duplicate_transitions():
    droid = run([1., 1., 0., 0.])
    transitions = [c for c in droid.gripper.mock_calls if c[0] in ('open_async', 'close_async')]
    assert transitions == [call.close_async(wait=False), call.open_async(wait=False)]
    assert droid.set_target_joint_state.call_count == 4
    droid.stop.assert_called_once()


def test_first_open_is_sent_even_if_previous_episode_left_hand_closed():
    droid = run([0., 0.])
    droid.gripper.open_async.assert_called_once_with(wait=False)
    droid.gripper.close_async.assert_not_called()


def test_dry_run_has_no_gripper_or_arm_commands():
    droid = run([1., 0.], dry_run=True)
    droid.gripper.open_async.assert_not_called()
    droid.gripper.close_async.assert_not_called()
    droid.set_target_joint_state.assert_not_called()
    droid.stop.assert_not_called()
