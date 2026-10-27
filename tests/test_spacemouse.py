"""Hardware-free input, kinematics, and episode regression tests."""
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from droid_plus.datagen.spacemouse import CartesianController, SpaceMouse, SpaceMouseConfig, axis_twist


def test_axis_mapping_deadzone_and_vector_speed():
    cfg = SpaceMouseConfig(axis_map=(-2, 1, 3, 4, 5, 6))
    assert np.all(axis_twist([0.05] * 6, cfg) == 0)
    twist = axis_twist([0, 1, 0, 1, 1, 1], cfg)
    assert twist[0] == -cfg.linear_speed
    assert np.linalg.norm(twist[3:]) == pytest.approx(cfg.angular_speed)
    with pytest.raises(ValueError):
        axis_twist([np.nan] * 6, cfg)


@pytest.mark.parametrize('kwargs', [dict(deadzone=1), dict(linear_speed=-1), dict(timeout=float('nan')),
                                    dict(axis_map=(1, 1, 2, 3, 4, 5)), dict(enable_button=1)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SpaceMouseConfig(**kwargs)


def mouse_sample(axes=(0,) * 6, buttons=(0, 0), age=0):
    mouse = SpaceMouse.__new__(SpaceMouse)
    mouse.config = SpaceMouseConfig()
    mouse._lock = threading.Lock()
    mouse._error = None
    mouse._sample = (axes, buttons, time.monotonic() - age)
    return mouse


def test_stale_active_input_fails_but_idle_holds():
    assert not mouse_sample(age=10).read()[1]
    for mouse in [mouse_sample(buttons=(1, 0), age=10), mouse_sample(axes=(1, 0, 0, 0, 0, 0), age=10)]:
        with pytest.raises(RuntimeError, match='timed out'):
            mouse.read()
    with pytest.raises(RuntimeError, match='unavailable'):
        mouse_sample(buttons=(0,)).read()


@pytest.fixture
def controller():
    import pinocchio as pin
    # Seven serial revolute joints, with alternating axes and a raised base.
    model = pin.Model()
    parent = 0
    for i in range(7):
        joint = (pin.JointModelRZ, pin.JointModelRY, pin.JointModelRX)[i % 3]()
        placement = pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.5 if i == 0 else 0.1]))
        parent = model.addJoint(parent, joint, placement, f'joint{i}')
    model.addFrame(pin.Frame('tool', parent, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    model.lowerPositionLimit[:] = -2.5
    model.upperPositionLimit[:] = 2.5
    return CartesianController(model, model.createData(), 'tool', np.zeros(7), SpaceMouseConfig(), 0.2)


def test_clutch_gripper_edges_and_joint_speed(controller):
    c = controller
    twist = np.array([0, 0.05, 0, 0, 0, 0])
    q, grip = c.step(twist, True, True, c.q, 0.01)
    assert np.all(q == 0) and grip is None  # held at startup cannot arm
    c.step(np.zeros(6), False, False, c.q, 0.01)
    q, grip = c.step(twist, True, True, c.q, 0.01)
    assert grip == 0
    assert 0 < np.max(np.abs(q)) <= c.config.joint_speed * 0.01
    _, grip = c.step(twist, True, True, q, 0.01)
    assert grip is None
    held = c.q.copy()
    q, _ = c.step(twist, False, False, c.q, 0.01)
    np.testing.assert_array_equal(q, held)
    _, grip = c.step(np.zeros(6), True, True, c.q, 0.01)
    assert grip == 100


def test_fk_direction_floor_and_tracking(controller):
    c = controller
    c.step(np.zeros(6), False, False, c.q, 0.01)
    c.safe(c.q)
    before = c.data.oMf[c.frame].translation.copy()
    c.step(np.array([0, .05, 0, 0, 0, 0]), True, False, c.q, .01)
    c.safe(c.q)
    assert c.data.oMf[c.frame].translation[1] > before[1]
    held = c.q.copy()
    c.min_z = 100
    c.step(np.array([0, .05, 0, 0, 0, 0]), True, False, c.q, .01)
    np.testing.assert_array_equal(c.q, held)
    with pytest.raises(RuntimeError, match='tracking error'):
        c.step(np.zeros(6), False, False, c.q + 1, .01)
    assert not c.safe(np.full(7, 3.0))


def test_runner_dry_run_and_disconnect_stop(controller, monkeypatch):
    from droid_plus.datagen import spacemouse_runner as runner
    from droid_plus.eval.episode_runner import EpisodeConfig
    mouse = mouse_sample()
    mouse.read = Mock(return_value=(np.zeros(6), False, False))
    droid = Mock()
    droid.get_current_joint_state.return_value = {'positions': controller.q}
    monkeypatch.setattr(runner, 'CartesianController', lambda *args: controller)
    kwargs = dict(config=EpisodeConfig(action_step_limit=2), droid=droid, teleop=mouse,
                  gripper_initialized=True, pin_model=None, pin_data=None, ee_frame='tool',
                  recorder=None, should_stop=lambda: False)
    runner.run_spacemouse_episode(session=runner.TeleopSessionConfig(dry_run=True, record=False), **kwargs)
    droid.set_target_joint_state.assert_not_called()
    droid.stop.assert_not_called()
    droid.gripper.close_async.assert_not_called()
    mouse.read.side_effect = RuntimeError('disconnected')
    with pytest.raises(RuntimeError, match='disconnected'):
        runner.run_spacemouse_episode(session=runner.TeleopSessionConfig(record=False), **kwargs)
    droid.stop.assert_called_once()


def test_uninitialized_input_cannot_arm(controller):
    mouse = mouse_sample()
    mouse._sample = None
    twist, enabled, grip = mouse.read()
    controller.step(twist, enabled, grip, controller.q, .01)
    assert not controller.armed


def test_live_runner_records_joint_targets_and_phase(controller, monkeypatch):
    from droid_plus.datagen import spacemouse_runner as runner
    from droid_plus.eval.episode_runner import EpisodeConfig
    mouse = mouse_sample()
    mouse.read = Mock(side_effect=[(np.zeros(6), False, False),
                                   (np.array([0, .05, 0, 0, 0, 0]), True, True)])
    droid = Mock(left_camera_id='', wrist_camera_id='', right_camera_id='')
    droid.get_current_joint_state.side_effect = lambda: {'positions': controller.q.copy()}
    droid.get_gripper_obs_value.return_value = 0.0
    recorder = Mock(episode_idx=0)
    monkeypatch.setattr(runner, 'CartesianController', lambda *args: controller)
    result = runner.run_spacemouse_episode(
        config=EpisodeConfig(action_step_limit=2),
        session=runner.TeleopSessionConfig(rate_hz=100, record_rate_hz=100),
        droid=droid, teleop=mouse, gripper_initialized=True,
        pin_model=None, pin_data=None, ee_frame='tool', recorder=recorder,
        should_stop=lambda: False, get_phase=lambda: 3,
    )
    assert result.seq == 2
    assert droid.set_target_joint_state.call_count == 2
    droid.gripper.close_async.assert_called_once()
    droid.stop.assert_called_once()
    recorded = recorder.record_step.call_args.kwargs
    assert recorded['phase_index'] == 3
    np.testing.assert_allclose(recorded['action_positions'][:7], controller.q)
    assert recorded['action_positions'][7] == 1.0


def test_roll_pitch_input_ignored(controller):
    cfg = SpaceMouseConfig()
    np.testing.assert_array_equal(axis_twist([0, 0, 0, 1, -1, 0], cfg), np.zeros(6))
    c = controller
    c.step(np.zeros(6), False, False, c.q, .01)
    before = c.q.copy()
    c.step(np.array([0, 0, 0, .2, -.2, 0]), True, False, c.q, .01)
    np.testing.assert_array_equal(c.q, before)


def test_panda_yaw_holds_tilt_and_timing_is_rate_independent():
    from pathlib import Path

    import pinocchio as pin

    from droid_plus.analysis.end_effector_pose import build_model_from_urdf
    from droid_plus.services.franky_client import HOME_POSITION
    source = Path(__file__).resolve().parents[1] / 'models/franka_panda/panda.urdf'
    results = []
    for dt in (.01, .02):
        m, d, f = build_model_from_urdf(source.read_text())
        c = CartesianController(m, d, f, HOME_POSITION, SpaceMouseConfig(), .23)
        initial = c.orientation.copy()
        c.step(np.zeros(6), False, False, c.q, dt)
        for _ in range(round(1 / dt)):
            c.step(np.array([.003, 0, 0, 0, 0, .03]), True, False, c.q, dt)
        c.safe(c.q)
        actual = d.oMf[c.frame].rotation
        expected = pin.exp3(np.array([0., 0., .03])) @ initial
        assert np.linalg.norm(pin.log3(expected @ actual.T)) < 1e-4
        results.append(c.q.copy())
    np.testing.assert_allclose(results[0], results[1], atol=1e-4)


def test_control_stall_stops(controller):
    with pytest.raises(RuntimeError, match='stalled'):
        controller.step(np.zeros(6), True, False, controller.q, .3)


def test_deoxys_button_behavior(controller):
    c = controller
    c.config = SpaceMouseConfig(button_mode='deoxys')
    c.step(np.zeros(6), True, False, c.q, .01)
    assert c.armed
    _, grip = c.step(np.zeros(6), True, True, c.q, .01)
    assert grip == 0
    _, grip = c.step(np.zeros(6), True, False, c.q, .01)
    assert grip == 100
    mouse = mouse_sample(buttons=(1, 0))
    mouse.config = c.config
    with pytest.raises(RuntimeError, match='stop button'):
        mouse.read()


def test_slow_camera_does_not_block_control_and_stop_precedes_join(controller, monkeypatch):
    from droid_plus.datagen import spacemouse_runner as runner
    from droid_plus.eval.episode_runner import EpisodeConfig
    camera_started = threading.Event()
    control_progressed = threading.Event()
    robot_stopped = threading.Event()
    writes = []
    droid = Mock(left_camera_id='left', wrist_camera_id='', right_camera_id='')
    droid.get_current_joint_state.side_effect = lambda: {'positions': controller.q.copy()}

    def camera(**kwargs):
        camera_started.set()
        assert control_progressed.wait(2), 'Camera acquisition blocked robot commands'
        assert robot_stopped.wait(2), 'Robot must stop before waiting for recorder'
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def command(*args, **kwargs):
        if kwargs['seq'] >= 3 and camera_started.is_set():
            control_progressed.set()

    droid.get_left_image.side_effect = camera
    droid.set_target_joint_state.side_effect = command
    droid.stop.side_effect = robot_stopped.set
    recorder = Mock(episode_idx=0)
    recorder.record_step.side_effect = lambda **kw: writes.append(kw)
    mouse = mouse_sample()
    mouse.read = Mock(return_value=(np.zeros(6), False, False))
    monkeypatch.setattr(runner, 'CartesianController', lambda *args: controller)
    result = runner.run_spacemouse_episode(
        config=EpisodeConfig(action_step_limit=15),
        session=runner.TeleopSessionConfig(rate_hz=100), droid=droid, teleop=mouse,
        gripper_initialized=False, pin_model=None, pin_data=None, ee_frame='tool',
        recorder=recorder, should_stop=lambda: False, get_phase=lambda: 2,
    )
    assert result.seq == len(writes) == 1
    assert writes[0]['phase_index'] == 2
    assert len(writes[0]['state_positions']) == len(writes[0]['action_positions']) == 8
    assert droid.set_target_joint_state.call_count == 15


def test_recording_snapshot_is_owned_and_worker_errors_surface():
    from droid_plus.datagen.async_recording import AsyncTeleopRecorder
    entered = threading.Event()
    release = threading.Event()
    droid = Mock(left_camera_id='left', wrist_camera_id='', right_camera_id='')

    def camera(**kwargs):
        entered.set()
        assert release.wait(2)
        return np.zeros((2, 2, 3))

    droid.get_left_image.side_effect = camera
    recorder = Mock()
    worker = AsyncTeleopRecorder(droid, recorder, 15, 85, False)
    q = np.arange(7, dtype=float)
    target = q + 1
    worker.submit({'positions': q}, target, 1., 3, 123.)
    assert entered.wait(2)
    q[:] = 99
    target[:] = 99
    release.set()
    worker.close()
    row = recorder.record_step.call_args.kwargs
    np.testing.assert_array_equal(row['state_positions'][:7], np.arange(7))
    np.testing.assert_array_equal(row['action_positions'][:7], np.arange(7) + 1)
    assert row['phase_index'] == 3 and row['t_wall_s'] == 123.

    droid.get_left_image.side_effect = RuntimeError('camera offline')
    worker = AsyncTeleopRecorder(droid, recorder, 15, 85, False)
    worker.submit({'positions': q}, target, 0., 0, 124.)
    worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()
    with pytest.raises(RuntimeError, match='Recording worker failed'):
        worker.close()


def test_runner_reports_transport_latency_on_stall(controller, monkeypatch, capsys):
    from droid_plus.datagen import spacemouse_runner as runner
    from droid_plus.eval.episode_runner import EpisodeConfig
    reads = 0

    def state():
        nonlocal reads
        reads += 1
        if reads == 3:  # Initial pose, first tick, then a slow second tick.
            time.sleep(.27)
        return {'positions': controller.q.copy()}

    droid = Mock()
    droid.get_current_joint_state.side_effect = state
    mouse = mouse_sample()
    mouse.read = Mock(return_value=(np.zeros(6), False, False))
    monkeypatch.setattr(runner, 'CartesianController', lambda *args: controller)
    with pytest.raises(RuntimeError, match='stalled'):
        runner.run_spacemouse_episode(
            config=EpisodeConfig(action_step_limit=3),
            session=runner.TeleopSessionConfig(record=False), droid=droid, teleop=mouse,
            gripper_initialized=False, pin_model=None, pin_data=None, ee_frame='tool',
            recorder=None, should_stop=lambda: False,
        )
    output = capsys.readouterr().out
    assert 'timing at failure' in output and 'state_get_ms' in output and 'target_post_ms' in output
    droid.stop.assert_called_once()
    assert droid.set_target_joint_state.call_count == 1
