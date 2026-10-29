"""SpaceMouse input and bounded differential IK for Franka teleoperation."""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpaceMouseConfig:
    linear_speed: float = 0.10  # m/s, vector norm
    angular_speed: float = 0.3  # rad/s, vector norm
    joint_speed: float = 0.5  # rad/s per joint; 10 cm/s needs up to ~0.34 near home
    deadzone: float = 0.1
    timeout: float = 0.25
    button_mode: str = "clutch"
    enable_button: int = 0
    gripper_button: int = 1
    axis_map: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def __post_init__(self):
        if self.button_mode not in ('clutch', 'deoxys'):
            raise ValueError('button_mode must be clutch or deoxys')
        for name in ('linear_speed', 'angular_speed', 'joint_speed', 'timeout'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.deadzone) or not 0 <= self.deadzone < 1:
            raise ValueError('deadzone must be in [0, 1)')
        if self.enable_button < 0 or self.gripper_button < 0 or self.enable_button == self.gripper_button:
            raise ValueError('Enable and gripper buttons must be distinct nonnegative indices')
        if sorted(abs(x) for x in self.axis_map) != list(range(1, 7)):
            raise ValueError('axis_map must be a signed permutation of 1,2,3,4,5,6')


def axis_twist(axes, config):
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (6,) or not np.all(np.isfinite(axes)):
        raise ValueError('Invalid SpaceMouse axes')
    axes = np.array([np.sign(i) * axes[abs(i) - 1] for i in config.axis_map])
    axes = np.sign(axes) * np.maximum(np.abs(axes) - config.deadzone, 0) / (1 - config.deadzone)
    axes[3:5] = 0.0  # XYZ translation and base-frame yaw only.
    for group, speed in ((slice(0, 3), config.linear_speed), (slice(3, 6), config.angular_speed)):
        axes[group] *= speed / max(1.0, np.linalg.norm(axes[group]))
    return axes


class SpaceMouse:
    """Drain HID reports independently of recording; never reuse stale motion."""
    def __init__(self, config=SpaceMouseConfig(), device_name=None):
        import pyspacemouse

        self.config = config
        self.device = pyspacemouse.open(
            device=device_name, axis_convention=pyspacemouse.AxisConvention.HID_Z_UP,
        )
        if not self.device:
            raise RuntimeError('No SpaceMouse found; check USB connection and HID permissions')
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sample = None
        self._error = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        previous_stamp = 0.0
        try:
            while not self._stop.is_set():
                state = self.device.read()
                if state is None or not self.device.connected:
                    raise RuntimeError('SpaceMouse disconnected')
                if state.t != previous_stamp:
                    previous_stamp = state.t
                    axes = (state.x, state.y, state.z, state.roll, state.pitch, state.yaw)
                    with self._lock:
                        self._sample = (axes, tuple(state.buttons), time.monotonic())
                self._stop.wait(0.001)
        except Exception as exc:
            with self._lock:
                self._error = exc

    def read(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError('SpaceMouse read failed') from self._error
            sample = self._sample
        if sample is None:
            # An uninitialized device must not satisfy the release-to-arm check.
            return np.zeros(6), True, True
        axes, buttons, received = sample
        if max(self.config.enable_button, self.config.gripper_button) >= len(buttons):
            raise RuntimeError('Configured SpaceMouse button is unavailable')
        twist = axis_twist(axes, self.config)
        enabled = bool(buttons[self.config.enable_button])
        grip = bool(buttons[self.config.gripper_button])
        if time.monotonic() - received > self.config.timeout and np.any(twist):
            # The device streams reports only while the cap is deflected, so stale
            # motion means lost input. A held button on a centered cap sends nothing.
            raise RuntimeError('SpaceMouse input timed out; episode stopped')
        if self.config.button_mode == 'deoxys':
            # In this mode the configured enable button acts as stop/reset.
            if enabled:
                raise RuntimeError('SpaceMouse stop button pressed')
            enabled = True
        return twist, enabled, grip

    def disconnect(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.device.close()


class CartesianController:
    """Base-frame velocity control, using a damped Jacobian inverse.

    Targets are integrated from the last accepted command, with measured joint
    tracking checks. Rejected steps do not accumulate Cartesian error.
    """
    def __init__(self, model, data, ee_frame, q, config, min_z):
        import pinocchio as pin

        if model is None or data is None or not ee_frame:
            raise ValueError('SpaceMouse requires the robot URDF and Pinocchio')
        if model.nq < 7 or model.nv < 7 or not model.existFrame(ee_frame):
            raise ValueError('Invalid Franka kinematic model')
        for i in range(1, 8):
            joint = model.joints[i]
            if joint.nq != 1 or joint.nv != 1 or joint.idx_q != i - 1 or joint.idx_v != i - 1:
                raise ValueError('Expected the first seven model coordinates to be arm joints')
        self.pin, self.model, self.data = pin, model, data
        self.frame = model.getFrameId(ee_frame)
        self.config, self.min_z = config, min_z
        self.q = np.asarray(q, dtype=float).copy()
        self.lower = np.asarray(model.lowerPositionLimit[:7]) + 0.01
        self.upper = np.asarray(model.upperPositionLimit[:7]) - 0.01
        self.armed = False
        self.last_grip = False
        self.closed = False
        self.status = "waiting for release and centered cap"
        if not self.safe(self.q):
            raise ValueError('Initial robot pose violates joint limits or minimum EE height')
        self.orientation = self.data.oMf[self.frame].rotation.copy()

    def full_q(self, q):
        full = self.pin.neutral(self.model)
        full[:7] = q
        return full

    def safe(self, q):
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return False
        if np.any(q < self.lower) or np.any(q > self.upper) or not np.isfinite(self.min_z):
            return False
        self.pin.framesForwardKinematics(self.model, self.data, self.full_q(q))
        return bool(self.data.oMf[self.frame].translation[2] >= self.min_z)

    def step(self, twist, enabled, grip, measured, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Control timestep must be finite and positive")
        if dt > 0.25:
            raise RuntimeError('Control loop stalled for over 250 ms; episode stopped')
        dt = min(dt, 0.05)  # Bound each step even after a short scheduling delay.
        twist = np.asarray(twist, dtype=float).copy()
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            raise ValueError('Invalid Cartesian twist')
        twist[3:5] = 0.0
        measured = np.asarray(measured, dtype=float)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            raise RuntimeError('Invalid measured joint state')
        if np.max(np.abs(self.q - measured)) > 0.15:
            raise RuntimeError('Robot tracking error exceeds 0.15 rad; episode stopped')
        if not self.armed:
            self.armed = (not enabled or self.config.button_mode == "deoxys") and not grip and not np.any(twist)
            self.status = "ready" if self.armed else "waiting for release and centered cap"
            self.last_grip = grip
            return self.q.copy(), None
        gripper_percent = None
        if self.config.button_mode == "deoxys" and grip != self.last_grip:
            self.closed = grip
            gripper_percent = 0.0 if grip else 100.0
        elif self.config.button_mode == "clutch" and enabled and grip and not self.last_grip:
            self.closed = not self.closed
            gripper_percent = 0.0 if self.closed else 100.0
        self.last_grip = grip
        self.status = 'holding (enable released)' if not enabled else 'centered / dead zone'
        if enabled and np.any(twist):
            self.pin.framesForwardKinematics(self.model, self.data, self.full_q(self.q))
            desired_orientation = self.pin.exp3(np.array([0., 0., twist[5] * dt])) @ self.orientation
            # Hold initial tilt; integrate only base-frame yaw. Correct numerical IK drift.
            twist[3:] = self.pin.log3(desired_orientation @ self.data.oMf[self.frame].rotation.T) / dt
            jac = self.pin.computeFrameJacobian(
                self.model, self.data, self.full_q(self.q), self.frame,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, :7]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.01**2 * np.eye(6), twist)
            dq *= min(1.0, self.config.joint_speed / max(np.max(np.abs(dq)), 1e-12))
            candidate = self.q + dq * dt
            if self.safe(candidate):
                self.q = candidate
                self.orientation = desired_orientation
                self.status = "moving XYZ+yaw"
            else:
                self.status = "target rejected: joint limit or height floor"
        return self.q.copy(), gripper_percent


def add_controller_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--button-mode', choices=['clutch', 'deoxys'], default='clutch',
                        help='clutch: hold enable to move; deoxys: move freely, enable button stops, hold gripper button to close')
    parser.add_argument('--controller', choices=['so101', 'spacemouse'], default='so101')
    parser.add_argument('--spacemouse-device', help='PySpaceMouse device name (default: auto-detect)')
    parser.add_argument('--linear-speed', type=float, default=0.10, help='SpaceMouse translation limit, m/s')
    parser.add_argument('--angular-speed', type=float, default=0.3, help='SpaceMouse rotation limit, rad/s')
    parser.add_argument('--joint-speed', type=float, default=0.5, help='SpaceMouse joint speed limit, rad/s')
    parser.add_argument('--deadzone', type=float, default=0.1)
    parser.add_argument('--input-timeout', type=float, default=0.25)
    parser.add_argument('--enable-button', type=int, default=0, help='Hold to enable motion (zero-based)')
    parser.add_argument('--gripper-button', type=int, default=1, help='Toggle gripper while enabled (zero-based)')
    parser.add_argument('--axis-map', default='1,2,3,4,5,6', help='Signed axis permutation for base-frame x,y,z,rx,ry,rz')


def connect_controller(args, *, settle_s=0.0):
    if args.controller == 'so101':
        from droid_plus.datagen.setup import connect_so101
        return connect_so101(args.port, settle_s=settle_s)
    config = SpaceMouseConfig(
        button_mode=args.button_mode, linear_speed=args.linear_speed, angular_speed=args.angular_speed,
        joint_speed=args.joint_speed, deadzone=args.deadzone, timeout=args.input_timeout,
        enable_button=args.enable_button, gripper_button=args.gripper_button,
        axis_map=tuple(int(x) for x in args.axis_map.split(',')),
    )
    return SpaceMouse(config, args.spacemouse_device)
