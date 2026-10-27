"""Independent camera acquisition with a single latest control snapshot."""
from __future__ import annotations

import threading
import time

import numpy as np

from droid_plus.robot.observations import pack_state_action


class AsyncTeleopRecorder:
    def __init__(self, droid, recorder, rate_hz, jpeg_quality, gripper_initialized):
        self.droid, self.recorder = droid, recorder
        self.period = 1.0 / rate_hz
        self.jpeg_quality = jpeg_quality
        self.gripper_initialized = gripper_initialized
        self.condition = threading.Condition()
        self.latest = None
        self.stopping = False
        self.error = None
        self.count = 0
        self.timestamps = []
        self.thread = threading.Thread(target=self._run, name='teleop-recording', daemon=True)
        self.thread.start()

    def submit(self, joint_state, target, gripper_cmd, phase, stamp):
        # Own every array: control must never mutate a queued recording snapshot.
        snapshot = (
            np.asarray(joint_state['positions'], dtype=float).copy(),
            np.asarray(joint_state.get('velocities', [0.] * 7), dtype=float).copy(),
            np.asarray(target, dtype=float).copy(), gripper_cmd, phase, stamp,
        )
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Recording worker failed') from self.error
            self.latest = snapshot  # Replace; never accumulate a stale frame queue.
            self.condition.notify()

    def _run(self):
        deadline = 0.0
        previous_stamp = None
        try:
            while True:
                with self.condition:
                    while True:
                        if self.stopping:
                            return
                        delay = deadline - time.monotonic()
                        if self.latest is not None and self.latest[-1] != previous_stamp and delay <= 0:
                            snapshot = self.latest
                            break
                        self.condition.wait(timeout=max(0.001, delay) if delay > 0 else None)
                q, dq, target, grip, phase, stamp = snapshot
                previous_stamp = stamp
                started = time.monotonic()
                images = {}
                for role in ('left', 'wrist', 'right'):
                    images[role + '_rgb'] = (
                        getattr(self.droid, f'get_{role}_image')(jpeg_quality=self.jpeg_quality)
                        if getattr(self.droid, f'{role}_camera_id') else None
                    )
                gripper_obs = float(self.droid.get_gripper_obs_value()) if self.gripper_initialized else 0.0
                if dq.shape != (7,):
                    dq = np.zeros(7)
                state_pos, state_vel = pack_state_action(q, dq, gripper_obs)
                action_pos, action_vel = pack_state_action(target, np.zeros(7), grip)
                self.recorder.record_step(
                    seq=self.count, **images, state_positions=state_pos, state_velocities=state_vel,
                    action_positions=action_pos, action_velocities=action_vel,
                    t_wall_s=stamp, phase_index=phase,
                )
                self.count += 1
                self.timestamps.append(stamp)
                deadline = max(started + self.period, time.monotonic())
        except Exception as exc:
            with self.condition:
                self.error = exc

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        # Join before callers finalize/close the recorder. HTTP clients have timeouts.
        self.thread.join()
        if self.error is not None:
            raise RuntimeError('Recording worker failed') from self.error

    @property
    def actual_rate(self):
        if len(self.timestamps) < 2:
            return 0.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
