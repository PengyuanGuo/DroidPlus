"""SpaceMouse-only XYZ/yaw episode loop, recording, and timing diagnostics.

No SO-101 dispatch or joint mapping lives here. The workstation entry points
select this runner explicitly; the robot-facing API and dataset schema are shared.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from droid_plus.datagen.async_recording import AsyncTeleopRecorder
from droid_plus.datagen.spacemouse import CartesianController
from droid_plus.datagen.teleop_runner import TeleopSessionConfig
from droid_plus.eval.episode_runner import EpisodeConfig, EpisodeResult
from droid_plus.logging import EpisodeRecorder

if TYPE_CHECKING:
    from droid_plus.robot import DroidPlus


def run_spacemouse_episode(
    *,
    config: EpisodeConfig,
    session: TeleopSessionConfig,
    droid: "DroidPlus",
    teleop: Any,
    gripper_initialized: bool,
    pin_model: Any,
    pin_data: Any,
    ee_frame: str,
    recorder: EpisodeRecorder | None,
    should_stop: Callable[[], bool],
    get_phase: Callable[[], int] | None = None,
) -> EpisodeResult:
    """Run a single teleop episode.

    Streams Cartesian SpaceMouse targets with kinematic limit checks.
    SpaceMouse camera recording runs independently of the motion loop. Returns an :class:`EpisodeResult` with the recorder
    **un-finalized** (the caller finalizes after collecting labels).
    """
    if not all(np.isfinite(x) and x > 0 for x in (session.rate_hz, session.record_rate_hz)):
        raise ValueError("Control and recording rates must be finite and positive")
    dt = 1.0 / float(session.rate_hz)

    seq = 0
    rec_seq = 0
    last_gripper_cmd: float = 0.0
    gripper_command_sent = False

    step_limit = config.action_step_limit if config.action_step_limit > 0 else None

    _stopped_by_caller = False
    _stopped_by_limit = False
    t_episode_start = time.time()

    async_recorder = None
    previous_tick = None
    status_time = time.monotonic()
    status_seq = 0
    timings = {}
    previous_timings = {}
    try:
        cartesian = CartesianController(
            pin_model, pin_data, ee_frame,
            droid.get_current_joint_state()["positions"], teleop.config, session.min_z,
        )
        print(f"SpaceMouse XYZ+yaw ({teleop.config.button_mode}): center cap and release buttons to arm.")
        if recorder is not None and session.record:
            async_recorder = AsyncTeleopRecorder(
                droid, recorder, session.record_rate_hz, session.record_jpeg_quality, gripper_initialized,
            )
        while True:
            if should_stop():
                _stopped_by_caller = True
                break
            if step_limit is not None and seq >= step_limit:
                _stopped_by_limit = True
                break

            previous_timings = timings
            timings = {}
            tick_start = time.monotonic()
            t0 = time.time()
            # Snapshot before image acquisition so an input during IO applies next step.
            phase_index = get_phase() if get_phase is not None else 0

            started = time.monotonic()
            js_control = droid.get_current_joint_state()
            timings["state_get_ms"] = (time.monotonic() - started) * 1000
            measured = cartesian.q if session.dry_run else js_control["positions"]
            control_now = time.monotonic()
            control_dt = dt if previous_tick is None else control_now - previous_tick
            previous_tick = control_now
            timings["interval_ms"] = control_dt * 1000
            started = time.monotonic()
            twist, enabled, grip = teleop.read()
            timings["input_ms"] = (time.monotonic() - started) * 1000
            started = time.monotonic()
            q_franka, gripper_percent = cartesian.step(
                twist, enabled, grip, measured, control_dt,
            )
            timings["ik_ms"] = (time.monotonic() - started) * 1000
            started = time.monotonic()
            if not session.dry_run:
                droid.set_target_joint_state(q_franka, velocities=[0.0] * 7, seq=seq)

            timings["target_post_ms"] = (time.monotonic() - started) * 1000
            started = time.monotonic()

            # Gripper — same binary open/close as policy rollout.
            gripper_cmd_frac = last_gripper_cmd
            if gripper_initialized and not session.dry_run:
                if gripper_percent is not None:
                    # LeRobot normalizes the gripper to 0–100, even when arm
                    # joints use degrees: 0 = closed, 100 = open.
                    gripper_cmd_frac = 1.0 if float(gripper_percent) <= 50.0 else 0.0
                    if not gripper_command_sent or gripper_cmd_frac != last_gripper_cmd:
                        try:
                            if gripper_cmd_frac >= 0.5:
                                droid.gripper.close_async(wait=False)
                            else:
                                droid.gripper.open_async(wait=False)
                            last_gripper_cmd = gripper_cmd_frac
                            gripper_command_sent = True
                        except Exception:
                            gripper_cmd_frac = last_gripper_cmd

            timings["gripper_ms"] = (time.monotonic() - started) * 1000
            if async_recorder is not None:
                async_recorder.submit(js_control, q_franka, gripper_cmd_frac, phase_index, t0)
            now = time.monotonic()
            if now - status_time >= 1.0:
                actual_hz = (seq + 1 - status_seq) / (now - status_time)
                record_hz = async_recorder.actual_rate if async_recorder is not None else 0.0
                print(f"[spacemouse] {cartesian.status}; enable={enabled} "
                      f"xyz={np.round(twist[:3], 3)} yaw={twist[5]:.3f}; "
                      f"control={actual_hz:.1f}Hz record={record_hz:.1f}Hz; "
                      + " ".join(f"{key}={value:.1f}" for key, value in timings.items()), flush=True)
                status_time, status_seq = now, seq + 1

            seq += 1

            elapsed = time.monotonic() - tick_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except Exception:
        print("[spacemouse timing at failure] current="
              + str({key: round(value, 1) for key, value in timings.items()})
              + " previous=" + str({key: round(value, 1) for key, value in previous_timings.items()}),
              flush=True)
        raise
    finally:
        if not session.dry_run:
            try:
                droid.stop()
            except Exception:
                pass

        if async_recorder is not None:
            async_recorder.close()
            rec_seq = async_recorder.count

    result = EpisodeResult(
        episode_idx=recorder.episode_idx if recorder is not None else 0,
        seq=rec_seq,
        t_start=t_episode_start,
        t_end=time.time(),
        inference_times=[],
        recorder=recorder,
        stopped_by_limit=_stopped_by_limit,
        stopped_by_caller=_stopped_by_caller,
    )

    if async_recorder is not None:
        result.actual_record_rate_hz = async_recorder.actual_rate
    return result


