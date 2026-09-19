# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Teleop episode runner: streams SO-101 leader actions to the Franka while
recording observations at a configurable sub-rate.

UI-agnostic — mirrors ``droid_plus.eval.episode_runner.run_episode``. The
caller supplies a ``should_stop`` callable (KeyPoller closure for CLI,
``threading.Event.is_set`` for the web UI) and constructs the
``EpisodeRecorder``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from droid_plus.datagen.safety import DEFAULT_MIN_EE_Z, enforce_min_z
from droid_plus.datagen.so101 import (
    action_to_so101_joints_deg,
    extract_so101_gripper_deg,
    so101_gripper_to_robotiq,
    so101_to_franka,
)
from droid_plus.eval.episode_runner import EpisodeConfig, EpisodeResult
from droid_plus.logging import EpisodeRecorder
from droid_plus.robot.observations import pack_state_action
from droid_plus.services.franky_client import HOME_POSITION

if TYPE_CHECKING:
    from droid_plus.robot import DroidPlus


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TeleopSessionConfig:
    """Session-level parameters for a teleop run (constant across episodes).

    Kept separate from ``eval.episode_runner.SessionConfig`` because teleop has
    no policy host/port, carries a control-vs-record rate split, and owns a
    safety knob (``min_z``).
    """
    rate_hz: float = 100.0
    record_rate_hz: float = 15.0
    jpeg_quality: int = 90
    record_jpeg_quality: int = 90
    min_z: float = DEFAULT_MIN_EE_Z
    dry_run: bool = False
    record: bool = True
    policy_name: str = "teleop_so101"


# ── Episode runner ───────────────────────────────────────────────────────────

def run_teleop_episode(
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
) -> EpisodeResult:
    """Run a single teleop episode.

    Streams SO-101 leader actions to the Franka at ``session.rate_hz``, with
    FK-based table-safety clamping. Records images + state/action at the
    configured sub-rate. Returns an :class:`EpisodeResult` with the recorder
    **un-finalized** (the caller finalizes after collecting labels).
    """
    dt = 1.0 / float(session.rate_hz)
    record_every_n = max(1, round(float(session.rate_hz) / float(session.record_rate_hz)))

    seq = 0
    rec_seq = 0
    last_gripper_pos: int | None = None
    last_gripper_cmd_frac: float = 0.0
    q_prev_safe = np.array(HOME_POSITION, dtype=float)

    step_limit = config.action_step_limit if config.action_step_limit > 0 else None

    _stopped_by_caller = False
    _stopped_by_limit = False
    t_episode_start = time.time()

    try:
        while True:
            if should_stop():
                _stopped_by_caller = True
                break
            if step_limit is not None and seq >= step_limit:
                _stopped_by_limit = True
                break

            t0 = time.time()

            action = teleop.get_action()
            so_deg = action_to_so101_joints_deg(action)
            so_rad = np.deg2rad(so_deg)
            q_franka = so101_to_franka(so_rad)

            q_franka, _ = enforce_min_z(
                q_franka, q_prev_safe, pin_model, pin_data, ee_frame, session.min_z,
            )
            q_prev_safe = q_franka.copy()

            if not session.dry_run:
                droid.set_target_joint_state(q_franka, velocities=[0.0] * 7, seq=seq)

            # Gripper — continuous mapping. Gate like GELLO: only send when the
            # previous command finished (not busy) and the target changed.
            gripper_cmd_frac = last_gripper_cmd_frac
            if gripper_initialized and not session.dry_run:
                so101_gripper_deg = extract_so101_gripper_deg(action)
                if so101_gripper_deg is not None:
                    robotiq_pos = so101_gripper_to_robotiq(so101_gripper_deg)
                    gripper_cmd_frac = float(robotiq_pos) / 255.0
                    if last_gripper_pos is None or abs(robotiq_pos - last_gripper_pos) > 5:
                        busy = False
                        try:
                            st_g = droid.gripper.gripper_state()
                            busy = bool(st_g.get("busy", False))
                        except Exception:
                            busy = False
                        if not busy:
                            try:
                                droid.gripper.go_to_async(robotiq_pos, wait=False)
                                last_gripper_pos = robotiq_pos
                            except Exception:
                                pass
                    last_gripper_cmd_frac = gripper_cmd_frac

            # Sub-rate recording.
            if recorder is not None and session.record and seq % record_every_n == 0:
                try:
                    left_rgb = droid.get_left_image(jpeg_quality=session.record_jpeg_quality)
                    wrist_rgb = droid.get_wrist_image(jpeg_quality=session.record_jpeg_quality)
                    right_rgb = droid.get_right_image(jpeg_quality=session.record_jpeg_quality)

                    js = droid.get_current_joint_state()
                    q_state = np.asarray(js["positions"], dtype=np.float64)
                    dq_state = np.asarray(js.get("velocities", [0.0] * 7), dtype=np.float64).reshape(-1)
                    if dq_state.shape != (7,):
                        dq_state = np.zeros((7,), dtype=np.float64)

                    gripper_obs = 0.0
                    if gripper_initialized:
                        try:
                            gripper_obs = float(droid.get_gripper_obs_value())
                        except Exception:
                            pass

                    state_pos, state_vel = pack_state_action(q_state, dq_state, gripper_obs)
                    action_pos, action_vel = pack_state_action(
                        q_franka, np.zeros((7,), dtype=np.float64), gripper_cmd_frac,
                    )

                    recorder.record_step(
                        seq=rec_seq,
                        left_rgb=left_rgb,
                        wrist_rgb=wrist_rgb,
                        right_rgb=right_rgb,
                        state_positions=state_pos,
                        state_velocities=state_vel,
                        action_positions=action_pos,
                        action_velocities=action_vel,
                        t_wall_s=t0,
                    )
                    rec_seq += 1
                except Exception as e:
                    print(f"[record] Warning: {e}")

            seq += 1

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        if not session.dry_run:
            try:
                droid.stop()
            except Exception:
                pass

    return EpisodeResult(
        episode_idx=recorder.episode_idx if recorder is not None else 0,
        seq=rec_seq,
        t_start=t_episode_start,
        t_end=time.time(),
        inference_times=[],
        recorder=recorder,
        stopped_by_limit=_stopped_by_limit,
        stopped_by_caller=_stopped_by_caller,
    )


# ── Recording finalization ───────────────────────────────────────────────────

def finalize_teleop_episode_recording(
    *,
    result: EpisodeResult,
    config: EpisodeConfig,
    session: TeleopSessionConfig,
    episode_valid: bool | None = None,
    episode_success: bool | None = None,
    episode_score: float | None = None,
    episode_notes: str | None = None,
) -> None:
    """Finalize a teleop episode recording: close writers, write meta.json.

    Mirrors ``eval.episode_runner.finalize_episode_recording`` but emits
    teleop-shaped ``policy`` and ``control`` meta blocks.
    """
    if result.recorder is None:
        return
    try:
        result.recorder.finalize(
            experiment=config.experiment,
            task=config.task,
            instruction=config.instruction,
            notes=config.notes,
            policy_info={"name": session.policy_name},
            control_info={
                "rate_hz": session.rate_hz,
                "record_rate_hz": session.record_rate_hz,
                "min_z": session.min_z,
                "action_step_limit": config.action_step_limit,
            },
            extra_meta={
                "valid": episode_valid,
                "success": episode_success,
                "score": episode_score,
                "episode_notes": episode_notes,
            },
        )
    except Exception:
        result.recorder.close()
