# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Episode runner: executes a single closed-loop policy episode.

This module is UI-agnostic — the caller controls how episodes are started/stopped
and how labels are collected. The ``should_stop`` callable is the integration seam:
pass a KeyPoller-backed closure for CLI, or ``threading.Event().is_set`` for a web UI.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm import tqdm

from droid_plus.logging import EpisodeRecorder
from droid_plus.robot.observations import make_policy_observation, pack_state_action
from droid_plus.utils import RateLimiter

if TYPE_CHECKING:
    from droid_plus.eval.base_client import InferenceClient
    from droid_plus.robot import DroidPlus


LOG = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeConfig:
    """Per-episode parameters that can change between episodes."""
    instruction: str = ""
    task: str = ""
    action_step_limit: int = -1  # -1 = no limit
    experiment: str | None = None
    notes: str = ""


@dataclass
class EpisodeResult:
    """Returned by :func:`run_episode`."""
    episode_idx: int
    seq: int                          # action steps executed
    t_start: float
    t_end: float
    inference_times: list[float]
    recorder: EpisodeRecorder | None  # un-finalized; caller finalizes after labeling
    stopped_by_limit: bool
    stopped_by_caller: bool           # should_stop() returned True


@dataclass(frozen=True)
class SessionConfig:
    """Session-level parameters (constant across episodes)."""
    policy_host: str = ""
    policy_port: int = 8000
    rate_hz: float = 15.0
    jpeg_quality: int = 90
    record_jpeg_quality: int = 90
    dry_run: bool = False
    record: bool = True
    policy_name: str = "pi0"


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(
    *,
    config: EpisodeConfig,
    session: SessionConfig,
    droid: "DroidPlus",
    policy: "InferenceClient",
    episode_idx: int,
    recorder: EpisodeRecorder | None,
    should_stop: Callable[[], bool],
) -> EpisodeResult:
    """Run a single closed-loop episode.

    Executes the policy loop: cameras+robot → policy inference → robot command.
    Returns an :class:`EpisodeResult` with the recorder **un-finalized** (the
    caller finalizes after collecting labels).

    Args:
        config: Per-episode parameters (instruction, task, step limit, etc.)
        session: Session-level parameters (policy host, rate_hz, etc.)
        droid: Robot client.
        policy: Policy inference client (any :class:`InferenceClient` subclass).
        episode_idx: Episode index for recording/display.
        recorder: Episode recorder (or ``None`` to skip recording).
        should_stop: Callable returning ``True`` to end the episode early.
    """
    print(f"Starting episode {episode_idx}...")

    # Reset policy chunk state at episode start.
    try:
        policy.reset()
    except Exception:
        pass

    seq = 0
    t_episode_start = time.time()
    inference_times: list[float] = []

    # Open-loop action chunk buffer.
    pending_chunk: np.ndarray | None = None
    pending_i: int = 0

    # Rate limiter for drift-corrected loop timing.
    rate = RateLimiter(rate_hz=session.rate_hz)

    # Create tqdm progress bar for action steps
    step_limit = config.action_step_limit if config.action_step_limit > 0 else None
    pbar = tqdm(
        total=step_limit,
        desc=f"Episode {episode_idx}",
        unit="step",
        dynamic_ncols=True,
        bar_format="{desc}: {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}, {rate_fmt}]" if step_limit else "{desc}: {n_fmt} steps [{elapsed}, {rate_fmt}]",
    )

    last_gripper_cmd: float | None = None
    last_gripper_obs: float = 0.0
    gripper_busy: bool = False
    _stopped_by_caller = False
    _stopped_by_limit = False

    try:
        while True:
            # Check stop condition (ESC, signal, threading.Event, etc.)
            if should_stop():
                _stopped_by_caller = True
                break

            # End the episode after N action steps (if enabled).
            if step_limit is not None and seq >= step_limit:
                _stopped_by_limit = True
                break

            t0 = time.time()

            # Refresh gripper feedback for observation and transition diagnostics (best-effort).
            try:
                st_g = droid.gripper.gripper_state()
                gripper_busy = bool(st_g.get("busy", False))
                if "position_frac" in st_g:
                    last_gripper_obs = float(st_g["position_frac"])
                else:
                    last_gripper_obs = float(droid.get_gripper_obs_value())
            except Exception:
                pass

            # Fetch a new action chunk only when we've exhausted the previous one.
            chunk_refresh = pending_chunk is None or pending_i >= int(pending_chunk.shape[0])

            # Per-step sensing for recording (and for policy chunk refresh).
            sense_this_step = bool(session.record) or bool(chunk_refresh)
            left: np.ndarray | None = None
            wrist: np.ndarray | None = None
            q: np.ndarray | None = None
            dq: np.ndarray | None = None
            if sense_this_step:
                left = droid.get_left_image(jpeg_quality=session.jpeg_quality)
                wrist = droid.get_wrist_image(jpeg_quality=session.jpeg_quality)

                js = droid.get_current_joint_state()
                q = np.asarray(js["positions"], dtype=np.float32)
                if q.shape != (7,):
                    raise RuntimeError(f"Expected 7 joint positions, got shape {q.shape}")
                dq = np.asarray(js.get("velocities", [0.0] * 7), dtype=np.float32).reshape(-1)
                if dq.shape != (7,):
                    dq = np.zeros((7,), dtype=np.float32)

            if chunk_refresh:
                if left is None or wrist is None or q is None:
                    raise RuntimeError("Internal error: missing observation when refreshing policy chunk")
                obs = make_policy_observation(
                    left_rgb=left,
                    wrist_rgb=wrist,
                    joint_pos=q,
                    gripper_pos=last_gripper_obs,
                )
                t_infer_start = time.time()
                pending_chunk = policy.infer_chunk(obs, config.instruction)
                t_infer_end = time.time()
                inference_times.append(t_infer_end - t_infer_start)
                pending_i = 0

            action = np.asarray(pending_chunk[pending_i], dtype=np.float32).reshape(-1)  # type: ignore[index]
            pending_i += 1
            if action.shape[0] < 7:
                raise RuntimeError(f"Expected action with at least 7 dims, got {action.shape}")

            q_cmd7 = action[:7].astype(np.float64, copy=False)
            dq_cmd7 = np.zeros((7,), dtype=np.float64)
            # Policy gripper output is a continuous target in [0,1].
            g_cmd_target = float(action[-1]) if action.shape[0] >= 8 else 0.0
            if not np.isfinite(g_cmd_target):
                g_cmd_target = 0.0
            g_cmd_target = float(np.clip(g_cmd_target, 0.0, 1.0))

            # For hardware: convert gripper target to a binary open/close command.
            g_cmd_binary = 1.0 if g_cmd_target >= 0.5 else 0.0

            # Record (images + state + action) for this tick.
            if recorder is not None and left is not None and wrist is not None and q is not None and dq is not None:
                state_pos, state_vel = pack_state_action(q, dq, last_gripper_obs)
                action_pos, action_vel = pack_state_action(q_cmd7, dq_cmd7, g_cmd_binary)
                recorder.record_step(
                    seq=seq,
                    left_rgb=left,
                    wrist_rgb=wrist,
                    state_positions=state_pos,
                    state_velocities=state_vel,
                    action_positions=action_pos,
                    action_velocities=action_vel,
                    t_wall_s=t0,
                )

            # Franky's nonblocking target interface can reverse a busy Hand.
            # Do not suppress OPEN while a previous CLOSE is still in progress.
            if (not session.dry_run) and g_cmd_binary != last_gripper_cmd:
                if g_cmd_binary >= 0.5:
                    droid.gripper.close_async(wait=False)
                else:
                    droid.gripper.open_async(wait=False)
                last_gripper_cmd = g_cmd_binary
                LOG.info("Gripper transition queued: %s (service busy=%s)",
                         "close" if g_cmd_binary else "open", gripper_busy)
            pbar.update(1)
            if not session.dry_run:
                droid.set_target_joint_state(q_cmd7, velocities=dq_cmd7.tolist(), seq=seq)
            seq += 1

            # Drift-corrected sleep for fixed stepping rate.
            rate.sleep()
    finally:
        # Close the progress bar
        pbar.close()

        # Stop robot at episode end (skip in dry-run to keep it command-free).
        if not session.dry_run:
            try:
                droid.stop()
            except Exception:
                pass

        # Reset policy chunk state at episode end (best-effort).
        try:
            policy.reset()
        except Exception:
            pass

    return EpisodeResult(
        episode_idx=episode_idx,
        seq=seq,
        t_start=t_episode_start,
        t_end=time.time(),
        inference_times=inference_times,
        recorder=recorder,
        stopped_by_limit=_stopped_by_limit,
        stopped_by_caller=_stopped_by_caller,
    )


# ── Recording finalization ───────────────────────────────────────────────────

def finalize_episode_recording(
    *,
    result: EpisodeResult,
    config: EpisodeConfig,
    session: SessionConfig,
    episode_valid: bool | None = None,
    episode_success: bool | None = None,
    episode_score: float | None = None,
    episode_notes: str | None = None,
) -> None:
    """Finalize recording: write meta.json via EpisodeRecorder.finalize().

    Call this after :func:`run_episode` returns and labels have been collected.
    """
    if result.recorder is None:
        return
    try:
        result.recorder.finalize(
            experiment=config.experiment,
            task=config.task,
            instruction=config.instruction,
            notes=config.notes,
            policy_info={
                "name": session.policy_name,
                "host": session.policy_host,
                "port": session.policy_port,
            },
            control_info={
                "rate_hz": session.rate_hz,
                "action_step_limit": config.action_step_limit,
            },
            inference_times=result.inference_times,
            extra_meta={
                "valid": episode_valid,
                "success": episode_success,
                "score": episode_score,
                "episode_notes": episode_notes,
            },
        )
    except Exception:
        result.recorder.close()
