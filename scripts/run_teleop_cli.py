#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CLI teleop runner — SO-101 leader arm streams to Franka, with recording.

Multi-episode loop: SPACE to start, ESC to stop, then post-episode prompts
(valid/success/score/notes). Records by default; pass --no-record to disable.

    Usage:
    python scripts/run_teleop_cli.py
    python scripts/run_teleop_cli.py --port /dev/ttyACM1 --gripper none
    python scripts/run_teleop_cli.py --gripper franka
    python scripts/run_teleop_cli.py --no-record
    python scripts/run_teleop_cli.py --task "pick_banana" --notes "first attempt"
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any, Callable

from droid_plus.analysis.end_effector_pose import compute_and_save_ee_trajectory_single
from droid_plus.constants import FRANKY_SERVICE_URL, GRIPPER_SERVICE_URL, RECORD_JPEG_QUALITY
from droid_plus.datagen import (
    DEFAULT_MIN_EE_Z,
    TeleopSessionConfig,
    build_fk_model,
    connect_so101,
    finalize_teleop_episode_recording,
    init_gripper,
    make_teleop_run_dir,
    run_teleop_episode,
)
from droid_plus.eval.episode_runner import EpisodeConfig
from droid_plus.eval.experiment_setup import wait_for_cameras
from droid_plus.logging import EpisodeRecorder
from droid_plus.robot import DroidPlus
from droid_plus.utils import (
    KeyPoller,
    prompt_score,
    prompt_success,
    prompt_text,
    prompt_valid,
)

_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _make_cli_should_stop(keys: KeyPoller, stop_flag: list[bool]) -> Callable[[], bool]:
    """ESC (during episode) or Ctrl+C (via stop_flag) ends the episode."""
    def should_stop() -> bool:
        if stop_flag[0]:
            return True
        if sys.stdin.isatty():
            ch = keys.poll_char()
            if ch == "\x1b":
                return True
        return False
    return should_stop


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop SO-101 leader arm to Franka with recording")
    parser.add_argument("--port", default="/dev/ttyACM0", help="SO-101 leader serial port")
    parser.add_argument("--franky-service-url", default=FRANKY_SERVICE_URL,
        help="Franky service URL (robot control + URDF fetch)")
    parser.add_argument(
        "--gripper",
        choices=["none", "robotiq", "franka"],
        default="robotiq",
        help="Gripper backend: robotiq (local gripper_service), franka (Hand via franky_service), none",
    )
    parser.add_argument("--no-gripper", action="store_true",
        help="Alias for --gripper none")
    parser.add_argument("--gripper-service-url", default=GRIPPER_SERVICE_URL,
        help="Robotiq gripper_service URL (ignored when --gripper franka)")
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Control loop rate (Hz)")
    parser.add_argument("--min-z", type=float, default=DEFAULT_MIN_EE_Z,
        help=f"Minimum EE Z height (m) — table safety threshold (default: {DEFAULT_MIN_EE_Z})")
    parser.add_argument("--no-record", action="store_true", help="Disable data recording")
    parser.add_argument("--record-rate-hz", type=float, default=15.0,
        help="Recording rate (images + state/action captured at this rate)")
    parser.add_argument("--record-jpeg-quality", type=int, default=RECORD_JPEG_QUALITY,
        help=f"JPEG quality for recorded images (default: {RECORD_JPEG_QUALITY})")
    parser.add_argument("--output-dir", default="output", help="Base output directory for recordings")
    parser.add_argument("--task", default="", help="Task name (recorded in meta.json + run dir)")
    parser.add_argument("--notes", default="", help="Free-form notes (recorded in meta.json)")
    parser.add_argument("--dry-run", action="store_true",
        help="Read SO-101 but do not command the robot or gripper")
    args = parser.parse_args()

    gripper_backend = "none" if args.no_gripper else str(args.gripper)
    if gripper_backend == "franka":
        gripper_url = str(args.franky_service_url)
    else:
        gripper_url = str(args.gripper_service_url)

    record = not args.no_record

    session = TeleopSessionConfig(
        rate_hz=float(args.rate_hz),
        record_rate_hz=float(args.record_rate_hz),
        record_jpeg_quality=int(args.record_jpeg_quality),
        min_z=float(args.min_z),
        dry_run=bool(args.dry_run),
        record=bool(record),
    )

    # ── Robot + cameras ──────────────────────────────────────────────────
    print("Initializing robot...")
    print(f"Franky service: {args.franky_service_url}")
    print(f"Gripper backend: {gripper_backend}"
          + (f" ({gripper_url})" if gripper_backend != "none" else ""))
    droid = DroidPlus(
        franky_service_url=args.franky_service_url,
        gripper_service_url=gripper_url,
    )
    wait_for_cameras(droid)

    try:
        pin_model, pin_data, ee_frame = build_fk_model(args.franky_service_url)
    except Exception as e:
        print(f"WARNING: FK unavailable ({e}) — Z safety DISABLED")
        pin_model = pin_data = ee_frame = None

    # Safety: stop before anything moves.
    try:
        droid.stop()
    except Exception:
        pass

    # ── Gripper, then SO-101 (ordering matters for shared USB hub / Robotiq) ──
    gripper_initialized = False
    if gripper_backend != "none" and not args.dry_run:
        gripper_initialized = init_gripper(droid, backend=gripper_backend)
    else:
        reason = "dry-run" if args.dry_run else "--gripper none"
        print(f"Skipping gripper initialization ({reason}).")

    # Robotiq shares a USB hub with SO-101; Franka Hand is Ethernet — no settle needed.
    settle_s = 2.0 if (gripper_initialized and gripper_backend == "robotiq") else 0.0
    teleop = connect_so101(args.port, settle_s=settle_s)

    # ── Run directory ────────────────────────────────────────────────────
    base_run_dir: str | None = None
    if session.record:
        base_run_dir = make_teleop_run_dir(args.task, parent=args.output_dir)
        print(f"Recording to {base_run_dir}")
        record_every_n = max(1, round(session.rate_hz / session.record_rate_hz))
        print(f"Control loop: {session.rate_hz} Hz | Recording every {record_every_n} ticks "
              f"(~{session.rate_hz / record_every_n:.1f} Hz)")

    # ── Stop flag for SIGINT/SIGTERM ─────────────────────────────────────
    stop_flag: list[bool] = [False]

    def _request_stop(*_sig: Any) -> None:
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if not sys.stdin.isatty():
        print("stdin is not a TTY; starting immediately and disabling space/escape controls.")

    # ── Episode loop ─────────────────────────────────────────────────────
    episode_idx = 0

    with KeyPoller() as keys:
        while not stop_flag[0]:
            # Wait for SPACE when interactive.
            if sys.stdin.isatty():
                print(f"\n{_CYAN}Press SPACE to start episode {episode_idx}, "
                      f"ESC to quit, Ctrl+C to quit.{_RESET}")
                while not stop_flag[0]:
                    ch = keys.poll_char()
                    if ch is None:
                        time.sleep(0.05)
                        continue
                    if ch == "\x1b":
                        stop_flag[0] = True
                        break
                    if ch == " ":
                        break
                if stop_flag[0]:
                    break

            config = EpisodeConfig(
                task=args.task,
                notes=args.notes,
            )

            recorder: EpisodeRecorder | None = None
            if session.record and base_run_dir is not None:
                recorder = EpisodeRecorder(
                    base_run_dir=base_run_dir,
                    episode_idx=episode_idx,
                    jpeg_quality=session.record_jpeg_quality,
                    cameras=["left", "wrist", "right"],
                )
                print(f"Recording episode {episode_idx} to {recorder.episode_dir}  "
                      f"(rate={session.record_rate_hz} Hz)")

            should_stop = _make_cli_should_stop(keys, stop_flag)
            result = run_teleop_episode(
                config=config,
                session=session,
                droid=droid,
                teleop=teleop,
                gripper_initialized=gripper_initialized,
                pin_model=pin_model,
                pin_data=pin_data,
                ee_frame=ee_frame,
                recorder=recorder,
                should_stop=should_stop,
            )

            duration = result.t_end - result.t_start
            print(f"\n{_YELLOW}Episode {episode_idx} ended: {result.seq} recorded steps, "
                  f"{duration:.1f}s{_RESET}")

            # Post-episode labels.
            episode_valid: bool | None = None
            episode_success: bool | None = None
            episode_score: float | None = None
            episode_notes: str | None = None
            if sys.stdin.isatty():
                episode_valid = prompt_valid(keys)
                episode_success = prompt_success(keys)
                episode_score = prompt_score(keys)
                episode_notes = prompt_text(keys, "episode_notes?")

            if session.record and result.recorder is not None:
                finalize_teleop_episode_recording(
                    result=result,
                    config=config,
                    session=session,
                    episode_valid=episode_valid,
                    episode_success=episode_success,
                    episode_score=episode_score,
                    episode_notes=episode_notes,
                )
                print(f"Recording saved to {result.recorder.episode_dir}")

                try:
                    compute_and_save_ee_trajectory_single(
                        episode_dir=result.recorder.episode_dir,
                        overwrite=True,
                        verbose=True,
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute EE trajectory: {e}")

            episode_idx += 1

    # ── Cleanup ──────────────────────────────────────────────────────────
    if gripper_initialized:
        try:
            droid.gripper.shutdown_async()
        except Exception:
            pass


if __name__ == "__main__":
    main()
