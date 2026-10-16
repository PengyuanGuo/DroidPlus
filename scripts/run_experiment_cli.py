#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CLI experiment runner — closed-loop policy episodes with recording.

Usage:
    python scripts/run_experiment_cli.py --exp banana_in_bowl
    python scripts/run_experiment_cli.py --instruction "pick up the can"
    python scripts/run_experiment_cli.py --exp-file experiments/experiments.json
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import replace
from typing import Any, Callable

from droid_plus.constants import DEFAULT_POLICY, POLICIES, RECORD_JPEG_QUALITY
from droid_plus.eval.episode_runner import (
    EpisodeConfig,
    SessionConfig,
    finalize_episode_recording,
    run_episode,
)
from droid_plus.eval.experiment_setup import (
    build_policy_client,
    init_droid_and_home,
    make_base_run_dir,
    maybe_compute_ee_trajectories,
    resolve_policy,
    wait_for_cameras,
)
from droid_plus.logging import EpisodeRecorder
from droid_plus.robot import DroidPlus
from droid_plus.utils import (
    KeyPoller,
    load_experiment,
    load_experiments_file,
    prompt_score,
    prompt_success,
    prompt_text,
    prompt_valid,
)

_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

# ── CLI helpers ──────────────────────────────────────────────────────────────

def _experiments_to_configs(experiments: list[dict[str, Any]]) -> list[EpisodeConfig]:
    """Convert experiment dicts (from experiments.json) to EpisodeConfig list."""
    return [EpisodeConfig(
        instruction=exp.get("instruction", ""),
        task=exp.get("task", ""),
        action_step_limit=exp.get("action_step_limit", -1),
        experiment=exp.get("experiment"),
        notes=exp.get("notes", ""),
    ) for exp in experiments]


def _make_cli_should_stop(keys: KeyPoller, stop_flag: list[bool]) -> Callable[[], bool]:
    """Build a should_stop callable for CLI mode.

    Returns True when ESC is pressed mid-episode or Ctrl+C sets the stop_flag.
    """
    def should_stop() -> bool:
        if stop_flag[0]:
            return True
        if sys.stdin.isatty():
            ch = keys.poll_char()
            if ch == "\x1b":
                return True
        return False
    return should_stop


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run a closed-loop: cameras+robot -> policy -> robot joint targets."""
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", "--exp", default=None,
        help="Load experiment parameters by name from experiments/experiments.json")
    p.add_argument("--experiment-file", "--exp-file", default=None,
        help="JSON file with list of experiments to run sequentially (same format as experiments.json)")
    p.add_argument("--policy", default=DEFAULT_POLICY,
        help=f"Policy name (available: {', '.join(POLICIES.keys())})")
    p.add_argument("--policy-host", default=None,
        help="Override policy host (default: from constants.py)")
    p.add_argument("--policy-port", type=int, default=None,
        help="Override policy port (default: from constants.py)")
    p.add_argument("--policy-path", default="experiments/dp_ckpts/dp_050000",
        help="Local LeRobot checkpoint directory. Used when --policy diffusion.")
    p.add_argument("--open-loop-horizon", type=int, default=10, help="Policy chunk horizon")
    p.add_argument("--rate-hz", type=float, default=15.0,
        help="Control loop rate (Hz). This is the action stepping rate.")
    p.add_argument("--instruction", default=None,
        help="Text instruction for policy (overrides experiment)")
    p.add_argument("--notes", default="",
        help="Free-form experiment note (recorded into meta.json when --record is set)")
    p.add_argument("--jpeg-quality", type=int, default=90,
        help="JPEG quality for camera snapshots")
    p.add_argument("--dry-run", action="store_true",
        help="Run policy but do not command the robot")
    p.add_argument("--record", default="True", action="store_true",
        help="Record images + state/action streams to runs/recordings/")
    p.add_argument("--task", default=None, help="Task name (overrides experiment)")
    p.add_argument("--record-jpeg-quality", type=int, default=RECORD_JPEG_QUALITY,
        help=f"JPEG quality for recorded images (default: {RECORD_JPEG_QUALITY}, from constants.py)")
    p.add_argument("--action-step-limit", type=int, default=None,
        help="Action step limit for each episode, maximum number of actions to take "
             "before stopping. -1 means no limit. (overrides experiment)")
    p.add_argument("--output-dir", default="output",
        help="Base output directory for recordings")
    args = p.parse_args()

    # ── Build episode config(s) ──────────────────────────────────────────
    configs: list[EpisodeConfig] | None = None  # None = infinite repeat with single_config
    single_config: EpisodeConfig | None = None

    if args.experiment_file:
        experiments = load_experiments_file(args.experiment_file)
        configs = _experiments_to_configs(experiments)
        if not configs:
            raise SystemExit(f"No experiments found in {args.experiment_file}")
        if args.instruction is not None:
            configs = [replace(c, instruction=args.instruction) for c in configs]
        if args.task is not None:
            configs = [replace(c, task=args.task) for c in configs]
        if args.action_step_limit is not None:
            configs = [replace(c, action_step_limit=args.action_step_limit) for c in configs]
        if args.notes:
            configs = [replace(c, notes=args.notes) for c in configs]
        print(f"{_CYAN}Loaded {len(configs)} experiment(s) from {args.experiment_file}{_RESET}")
    elif args.experiment:
        exp_params = load_experiment(args.experiment)
        single_config = EpisodeConfig(
            instruction=args.instruction if args.instruction is not None else exp_params.get("instruction", ""),
            task=args.task if args.task is not None else exp_params.get("task", ""),
            action_step_limit=args.action_step_limit if args.action_step_limit is not None else exp_params.get("action_step_limit", -1),
            experiment=args.experiment,
            notes=args.notes or "",
        )
    else:
        single_config = EpisodeConfig(
            instruction=args.instruction or "",
            task=args.task or "",
            action_step_limit=args.action_step_limit if args.action_step_limit is not None else -1,
            notes=args.notes or "",
        )

    # ── Resolve policy host/port ─────────────────────────────────────────
    # Diffusion is a local checkpoint, not a websocket server.
    diffusion_path = str(args.policy_path) if args.policy == "diffusion" else ""
    if args.policy == "diffusion":
        host, port, from_defaults = "local", 0, False
    else:
        host, port, from_defaults = resolve_policy(args.policy, args.policy_host, args.policy_port)

    record_jpeg_quality = int(args.record_jpeg_quality)

    session = SessionConfig(
        policy_host=host,
        policy_port=port,
        rate_hz=float(args.rate_hz),
        jpeg_quality=int(args.jpeg_quality),
        record_jpeg_quality=record_jpeg_quality,
        dry_run=bool(args.dry_run),
        record=bool(args.record),
        policy_name=str(args.policy),
    )

    # ── Print session info ───────────────────────────────────────────────
    source = " (from constants.py, use --policy-host/--policy-port to override)" if from_defaults else ""
    print(f"{_CYAN}{'─' * 50}")
    if configs is not None:
        print(f"  Mode:          batch ({len(configs)} experiments)")
        print(f"  Source:        {args.experiment_file}")
    elif single_config is not None:
        print("  Mode:          single experiment (repeat)")
        print(f"    Task:        {single_config.task or '(none)'}")
        print(f"    Instruction: {single_config.instruction or '(none)'}")
        print(f"    Notes:       {single_config.notes or '(none)'}")
        step_limit_str = str(single_config.action_step_limit) if single_config.action_step_limit > 0 else "unlimited"
        print(f"    Step limit:  {step_limit_str}")
    print("  Policy")
    print(f"    Name:        {session.policy_name}")
    if diffusion_path:
        print(f"    Checkpoint:  {diffusion_path}")
    else:
        print(f"    Server:      {session.policy_host}:{session.policy_port}{source}")
    print(f"{'─' * 50}{_RESET}")

    # ── Init resources ───────────────────────────────────────────────────
    droid = DroidPlus()
    wait_for_cameras(droid)

    if diffusion_path:
        from droid_plus.policies.diffusion_droid import DiffusionDroidClient
        policy = DiffusionDroidClient(diffusion_path)
    else:
        policy = build_policy_client(session.policy_host, session.policy_port, args.open_loop_horizon)

    base_run_dir: str | None = None
    if session.record:
        if configs is not None:
            task_str = "batch"
        elif single_config is not None:
            task_str = single_config.task if single_config.task else ""
        else:
            task_str = ""
        base_run_dir = make_base_run_dir(task_str, session.policy_name, parent=args.output_dir)

    init_droid_and_home(droid, dry_run=session.dry_run)

    stop_flag: list[bool] = [False]

    def _request_stop(*_sig: Any) -> None:
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if not sys.stdin.isatty():
        print("stdin is not a TTY; starting immediately and disabling space/escape controls.")

    # ── Episode loop ─────────────────────────────────────────────────────
    episode_idx = 0
    config_iter = iter(configs) if configs is not None else None

    with KeyPoller() as keys:
        while not stop_flag[0]:
            if config_iter is not None:
                config = next(config_iter, None)
                if config is None:
                    print(f"\n{_CYAN}All experiments complete.{_RESET}")
                    break
            else:
                assert single_config is not None
                config = single_config

            step_limit_str = str(config.action_step_limit) if config.action_step_limit > 0 else "unlimited"
            if configs is not None:
                remaining = len(configs) - episode_idx
                print(f"\n{_CYAN}{'─' * 50}")
                print(f"  Episode {episode_idx} ({remaining} remaining)")
                print(f"    Task:        {config.task or '(none)'}")
                print(f"    Instruction: {config.instruction or '(none)'}")
                print(f"    Step limit:  {step_limit_str}")
                print(f"{'─' * 50}{_RESET}")

            if sys.stdin.isatty():
                print(f"{_CYAN}Press SPACE to start episode {episode_idx}, ESC to quit, Ctrl+C to quit.{_RESET}")
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

            recorder: EpisodeRecorder | None = None
            if session.record and base_run_dir is not None:
                recorder = EpisodeRecorder(
                    base_run_dir=base_run_dir,
                    episode_idx=episode_idx,
                    jpeg_quality=session.record_jpeg_quality,
                )

            should_stop_fn = _make_cli_should_stop(keys, stop_flag)
            result = run_episode(
                config=config,
                session=session,
                droid=droid,
                policy=policy,
                episode_idx=episode_idx,
                recorder=recorder,
                should_stop=should_stop_fn,
            )

            if not session.dry_run:
                droid.move_to_home()
                try:
                    droid.gripper.open_async(wait=False)
                except Exception:
                    pass

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
                finalize_episode_recording(
                    result=result,
                    config=config,
                    session=session,
                    episode_valid=episode_valid,
                    episode_success=episode_success,
                    episode_score=episode_score,
                    episode_notes=episode_notes,
                )

            episode_duration = result.t_end - result.t_start
            step_limit_str = str(config.action_step_limit) if config.action_step_limit > 0 else "\u221e"
            print(f"{_YELLOW}Episode {episode_idx} ended: {result.seq}/{step_limit_str} steps, {episode_duration:.1f}s{_RESET}")
            episode_idx += 1
            if not session.dry_run:
                droid.stop()

    # ── Post-loop cleanup ────────────────────────────────────────────────

    if not session.dry_run:
        try:
            droid.gripper.shutdown_async()
        except Exception:
            pass

    if session.record and episode_idx > 0:
        maybe_compute_ee_trajectories(base_run_dir)


if __name__ == "__main__":
    main()
