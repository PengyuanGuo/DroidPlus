# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Episode recording utilities for experiment data collection.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from droid_plus.analysis.timing_stats import compute_timing_stats
from droid_plus.logging.async_data_writer import AsyncJsonlWriter
from droid_plus.logging.async_image_writer import AsyncImageWriter


class EpisodeRecorder:
    """
    Manages per-episode recording: images, steps.jsonl, and meta.json.

    Handles the lifecycle of recording a single episode:
    1. Creates episode directory structure
    2. Records per-step images and state/action data
    3. Writes meta.json on finalization

    Example:
        recorder = EpisodeRecorder(
            base_run_dir="output/experiment_20260118",
            episode_idx=0,
            jpeg_quality=90,
        )

        for step in steps:
            recorder.record_step(
                seq=step.seq,
                left_rgb=step.left_image,
                wrist_rgb=step.wrist_image,
                state_positions=step.state_pos,
                state_velocities=step.state_vel,
                action_positions=step.action_pos,
                action_velocities=step.action_vel,
                notes=step.notes,
            )

        recorder.finalize(
            experiment=args.experiment,
            task=args.task,
            instruction=args.instruction,
            policy_info={...},
            control_info={...},
        )
    """

    def __init__(
        self,
        base_run_dir: str,
        episode_idx: int,
        *,
        jpeg_quality: int = 90,
        max_image_queue: int = 8192,
        cameras: list[str] | None = None,
    ) -> None:
        """
        Initialize episode recorder.

        Args:
            base_run_dir: Base directory for the run (e.g., "output/experiment_20260118")
            episode_idx: Episode index (0, 1, 2, ...)
            jpeg_quality: JPEG quality for recorded images (1-100)
            max_image_queue: Max queue size for async image writer
            cameras: List of camera names to record (default: ["left", "wrist"])
        """
        self.base_run_dir = base_run_dir
        self.episode_idx = episode_idx
        self.jpeg_quality = jpeg_quality
        self.cameras = cameras or ["left", "wrist"]

        # Create episode directory
        self.episode_dir = os.path.join(base_run_dir, f"episode_{episode_idx:03d}")
        for cam in self.cameras:
            os.makedirs(os.path.join(self.episode_dir, cam), exist_ok=True)

        # Initialize writers
        self._image_writer = AsyncImageWriter(self.episode_dir, max_queue=max_image_queue)
        self._data_writer = AsyncJsonlWriter(os.path.join(self.episode_dir, "steps.jsonl"))

        # Timing
        self._t_start = time.time()
        self._step_count = 0
        self._closed = False
        self.phase_labels: list[int] = []

    def record_step(
        self,
        seq: int,
        *,
        left_rgb: np.ndarray | None = None,
        wrist_rgb: np.ndarray | None = None,
        right_rgb: np.ndarray | None = None,
        state_positions: list[float] | np.ndarray,
        state_velocities: list[float] | np.ndarray,
        action_positions: list[float] | np.ndarray,
        action_velocities: list[float] | np.ndarray,
        t_wall_s: float | None = None,
        extra: dict[str, Any] | None = None,
        phase_index: int | None = None,
    ) -> dict[str, str]:
        """
        Record a single step (images + state/action data).

        Args:
            seq: Step sequence number
            left_rgb: Left camera RGB image (H, W, 3) uint8
            wrist_rgb: Wrist camera RGB image (H, W, 3) uint8
            right_rgb: Right camera RGB image (H, W, 3) uint8
            state_positions: Current state positions (8,) - 7 joints + gripper
            state_velocities: Current state velocities (8,)
            action_positions: Commanded action positions (8,)
            action_velocities: Commanded action velocities (8,)
            t_wall_s: Wall clock time (defaults to now)
            phase_index: Optional zero-based operator phase label (0–127).
            extra: Additional fields to include in step data

        Returns:
            Dictionary with image paths: {"left": "left/000000000.jpg", ...}
        """
        if self._closed:
            raise RuntimeError("EpisodeRecorder is closed")

        if phase_index is not None and (type(phase_index) is not int or not 0 <= phase_index <= 127):
            raise ValueError("phase_index must be an integer in [0, 127]")
        t_wall_s = t_wall_s or time.time()
        image_paths: dict[str, str] = {}

        # Write images
        if left_rgb is not None:
            image_paths["left"] = self._image_writer.write("left", seq, left_rgb, jpeg_quality=self.jpeg_quality)
        if wrist_rgb is not None:
            image_paths["wrist"] = self._image_writer.write("wrist", seq, wrist_rgb, jpeg_quality=self.jpeg_quality)
        if right_rgb is not None:
            image_paths["right"] = self._image_writer.write("right", seq, right_rgb, jpeg_quality=self.jpeg_quality)

        # Convert to lists
        state_pos = list(np.asarray(state_positions, dtype=float).tolist())
        state_vel = list(np.asarray(state_velocities, dtype=float).tolist())
        action_pos = list(np.asarray(action_positions, dtype=float).tolist())
        action_vel = list(np.asarray(action_velocities, dtype=float).tolist())

        # Build step data
        step_data: dict[str, Any] = {
            "t_wall_s": float(t_wall_s),
            "seq": int(seq),
            "images": image_paths,
            "state": {
                "positions": state_pos,
                "velocities": state_vel,
            },
            "action": {
                "positions": action_pos,
                "velocities": action_vel,
            },
        }

        if extra:
            step_data.update(extra)

        if phase_index is not None:
            step_data["phase_index"] = phase_index
        self._data_writer.append(step_data)
        if phase_index is not None:
            self.phase_labels.append(phase_index)
        self._step_count += 1

        return image_paths

    def finalize(
        self,
        *,
        experiment: str | None = None,
        task: str = "",
        instruction: str = "",
        notes: str = "",
        policy_info: dict[str, Any] | None = None,
        control_info: dict[str, Any] | None = None,
        inference_times: list[float] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """
        Finalize the episode recording: close writers and write meta.json.

        Args:
            experiment: Experiment name
            task: Task name
            instruction: Text instruction
            notes: Free-form notes
            policy_info: Policy configuration dict
            control_info: Control configuration dict
            inference_times: List of inference times for stats
            extra_meta: Additional fields to include in meta.json

        Returns:
            Path to meta.json
        """
        if self._closed:
            raise RuntimeError("EpisodeRecorder is already closed")

        self._closed = True
        t_end = time.time()

        # Close writers
        try:
            self._data_writer.close()
        except Exception:
            pass
        try:
            self._image_writer.close()
        except Exception:
            pass

        # Compute inference time stats
        inference_stats = compute_timing_stats(inference_times or [])

        # Build meta
        meta: dict[str, Any] = {
            "t_start_wall_s": float(self._t_start),
            "t_end_wall_s": float(t_end),
            "duration_s": float(t_end - self._t_start),
            "episode_idx": int(self.episode_idx),
            "step_count": int(self._step_count),
            "experiment": experiment,
            "notes": str(notes),
            "task": str(task),
            "instruction": str(instruction),
            "recording": {
                "format": "jsonl+jpg",
                "steps_file": "steps.jsonl",
                "cameras": self.cameras,
                "jpeg_quality": int(self.jpeg_quality),
            },
            "policy": policy_info or {},
            "inference_time": inference_stats,
            "control": control_info or {},
            "schema": {
                "steps_jsonl": {
                    "t_wall_s": "float",
                    "seq": "int",
                    "images": {"left": "relpath", "wrist": "relpath", "right": "relpath"},
                    "state": {
                        "positions": "float[8] (last is gripper_position_frac in [0,1])",
                        "velocities": "float[8] (last is 0.0)",
                    },
                    "action": {
                        "positions": "float[8] (last is gripper_cmd_sent in {0,1})",
                        "velocities": "float[8] (last is 0.0)",
                    },
                }
            },
        }

        if self.phase_labels:
            meta["schema"]["steps_jsonl"]["phase_index"] = "int (zero-based operator phase)"
        if extra_meta:
            meta.update(extra_meta)

        # Write meta.json
        meta_path = os.path.join(self.episode_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta_path

    def close(self) -> None:
        """Close without writing meta.json (for error cases)."""
        if self._closed:
            return
        self._closed = True

        try:
            self._data_writer.close()
        except Exception:
            pass
        try:
            self._image_writer.close()
        except Exception:
            pass
