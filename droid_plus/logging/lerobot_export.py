# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Export franky_service runs to LeRobot v2.1 dataset format.

Reads:
    output/<run>/episode_NNN/{meta.json, steps.jsonl, <cam>/*.jpg}

Writes:
    <out>/
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl
            episodes_stats.jsonl
            stats.json                  (aggregate, for v2.0 compatibility)
        data/chunk-000/
            episode_000000.parquet
            ...
        videos/chunk-000/
            observation.images.left/episode_000000.mp4
            observation.images.right/episode_000000.mp4
            observation.images.wrist/episode_000000.mp4

LeRobot v2.1 conventions used here
- ``observation.state``: 8-D float32 = 7 joint positions + gripper fraction
- ``action``:            8-D float32 = 7 commanded joints + gripper command
- ``observation.images.{left,right,wrist}``: video features, always present.
  If a camera is missing for a given episode, a black MP4 of the same length
  is emitted so the dataset has a uniform feature set.
- ``phase_index``: operator phase ID (int64), -1 for legacy unlabelled episodes.
- Per-episode parquet schema:
    observation.state, action, timestamp, frame_index, episode_index, index, task_index

Optional ``push_to_hub`` path uses the ``lerobot`` library if installed.
The direct writer has no ``lerobot`` dependency.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

CAMERAS: tuple[str, ...] = ("left", "right", "wrist")
JOINT_NAMES: tuple[str, ...] = ("j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper")
STATE_DIM: int = 8
ACTION_DIM: int = 8
CHUNK_SIZE: int = 1000
CODEBASE_VERSION: str = "v2.1"
DEFAULT_VCODEC: str = "libx264"
DEFAULT_PIX_FMT: str = "yuv420p"
DEFAULT_CRF: int = 23
DEFAULT_ROBOT_TYPE: str = "franka"
DATA_PATH_TEMPLATE: str = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE: str = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

IMAGE_STATS_SAMPLE_FRAMES: int = 8


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ExportConfig:
    """Configuration for a single run-dir → LeRobot dataset export.

    """
    run_dir: Path
    out_dir: Path
    fps: Optional[float] = None
    robot_type: str = DEFAULT_ROBOT_TYPE
    vcodec: str = DEFAULT_VCODEC
    pix_fmt: str = DEFAULT_PIX_FMT
    crf: int = DEFAULT_CRF
    overwrite: bool = False
    image_stats_sample: int = IMAGE_STATS_SAMPLE_FRAMES


# ── Episode-level structures ─────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    """Resolved per-episode data after reading meta + steps."""
    src_dir: Path
    episode_index: int
    instruction: str
    state: np.ndarray            # (N, 8) float32
    action: np.ndarray           # (N, 8) float32
    timestamps: np.ndarray       # (N,) float32, seconds since episode start
    image_paths: dict[str, list[Optional[Path]]]  # cam → list of N paths (None if missing)
    image_hw: Optional[tuple[int, int]]           # (H, W) discovered from any present frame
    fps_source: str              # "control.record_rate_hz" | "control.rate_hz" | "computed"
    fps_value: float
    phase_index: np.ndarray | None = None  # None for legacy unlabelled episodes


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH; required for LeRobot video export")


def _detect_image_hw(frame_path: Path) -> tuple[int, int]:
    import cv2  # imported lazily; opencv-python is a hard dep already
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image {frame_path}")
    h, w = img.shape[:2]
    return int(h), int(w)


def _infer_fps(meta: dict[str, Any], timestamps: Optional[np.ndarray] = None) -> tuple[float, str]:
    """Infer fps from meta. Prefers control.record_rate_hz, falls back to control.rate_hz."""
    ctrl = meta.get("control") or {}
    if "record_rate_hz" in ctrl and ctrl["record_rate_hz"]:
        return float(ctrl["record_rate_hz"]), "control.record_rate_hz"
    if "rate_hz" in ctrl and ctrl["rate_hz"]:
        return float(ctrl["rate_hz"]), "control.rate_hz"
    if timestamps is not None and len(timestamps) >= 2:
        dt = float(np.median(np.diff(timestamps)))
        if dt > 0:
            return 1.0 / dt, "computed"
    raise ValueError(
        "Could not infer fps: meta.control has neither record_rate_hz nor rate_hz, "
        "and timestamps are unusable."
    )


def _discover_episode_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "steps.jsonl").exists():
        return [run_dir]
    return sorted(p for p in run_dir.glob("episode_*") if p.is_dir() and (p / "steps.jsonl").exists())


def _zero_frame_jpeg(h: int, w: int, *, quality: int = 90) -> bytes:
    import cv2
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for zero frame")
    return buf.tobytes()


# ── Episode loading ──────────────────────────────────────────────────────────

def _load_episode(ep_dir: Path) -> EpisodeRecord:
    meta_path = ep_dir / "meta.json"
    steps_path = ep_dir / "steps.jsonl"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    steps = _read_jsonl(steps_path)
    if not steps:
        raise RuntimeError(f"Empty episode: {ep_dir}")

    n = len(steps)
    phase_values = [step.get("phase_index") for step in steps]
    phase_index = None
    if any(value is not None for value in phase_values):
        if any(type(value) is not int or not 0 <= value <= 127 for value in phase_values):
            raise ValueError(f"{ep_dir}: phase_index must be present and an integer in [0, 127] on every frame")
        phase_index = np.asarray(phase_values, dtype=np.int64)
    state = np.zeros((n, STATE_DIM), dtype=np.float32)
    action = np.zeros((n, ACTION_DIM), dtype=np.float32)
    t_wall = np.zeros((n,), dtype=np.float64)
    image_paths: dict[str, list[Optional[Path]]] = {cam: [None] * n for cam in CAMERAS}

    for i, step in enumerate(steps):
        st = step.get("state") or {}
        ac = step.get("action") or {}

        st_pos = list(st.get("positions") or [])
        ac_pos = list(ac.get("positions") or [])

        if len(st_pos) < STATE_DIM:
            st_pos = st_pos + [0.0] * (STATE_DIM - len(st_pos))
        if len(ac_pos) < ACTION_DIM:
            ac_pos = ac_pos + [0.0] * (ACTION_DIM - len(ac_pos))

        state[i, :] = np.asarray(st_pos[:STATE_DIM], dtype=np.float32)
        action[i, :] = np.asarray(ac_pos[:ACTION_DIM], dtype=np.float32)
        t_wall[i] = float(step.get("t_wall_s", 0.0))

        imgs = step.get("images") or {}
        for cam in CAMERAS:
            rel = imgs.get(cam)
            if rel:
                p = ep_dir / rel
                if p.exists():
                    image_paths[cam][i] = p

    t0 = float(meta.get("t_start_wall_s") or (t_wall[0] if t_wall.size else 0.0))
    timestamps = (t_wall - t0).astype(np.float32)
    if timestamps[0] < 0:
        timestamps = (t_wall - t_wall[0]).astype(np.float32)

    fps_value, fps_source = _infer_fps(meta, timestamps if timestamps.size >= 2 else None)

    image_hw: Optional[tuple[int, int]] = None
    for cam in CAMERAS:
        for p in image_paths[cam]:
            if p is not None:
                image_hw = _detect_image_hw(p)
                break
        if image_hw is not None:
            break

    return EpisodeRecord(
        src_dir=ep_dir,
        episode_index=int(meta.get("episode_idx", 0)),
        instruction=str(meta.get("instruction") or meta.get("task") or ""),
        state=state,
        action=action,
        timestamps=timestamps,
        image_paths=image_paths,
        image_hw=image_hw,
        fps_source=fps_source,
        fps_value=fps_value,
        phase_index=phase_index,
    )


# ── Video encoding ───────────────────────────────────────────────────────────

def _encode_video_from_frames(
    frame_paths: list[Optional[Path]],
    out_path: Path,
    *,
    fps: float,
    h: int,
    w: int,
    vcodec: str,
    pix_fmt: str,
    crf: int,
) -> None:
    """
    Encode an MP4 from a list of frame paths.

    Frames that are ``None`` are replaced with a black JPEG of the correct size,
    so the resulting video always has ``len(frame_paths)`` frames.

    Uses ffmpeg's concat demuxer over a temp directory of symlinks/copies to
    avoid issues with non-contiguous source filenames.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        zero_jpeg: Optional[bytes] = None
        for i, src in enumerate(frame_paths):
            dst = tmp_dir / f"{i:09d}.jpg"
            if src is not None and src.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    shutil.copyfile(src, dst)
            else:
                if zero_jpeg is None:
                    zero_jpeg = _zero_frame_jpeg(h, w)
                dst.write_bytes(zero_jpeg)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-framerate", f"{fps:g}",
            "-i", str(tmp_dir / "%09d.jpg"),
            "-c:v", vcodec,
            "-pix_fmt", pix_fmt,
            "-crf", str(int(crf)),
            "-vf", f"scale={w}:{h}",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)


# ── Stats ────────────────────────────────────────────────────────────────────

def _array_stats(arr: np.ndarray) -> dict[str, list[float]]:
    """Per-dimension stats over axis 0; works for shape (N, D)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    a = arr.astype(np.float64, copy=False)
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "count": [int(a.shape[0])],
    }


def _image_stats(
    frame_paths: list[Optional[Path]],
    h: int,
    w: int,
    *,
    sample: int = IMAGE_STATS_SAMPLE_FRAMES,
) -> dict[str, list[float]]:
    """Per-channel stats sampled over a few frames; channel order = RGB.

    Pixel values are normalized to [0, 1] to match LeRobot conventions.
    """
    import cv2

    n = len(frame_paths)
    if n == 0:
        zeros3 = [0.0, 0.0, 0.0]
        return {"mean": zeros3, "std": zeros3, "min": zeros3, "max": zeros3, "count": [0]}

    sample = max(1, min(sample, n))
    idxs = np.linspace(0, n - 1, num=sample, dtype=int).tolist()

    means: list[np.ndarray] = []
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    sq_means: list[np.ndarray] = []
    counts = 0

    for i in idxs:
        src = frame_paths[i]
        if src is not None and src.exists():
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            rgb = np.zeros((h, w, 3), dtype=np.float32)
        flat = rgb.reshape(-1, 3)
        means.append(flat.mean(axis=0))
        sq_means.append((flat ** 2).mean(axis=0))
        mins.append(flat.min(axis=0))
        maxs.append(flat.max(axis=0))
        counts += flat.shape[0]

    if not means:
        zeros3 = [0.0, 0.0, 0.0]
        return {"mean": zeros3, "std": zeros3, "min": zeros3, "max": zeros3, "count": [0]}

    mean = np.mean(np.stack(means, axis=0), axis=0)
    sq_mean = np.mean(np.stack(sq_means, axis=0), axis=0)
    var = np.maximum(sq_mean - mean ** 2, 0.0)
    std = np.sqrt(var)
    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": mn.tolist(),
        "max": mx.tolist(),
        "count": [int(counts)],
    }


def _aggregate_global_stats(per_episode: list[dict[str, dict[str, list[float]]]]) -> dict[str, dict[str, list[float]]]:
    """Aggregate per-episode stats into global stats (count-weighted mean/std, elementwise min/max)."""
    if not per_episode:
        return {}

    feature_names = list(per_episode[0].keys())
    out: dict[str, dict[str, list[float]]] = {}

    for fname in feature_names:
        means: list[np.ndarray] = []
        sq_means: list[np.ndarray] = []
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        counts: list[int] = []

        for ep_stats in per_episode:
            s = ep_stats.get(fname)
            if not s:
                continue
            mean = np.asarray(s["mean"], dtype=np.float64)
            std = np.asarray(s["std"], dtype=np.float64)
            mn = np.asarray(s["min"], dtype=np.float64)
            mx = np.asarray(s["max"], dtype=np.float64)
            cnt = int(s["count"][0]) if s.get("count") else 0
            if cnt <= 0:
                continue
            means.append(mean)
            sq_means.append(std ** 2 + mean ** 2)
            mins.append(mn)
            maxs.append(mx)
            counts.append(cnt)

        if not counts:
            continue

        w = np.asarray(counts, dtype=np.float64)
        w = w / w.sum()
        agg_mean = np.sum(np.stack(means, axis=0) * w[:, None], axis=0)
        agg_sq = np.sum(np.stack(sq_means, axis=0) * w[:, None], axis=0)
        agg_var = np.maximum(agg_sq - agg_mean ** 2, 0.0)
        agg_std = np.sqrt(agg_var)
        agg_min = np.min(np.stack(mins, axis=0), axis=0)
        agg_max = np.max(np.stack(maxs, axis=0), axis=0)

        out[fname] = {
            "mean": agg_mean.tolist(),
            "std": agg_std.tolist(),
            "min": agg_min.tolist(),
            "max": agg_max.tolist(),
            "count": [int(sum(counts))],
        }
    return out


# ── Parquet writing ──────────────────────────────────────────────────────────

def _write_episode_parquet(
    out_path: Path,
    *,
    episode: EpisodeRecord,
    episode_index: int,
    global_index_start: int,
    task_index: int,
    fps: float,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = episode.state.shape[0]
    frame_index = np.arange(n, dtype=np.int64)
    timestamps = (frame_index.astype(np.float64) / float(fps)).astype(np.float32)
    episode_idx_col = np.full((n,), episode_index, dtype=np.int64)
    task_idx_col = np.full((n,), task_index, dtype=np.int64)
    global_index = np.arange(global_index_start, global_index_start + n, dtype=np.int64)

    state_list = pa.array(
        [row.tolist() for row in episode.state.astype(np.float32)],
        type=pa.list_(pa.float32(), STATE_DIM),
    )
    action_list = pa.array(
        [row.tolist() for row in episode.action.astype(np.float32)],
        type=pa.list_(pa.float32(), ACTION_DIM),
    )

    table = pa.table(
        {
            "observation.state": state_list,
            "action": action_list,
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_idx_col, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_idx_col, type=pa.int64()),
            "phase_index": pa.array(
                episode.phase_index if episode.phase_index is not None else np.full(n, -1, dtype=np.int64),
                type=pa.int64(),
            ),
        }
    )
    pq.write_table(table, out_path)


# ── Meta writers ─────────────────────────────────────────────────────────────

def _build_features(h: int, w: int, fps: float) -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": list(JOINT_NAMES),
        },
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": DEFAULT_VCODEC,
                "video.pix_fmt": DEFAULT_PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
                "video.fps": float(fps),
            },
        }
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    feats["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["phase_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return feats


def _write_info_json(
    out_dir: Path,
    *,
    robot_type: str,
    fps: float,
    h: int,
    w: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> None:
    total_videos = total_episodes * len(CAMERAS)
    total_chunks = max(1, math.ceil(total_episodes / CHUNK_SIZE))
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": int(total_videos),
        "total_chunks": int(total_chunks),
        "chunks_size": int(CHUNK_SIZE),
        "fps": float(fps),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": _build_features(h, w, fps),
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _write_episodes_jsonl(out_dir: Path, episodes: list[tuple[int, str, int]]) -> None:
    """episodes: list of (episode_index, instruction, length)."""
    path = out_dir / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ep_idx, instruction, length in episodes:
            f.write(json.dumps({
                "episode_index": int(ep_idx),
                "tasks": [instruction],
                "length": int(length),
            }) + "\n")


def _write_tasks_jsonl(out_dir: Path, tasks: list[tuple[int, str]]) -> None:
    path = out_dir / "meta" / "tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tidx, task in tasks:
            f.write(json.dumps({"task_index": int(tidx), "task": task}) + "\n")


def _write_episodes_stats_jsonl(
    out_dir: Path,
    per_episode: list[tuple[int, dict[str, dict[str, list[float]]]]],
) -> None:
    path = out_dir / "meta" / "episodes_stats.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ep_idx, stats in per_episode:
            f.write(json.dumps({"episode_index": int(ep_idx), "stats": stats}) + "\n")


def _write_stats_json(out_dir: Path, stats: dict[str, dict[str, list[float]]]) -> None:
    (out_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


# ── Main exporter ────────────────────────────────────────────────────────────

class LeRobotExporter:
    """Convert a franky_service run directory into a LeRobot v2.1 dataset.

    Example
    -------
    >>> cfg = ExportConfig(
    ...     run_dir=Path("output/banana_pi05_20260118"),
    ...     out_dir=Path("output/lerobot/banana_pi05_20260118"),
    ... )
    >>> exporter = LeRobotExporter(cfg)
    >>> exporter.export()
    """

    def __init__(self, cfg: ExportConfig) -> None:
        self.cfg = cfg

    def export(self) -> Path:
        cfg = self.cfg
        _check_ffmpeg()

        run_dir = cfg.run_dir.resolve()
        out_dir = cfg.out_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(str(run_dir))

        episode_dirs = _discover_episode_dirs(run_dir)
        if not episode_dirs:
            raise RuntimeError(f"No episodes found under {run_dir}")

        if out_dir.exists():
            if not cfg.overwrite:
                raise FileExistsError(
                    f"Output directory exists: {out_dir} (pass overwrite=True to replace)"
                )
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (out_dir / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(
                parents=True, exist_ok=True,
            )

        # Pass 1: load all episodes (to determine fps + image H/W).
        episodes: list[EpisodeRecord] = []
        for ep_dir in episode_dirs:
            episodes.append(_load_episode(ep_dir))

        if cfg.fps is not None:
            fps = float(cfg.fps)
        else:
            fps_values = [e.fps_value for e in episodes]
            if len(set(round(v, 4) for v in fps_values)) > 1:
                print(f"[lerobot_export] Warning: episodes have heterogeneous fps {fps_values}; using first ({fps_values[0]})")
            fps = fps_values[0]

        hw_candidates = [e.image_hw for e in episodes if e.image_hw is not None]
        if not hw_candidates:
            raise RuntimeError("No image frames found in any episode; cannot determine H/W")
        h, w = hw_candidates[0]
        if any(hw != (h, w) for hw in hw_candidates):
            raise RuntimeError(
                f"Heterogeneous image sizes across episodes: {set(hw_candidates)}; "
                "LeRobot requires a single H/W per feature."
            )

        # Build task table (dedupe instructions, preserve first-seen order).
        instruction_to_task_idx: dict[str, int] = {}
        for ep in episodes:
            instr = ep.instruction
            if instr not in instruction_to_task_idx:
                instruction_to_task_idx[instr] = len(instruction_to_task_idx)
        tasks: list[tuple[int, str]] = sorted(
            ((idx, instr) for instr, idx in instruction_to_task_idx.items()),
            key=lambda x: x[0],
        )

        # Pass 2: write per-episode parquet + videos, accumulate stats and meta.
        episodes_meta: list[tuple[int, str, int]] = []
        per_episode_stats: list[tuple[int, dict[str, dict[str, list[float]]]]] = []
        global_index = 0
        total_frames = 0

        for new_ep_idx, ep in enumerate(episodes):
            n = ep.state.shape[0]
            length = int(n)

            parquet_path = out_dir / DATA_PATH_TEMPLATE.format(
                episode_chunk=0, episode_index=new_ep_idx,
            )
            _write_episode_parquet(
                parquet_path,
                episode=ep,
                episode_index=new_ep_idx,
                global_index_start=global_index,
                task_index=instruction_to_task_idx[ep.instruction],
                fps=fps,
            )

            for cam in CAMERAS:
                video_path = out_dir / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=0,
                    video_key=f"observation.images.{cam}",
                    episode_index=new_ep_idx,
                )
                _encode_video_from_frames(
                    ep.image_paths[cam],
                    video_path,
                    fps=fps,
                    h=h,
                    w=w,
                    vcodec=cfg.vcodec,
                    pix_fmt=cfg.pix_fmt,
                    crf=cfg.crf,
                )

            ep_stats: dict[str, dict[str, list[float]]] = {
                "observation.state": _array_stats(ep.state),
                "action": _array_stats(ep.action),
                "phase_index": _array_stats(
                    ep.phase_index if ep.phase_index is not None else np.full(n, -1, dtype=np.int64)
                ),
                "timestamp": _array_stats((np.arange(n, dtype=np.float32) / float(fps))),
                "frame_index": _array_stats(np.arange(n, dtype=np.int64).astype(np.float64)),
                "episode_index": _array_stats(np.full((n,), new_ep_idx, dtype=np.float64)),
                "index": _array_stats(np.arange(global_index, global_index + n, dtype=np.float64)),
                "task_index": _array_stats(
                    np.full((n,), instruction_to_task_idx[ep.instruction], dtype=np.float64)
                ),
            }
            for cam in CAMERAS:
                ep_stats[f"observation.images.{cam}"] = _image_stats(
                    ep.image_paths[cam], h, w, sample=cfg.image_stats_sample,
                )

            per_episode_stats.append((new_ep_idx, ep_stats))
            episodes_meta.append((new_ep_idx, ep.instruction, length))
            global_index += n
            total_frames += n

        # Write meta files.
        _write_info_json(
            out_dir,
            robot_type=cfg.robot_type,
            fps=fps,
            h=h,
            w=w,
            total_episodes=len(episodes),
            total_frames=total_frames,
            total_tasks=len(tasks),
        )
        _write_episodes_jsonl(out_dir, episodes_meta)
        _write_tasks_jsonl(out_dir, tasks)
        _write_episodes_stats_jsonl(out_dir, per_episode_stats)
        global_stats = _aggregate_global_stats([s for _, s in per_episode_stats])
        _write_stats_json(out_dir, global_stats)

        return out_dir


# ── Optional: push via the lerobot library ───────────────────────────────────

def push_to_hub(
    dataset_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    tags: Optional[list[str]] = None,
) -> None:
    """Push an already-exported LeRobot v2.1 dataset to the Hugging Face Hub.

    Lazily imports ``lerobot``; raises a clear error if it isn't installed.

    Use the direct exporter first, then this to publish:

        exporter = LeRobotExporter(cfg)
        out = exporter.export()
        push_to_hub(out, repo_id="my-org/my-dataset")
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A compatible lerobot installation is required to push to the Hub; "
            'install the DROID+ teleop extra with `pip install -e ".[teleop]"`.'
        ) from e

    ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_dir))
    ds.push_to_hub(private=private, tags=tags)


# ── Convenience entry ────────────────────────────────────────────────────────

def export_run(
    run_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    fps: Optional[float] = None,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    vcodec: str = DEFAULT_VCODEC,
    pix_fmt: str = DEFAULT_PIX_FMT,
    crf: int = DEFAULT_CRF,
    overwrite: bool = False,
) -> Path:
    """Functional shortcut around ``LeRobotExporter``."""
    cfg = ExportConfig(
        run_dir=Path(run_dir),
        out_dir=Path(out_dir),
        fps=fps,
        robot_type=robot_type,
        vcodec=vcodec,
        pix_fmt=pix_fmt,
        crf=crf,
        overwrite=overwrite,
    )
    return LeRobotExporter(cfg).export()
