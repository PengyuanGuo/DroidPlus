# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


def _iter_steps(jsonl_path: Path) -> Iterable[dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _find_episode_dirs(run_dir: Path) -> list[Path]:
    # If run_dir itself looks like an episode dir, just return it.
    if (run_dir / "steps.jsonl").exists():
        return [run_dir]

    eps = sorted([p for p in run_dir.glob("episode_*") if p.is_dir()])
    return [p for p in eps if (p / "steps.jsonl").exists()]


def export_episode(episode_dir: Path) -> Path:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to export parquet. Install it with: pip install pyarrow"
        ) from e

    steps_path = episode_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing {steps_path}")

    meta_path = episode_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    episode_idx = meta.get("episode_idx")
    instruction = meta.get("instruction")

    t_wall_s: list[float] = []
    seq: list[int] = []
    state_positions: list[list[float]] = []
    state_velocities: list[list[float]] = []
    action_positions: list[list[float]] = []
    action_velocities: list[list[float]] = []
    episode_idx_col: list[int | None] = []
    instruction_col: list[str | None] = []
    phase_index: list[int | None] = []

    for step in _iter_steps(steps_path):
        phase_index.append(step.get("phase_index"))
        t_wall_s.append(float(step.get("t_wall_s", 0.0)))
        seq.append(int(step.get("seq", len(seq))))

        st = step.get("state") or {}
        ac = step.get("action") or {}
        state_positions.append([float(x) for x in (st.get("positions") or [])])
        state_velocities.append([float(x) for x in (st.get("velocities") or [])])
        action_positions.append([float(x) for x in (ac.get("positions") or [])])
        action_velocities.append([float(x) for x in (ac.get("velocities") or [])])

        episode_idx_col.append(int(episode_idx) if episode_idx is not None else None)
        instruction_col.append(str(instruction) if instruction is not None else None)

    table = pa.table(
        {
            "t_wall_s": pa.array(t_wall_s, type=pa.float64()),
            "seq": pa.array(seq, type=pa.int64()),
            "phase_index": pa.array(phase_index, type=pa.int64()),
            "state_positions": pa.array(state_positions, type=pa.list_(pa.float64())),
            "state_velocities": pa.array(state_velocities, type=pa.list_(pa.float64())),
            "action_positions": pa.array(action_positions, type=pa.list_(pa.float64())),
            "action_velocities": pa.array(action_velocities, type=pa.list_(pa.float64())),
            "episode_idx": pa.array(episode_idx_col, type=pa.int64()),
            "instruction": pa.array(instruction_col, type=pa.string()),
        }
    )

    out_path = episode_dir / "steps.parquet"
    pq.write_table(table, out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export runs/recordings JSONL steps to Parquet.")
    p.add_argument("run_dir", help="Run dir (contains episode_*/ ) or episode dir (contains steps.jsonl)")
    args = p.parse_args()

    run_dir = Path(os.path.expanduser(args.run_dir)).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    episode_dirs = _find_episode_dirs(run_dir)
    if not episode_dirs:
        raise RuntimeError(f"No episodes found under {run_dir} (expected episode_*/steps.jsonl)")

    for ep in episode_dirs:
        out = export_episode(ep)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
