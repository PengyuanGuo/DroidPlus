#!/usr/bin/env python3
"""Visualize sorting zarr in Rerun: camera images, points, phases and robot signals.

Run from DroidPlus:
    .venv-viz/bin/python scripts/visualize_zarr.py --episode 0
    .venv-viz/bin/python scripts/visualize_zarr.py --all-episodes --stride 3
    .venv-viz/bin/python scripts/visualize_zarr.py --no-spawn --output reports/demo.rrd

The zarr is opened read-only. An RRD is always saved; the viewer opens by default.
For schema/source checks use scripts/verify_droidplus_zarr.py instead.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .visualize_droidplus_zarr_rerun import export_recording
else:
    from visualize_droidplus_zarr_rerun import export_recording

PROJECT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", type=Path, default=PROJECT / "data/droidplus_sorting_annotated.zarr")
    parser.add_argument("--episode", type=int, default=0, help="Global zarr episode index, not the source folder suffix")
    parser.add_argument("--all-episodes", action="store_true", help="Visualize all episodes")
    parser.add_argument("--stride", type=int, default=1, help="Log every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit logged frames; 0 means unlimited")
    parser.add_argument("--output", type=Path, help="Save .rrd here; default uses a new timestamped path in reports/rerun/")
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True, help="Open Rerun viewer (default); use --no-spawn to save only")
    args = parser.parse_args(argv)
    if args.episode < 0 or args.stride <= 0 or args.max_frames < 0:
        parser.error("episode/max-frames must be nonnegative and stride must be positive")
    path = args.zarr.resolve()
    if not path.is_dir():
        parser.error(f"Zarr directory does not exist: {path}")
    label = "all" if args.all_episodes else f"episode{args.episode:03d}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or PROJECT / "reports/rerun" / f"sorting_{label}_{stamp}.rrd").resolve()
    if output.suffix.lower() != ".rrd":
        parser.error("output must have the .rrd extension")
    try:
        count = export_recording(path, output, episode=args.episode, all_episodes=args.all_episodes,
                                 stride=args.stride, max_frames=args.max_frames, spawn=args.spawn)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("rerun"):
            parser.error("Rerun SDK is missing. Install rerun-sdk in a separate visualization environment; see docs/sorting-zarr-inspection.md")
        raise
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {count} frames to {output}")
    print("Views: agent + point, wrist, phase/task text, gripper and joint state/action plots.")
    if not args.spawn:
        print(f"Open with the Rerun executable from this Python environment: rerun {output}")
    return output


if __name__ == "__main__":
    main()
