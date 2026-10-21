#!/usr/bin/env python3
"""Export converted sorting episodes to a Rerun recording; never modify zarr."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def export_recording(path: Path, output: Path, *, episode=0, all_episodes=False, stride=1, max_frames=0, spawn=False) -> int:
    import rerun as rr
    import rerun.blueprint as rrb

    root = zarr.open_group(str(path), mode="r")
    data = root["data"]
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    if not 0 <= episode < len(ends) or stride < 1 or max_frames < 0:
        raise ValueError("Invalid episode, stride or max-frames")
    if output.exists():
        raise FileExistsError(f"Recording already exists: {output}; use a new output path")
    manifest_path = path / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("droidplus_sorting_inspection", spawn=spawn, strict=True)
    rr.save(str(output))
    rr.send_blueprint(rrb.Vertical(
        rrb.Horizontal(rrb.Spatial2DView(origin="cameras/agent_view", name="Agent + point"), rrb.Spatial2DView(origin="cameras/wrist_cam", name="Wrist")),
        rrb.Horizontal(rrb.TextDocumentView(origin="phase", name="Episode and phase"), rrb.TimeSeriesView(origin="signals/gripper", name="Gripper: 1=open")),
        rrb.TimeSeriesView(origin="signals/joints", name="Joint state and commanded targets"),
    ))
    count, clock_offset = 0, 0.0
    selected = range(len(ends)) if all_episodes else [episode]
    try:
        for episode_index in selected:
            start, end = int(starts[episode_index]), int(ends[episode_index])
            timestamps = data["timestamp"][start:end]
            labels = data["phase_label"][start:end]
            segment_ids = np.cumsum(np.r_[False, labels[1:] != labels[:-1]])
            source_name = str(episode_index)
            if manifest:
                source_name = str(Path(manifest["episodes"][episode_index]["source_steps"]).parent)
            # Read the CHW caches actually used by the training loader.
            agent = root["_cache/224x224/agent_view.zarr/images"]
            wrist = root["_cache/224x224/wrist_cam.zarr/images"]
            for frame in range(start, end, stride):
                if max_frames and count >= max_frames:
                    break
                local = frame - start
                rr.set_time("frame", sequence=frame)
                rr.set_time("recording_time", duration=clock_offset + float(timestamps[local]))
                agent_rgb, wrist_rgb = agent[frame].transpose(1, 2, 0), wrist[frame].transpose(1, 2, 0)
                point = data["point_pixel"][frame]
                prompt = bytes(data["phase_prompt"][frame]).rstrip(b"\0").decode("utf-8")
                task = bytes(data["prompt"][frame]).rstrip(b"\0").decode("utf-8")
                review = bool(data["point_needs_review"][frame])
                rr.log("cameras/agent_view", rr.Image(agent_rgb).compress(jpeg_quality=95))
                rr.log("cameras/wrist_cam", rr.Image(wrist_rgb).compress(jpeg_quality=95))
                height, width = agent_rgb.shape[:2]
                if np.isfinite(point).all() and ((point >= 0) & (point <= 1)).all():
                    pixel = point * [width, height]
                    rr.log("cameras/agent_view/point", rr.Points2D([pixel], radii=3, colors=[255, 70, 50] if review else [30, 255, 80], labels=[prompt]))
                else:
                    rr.log("cameras/agent_view/point", rr.Clear(recursive=True))
                state, action = data["state"][frame], data["action"][frame]
                for joint in range(7):
                    rr.log(f"signals/joints/j{joint}/state", rr.Scalars(float(state[joint])))
                    rr.log(f"signals/joints/j{joint}/action", rr.Scalars(float(action[joint])))
                rr.log("signals/gripper/state", rr.Scalars(float(state[7])))
                rr.log("signals/gripper/action", rr.Scalars(float(action[7])))
                rr.log("signals/phase_label", rr.Scalars(int(labels[local])))
                rr.log("signals/needs_review", rr.Scalars(int(review)))
                rr.log("phase", rr.TextDocument(
                    f"Episode index: {episode_index}\n{source_name}\n\nTask: {task}\n"
                    f"Frame: {local} (global {frame}, source seq {int(data['source_seq'][frame])})\n"
                    f"Segment: {int(segment_ids[local]):02d}; phase_index: {int(labels[local])}\n"
                    f"Phase: {prompt}\nPoint xy (normalized): {point.tolist()}\nNeeds review: {review}\n"
                    "\nGreen point = no review flag; red point = needs review.\n"
                    "Points are phase-level conditioning, not per-frame object tracking."
                ))
                count += 1
            step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.1
            clock_offset += float(timestamps[-1]) + step
            if max_frames and count >= max_frames:
                break
    finally:
        rr.disconnect()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .rrd recording")
    parser.add_argument("--episode", type=int, default=0, help="Global episode index in manifest (not folder suffix)")
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all selected frames")
    parser.add_argument("--spawn", action="store_true", help="Also open the native Rerun viewer")
    args = parser.parse_args()
    count = export_recording(args.zarr.resolve(), args.output.resolve(), episode=args.episode, all_episodes=args.all_episodes, stride=args.stride, max_frames=args.max_frames, spawn=args.spawn)
    print(f"Exported {count} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
