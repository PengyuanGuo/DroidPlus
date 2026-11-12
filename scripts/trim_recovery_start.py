#!/usr/bin/env python3
"""Cut the start of annotated recovery episodes so they begin at the failed state (failure -> success only).

The start is given as an image file number (wrist/000000100.jpg -> 100), as picked by eye. Everything before it is
dropped from steps.jsonl, steps_molmo.jsonl and ee_trajectory.json; frame indices in molmo_point_annotations.json
(segments, samples, wrist windows) shift down; meta.json gets the new step_count and recovery.trim_start / frames /
trim_reason. The labels on the kept frames do not change (left points are fixed per segment, wrist points are
per-frame projections of a fixed anchor), so no re-annotation is needed.

The files as they were before the first cut are kept as *.pretrim; every run starts from those, so changing a start
frame and rerunning is safe. Afterwards rerun scripts/recovery_qa.py (review page) and the zarr / MolmoBot / LeRobot
conversion.

    python scripts/trim_recovery_start.py scripts/recovery_starts/crossword_bigletter.json [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

FILES = ("steps.jsonl", "steps_molmo.jsonl", "ee_trajectory.json", "meta.json", "molmo_point_annotations.json")
MIN_KEEP = 30


DRAWER_PLACE_Z = 0.08   # median bin target height of the full drawer data


def frame_no(row: dict) -> int:
    return int(re.search(r"(\d+)\.\w+$", row["images"]["wrist"])[1])


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extend_front(ep: Path, steps, molmo, ee, start_img: int, first_img: int):
    """Prepend raw rows (steps.jsonl.orig, ee_trajectory.json.orig) for images start_img .. first_img-1, labelled like
    the first kept row (phase, prompt, left point); the gripper command is the raw one (0/1)."""
    raw = read_jsonl(ep / "steps.jsonl.orig")
    raw_nums = [frame_no(r) for r in raw]
    if len(set(raw_nums)) != len(raw_nums):
        raise ValueError(f"{ep}: repeated image numbers in steps.jsonl.orig")
    idx = [raw_nums.index(i) for i in range(start_img, first_img)]
    s0, m0 = steps[0], molmo[0]
    new_s, new_m = [], []
    for i in idx:
        r = dict(raw[i])
        pos = list(r["action"]["positions"])
        pos[7] = 1.0 if pos[7] > 0.5 else 0.0
        r["action"] = {**r["action"], "positions": pos}
        r["phase_index"], r["phase_prompt"] = s0["phase_index"], s0["phase_prompt"]
        new_s.append(r)
        m = dict(r)
        m["point_pixel"] = m0["point_pixel"]
        new_m.append(m)
    ee_raw = json.loads((ep / "ee_trajectory.json.orig").read_text())
    n_raw, n = len(raw), len(molmo)
    for key, v in ee.items():
        if isinstance(v, list) and len(v) == n:
            rv = ee_raw[key][len(ee_raw[key]) - n_raw:]
            ee[key] = [rv[i] for i in idx] + v
    return new_s + steps, new_m + molmo, ee, len(idx)


def relabel_wrist(molmo: list[dict], ann: dict, rec: dict):
    """Recompute the wrist labels of a recovery episode (same rule as scripts/annotate_wrist_points.py)."""
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "atomic_policy"))
    from atomic_policy.wrist.geometry import WristCamera, panda_fk_batch
    from atomic_policy.wrist.targets import wrist_labels
    drawer = rec.get("layout") == "chicken_six_phase"
    if drawer and (rec.get("phases") or [None])[0] != 3:
        raise ValueError("restoring frames is only supported for drawer grasp + pre-place recoveries (contact-anchored)")
    if not drawer and (rec.get("phase") not in (1, 3) or rec.get("layout", "four_phase") != "four_phase"):
        raise ValueError("restoring frames is only supported for grasp / place recoveries (contact-anchored)")
    w = ann["wrist"]
    cam = WristCamera.load(Path(w["calibration"]))
    T = panda_fk_batch(np.array([r["state"]["positions"][:7] for r in molmo]))
    cmd = np.array([r["action"]["positions"][7] for r in molmo]) > 0.5
    phases = [r["phase_index"] for r in molmo]
    if drawer:   # as annotate_wrist_points.py --task drawer_chicken --place-z 0.08 (docs/recovery-stacking-drawer-20261107.md)
        lab = wrist_labels(cam, T, cmd, phases, w["place_drop_m"], task="drawer_chicken", place_z=DRAWER_PLACE_Z)
    else:
        lab = wrist_labels(cam, T, cmd, phases, w["place_drop_m"], task="recovery", phase=rec["phase"])
    for f, r in enumerate(molmo):
        vis = bool(lab["visible"][f])
        r["wrist_point_pixel"] = [round(float(v), 6) for v in lab["uv"][f]] if vis else [-1.0, -1.0]
        r["wrist_point_visible"] = vis
        r["wrist_point_held"] = bool(lab["held"][f])
    w["windows"] = [{"segments": x.segments, "phases": [lab["segments"][k][0] for k in x.segments],
                     "start_frame": x.start, "end_frame": x.end, "role": x.role, "contact_frame": x.contact,
                     "release_frame": x.release, "target_base_m": [round(v, 5) for v in x.point_base],
                     "visible_frames": int(lab["visible"][x.start:x.end].sum()), "needs_review": False}
                    for x in lab["windows"]]


def trim_episode(ep: Path, start_img: int, dry: bool) -> dict:
    src = {n: (ep / f"{n}.pretrim" if (ep / f"{n}.pretrim").exists() else ep / n) for n in FILES}
    steps, molmo = read_jsonl(src["steps.jsonl"]), read_jsonl(src["steps_molmo.jsonl"])
    n = len(molmo)
    if len(steps) != n or [frame_no(r) for r in steps] != [frame_no(r) for r in molmo]:
        raise ValueError(f"{ep}: steps.jsonl and steps_molmo.jsonl rows differ")
    nums = [frame_no(r) for r in molmo]
    meta = json.loads(src["meta.json"].read_text())
    rec = meta["recovery"]
    base_trim = rec.get("trim_start") or 0
    ee = json.loads(src["ee_trajectory.json"].read_text())
    added = 0
    if start_img < nums[0]:
        # the prep cut too late: restore the raw rows from start_img up to the current first row
        steps, molmo, ee, added = extend_front(ep, steps, molmo, ee, start_img, nums[0])
        n, base_trim = len(molmo), base_trim - added
        nums = [frame_no(r) for r in molmo]
    if start_img not in nums:
        raise ValueError(f"{ep}: no row with image {start_img} (images {nums[0]}-{nums[-1]})")
    k = nums.index(start_img)
    if n - k < MIN_KEEP:
        raise ValueError(f"{ep}: only {n - k} frames left after the cut")
    info = {"episode": str(ep), "start_image": start_img, "rows_cut": k, "rows_restored": added,
            "frames_before": n - added, "frames_after": n - k}
    if dry:
        return info
    for name in FILES:
        if not (ep / f"{name}.pretrim").exists():
            shutil.copy2(ep / name, ep / f"{name}.pretrim")

    ann = json.loads(src["molmo_point_annotations.json"].read_text())
    if added:
        for s in ann["segments"]:
            s["end_frame"] += added
            if s["start_frame"]:
                s["start_frame"] += added
            if "preview_frame" in s:
                s["preview_frame"] += added
            for smp in s.get("samples", []):
                smp["frame_index"] += added
            if "sample_indices" in s:
                s["sample_indices"] = [i + added for i in s["sample_indices"]]
        relabel_wrist(molmo, ann, rec)
    write_jsonl(ep / "steps.jsonl", steps[k:])
    write_jsonl(ep / "steps_molmo.jsonl", molmo[k:])
    for key, v in ee.items():
        if isinstance(v, list) and len(v) == n:
            ee[key] = v[k:]
    (ep / "ee_trajectory.json").write_text(json.dumps(ee))

    sh = lambda f: None if f is None else max(0, f - k)
    for s in ann["segments"]:
        s["start_frame"], s["end_frame"] = sh(s["start_frame"]), sh(s["end_frame"])
        if "preview_frame" in s:
            s["preview_frame"] = min(max(sh(s["preview_frame"]), s["start_frame"]), s["end_frame"] - 1)
        # samples before the cut keep their (now negative) index so the point's provenance stays visible
        for smp in s.get("samples", []):
            smp["frame_index"] -= k
        if "sample_indices" in s:
            s["sample_indices"] = [i - k for i in s["sample_indices"]]
    vis = [bool(r.get("wrist_point_visible")) for r in molmo[k:]]
    for w in ann.get("wrist", {}).get("windows", []):
        for f in ("start_frame", "end_frame", "contact_frame", "release_frame"):
            w[f] = sh(w.get(f))
        w["visible_frames"] = sum(vis[w["start_frame"]:w["end_frame"]])
    if "wrist" in ann:
        ann["wrist"]["steps_sha256"] = sha(ep / "steps_molmo.jsonl")
    ann["source_sha256"] = sha(ep / "steps.jsonl")
    ann["trim"] = {"rows_cut": k, "rows_restored": added, "start_image": start_img, "date": str(date.today()),
                   "note": "frame indices shifted by scripts/trim_recovery_start.py; originals in *.pretrim"}
    (ep / "molmo_point_annotations.json").write_text(json.dumps(ann, indent=2))

    rec.update(trim_start=base_trim + k, frames=n - k,
               trim_reason=f"manual start at image {start_img} (failed state onward, set by eye {date.today()})"
               + (f"; before that: {rec['trim_reason']}" if rec.get("trim_reason") else ""))
    meta["step_count"] = n - k
    (ep / "meta.json").write_text(json.dumps(meta, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("starts", type=Path, help="JSON with 'dataset' (relative to the DroidPlus root) and 'starts'")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.starts.read_text())
    ds = args.root / cfg["dataset"]
    for rel, img in cfg["starts"].items():
        i = trim_episode(ds / rel, int(img), args.dry_run)
        print(f"{rel}: start image {img}, restored {i['rows_restored']}, cut {i['rows_cut']} rows, "
              f"{i['frames_before']} -> {i['frames_after']} frames")


if __name__ == "__main__":
    main()
