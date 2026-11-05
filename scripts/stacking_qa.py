#!/usr/bin/env python3
"""Automatic second-look flags for stacking point labels (left MolmoPoint points + wrist projections).

Writes DATASET/stacking_qa.json: per episode the segments / wrist windows with their checks and a list of flags.
Checks:
  left colour   share of block-coloured pixels around the left point on its preview frame
  left target   cycle-2 stack target vs the cycle-1 stack point (the bottom block never moves, block 1 sits on it)
  left support  MolmoPoint inliers below the required count, or a point copied from the partner phase
  wrist colour  grasp targets: share of frames (first 40% of each window) whose wrist point lands on the block colour
                (place targets are mostly hidden behind the held block early on: colour kept as info only)
  wrist target  2nd stack target vs the 1st: < 2.5 cm sideways and 3.5-7 cm higher (block height 5.2 cm)
  wrist windows not 4 target windows (re-grasp or missing release), low wrist visibility
Usage: python scripts/stacking_qa.py --dataset-root DATASET
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HUE = {"red": [(0, 10), (170, 180)], "yellow": [(18, 38)], "blue": [(88, 125)]}


def colour_share(img, x, y, colour, r=7):
    h, w = img.shape[:2]
    cx, cy = int(x * w), int(y * h)
    patch = img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    ok = np.zeros(len(hsv), bool)
    for lo, hi in HUE[colour]:
        ok |= (hsv[:, 0] >= lo) & (hsv[:, 0] <= hi)
    ok &= hsv[:, 1] >= 90
    ok &= hsv[:, 2] >= (110 if colour == "blue" else 70)  # the gripper fingers are dark blue
    return float(ok.mean())


def episode_qa(ep: Path, args) -> dict:
    ann = json.loads((ep / "molmo_point_annotations.json").read_text())
    rows = [json.loads(l) for l in (ep / "steps_molmo.jsonl").read_text().splitlines() if l.strip()]
    flags, segs = [], []
    for i, s in enumerate(ann["segments"]):
        colour = s["moved_block"] if s["phase_index"] < 2 else s["target_block"]
        pf = s.get("preview_frame", s["start_frame"])
        img = cv2.imread(str(ep / rows[pf]["images"]["left"]))
        share = colour_share(img, *s["point_pixel"], colour) if img is not None else 0.0
        info = {"i": i, "phase": s["phase_index"], "prompt": s["phase_prompt"], "colour": colour, "start": s["start_frame"],
                "end": s["end_frame"], "preview_frame": pf, "point": s["point_pixel"], "colour_share": round(share, 2),
                "inliers": s.get("inliers"), "required": s.get("required_inliers"), "copied_from": s.get("copied_from_phase"), "flags": []}
        if share < args.min_share and s["phase_index"] in (0, 2):
            info["flags"].append(f"left point colour: {share:.0%} {colour}")
        if s.get("copied_from_phase") is None and s.get("inliers") is not None and s.get("required_inliers") and s["inliers"] < s["required_inliers"]:
            info["flags"].append("left point: few MolmoPoint inliers")
        if s.get("needs_review"):
            info["flags"].append("left point: needs_review")
        segs.append(info)
    if len(segs) == 8:
        d = float(np.hypot(*(np.array(segs[6]["point"]) - np.array(segs[2]["point"]))))
        segs[6]["stack_shift"] = round(d, 3)
        if d > args.max_shift:
            segs[6]["flags"].append(f"left target: cycle-2 stack point {d:.2f} from cycle-1 stack point")
    wins = []
    for k, w in enumerate(ann.get("wrist", {}).get("windows", [])):
        colour = segs[w["segments"][0]]["colour"]
        end = w["contact_frame"] or w["end_frame"]
        n = max(1, int(0.4 * (end - w["start_frame"])))
        fr = np.linspace(w["start_frame"], w["start_frame"] + n, 6).astype(int)
        hits = []
        for f in fr:
            r = rows[f]
            if not r.get("wrist_point_visible"):
                continue
            img = cv2.imread(str(ep / r["images"]["wrist"]))
            hits.append(colour_share(img, *r["wrist_point_pixel"], colour, r=9) >= 0.2)
        vis = w["visible_frames"] / max(1, w["end_frame"] - w["start_frame"])
        info = {"k": k, "role": w["role"], "segments": w["segments"], "start": w["start_frame"], "end": w["end_frame"],
                "contact": w["contact_frame"], "release": w["release_frame"], "colour": colour,
                "colour_hit": round(float(np.mean(hits)), 2) if hits else None, "visible": round(vis, 2), "flags": []}
        info["target_base_m"] = w["target_base_m"]
        if hits and np.mean(hits) < args.min_wrist_hit and w["role"] == "object":  # place targets hide behind the held block
            info["flags"].append(f"wrist point colour: {np.mean(hits):.0%} of early frames on the {colour} block")
        if vis < 0.8:
            info["flags"].append(f"wrist point visible on {vis:.0%} of the window")
        wins.append(info)
    places = [w for w in wins if w["role"] == "place"]
    if len(places) == 2:
        a, b = np.array(places[0]["target_base_m"]), np.array(places[1]["target_base_m"])
        dxy, dz = float(np.hypot(*(b - a)[:2])), float((b - a)[2])
        places[1]["stack_dxy_m"], places[1]["stack_dz_m"] = round(dxy, 4), round(dz, 4)
        if dxy > args.max_stack_dxy:
            places[1]["flags"].append(f"wrist target: 2nd stack {dxy * 100:.1f} cm sideways from the 1st")
        if not 0.035 <= dz <= 0.07:
            places[1]["flags"].append(f"wrist target: 2nd stack {dz * 100:.1f} cm above the 1st (block = 5.2 cm)")
    if len(wins) != 4:
        flags.append(f"{len(wins)} wrist target windows (expected 4): re-grasp or missing release")
    for x in segs + wins:
        flags += [f"{'seg ' + str(x['i']) if 'i' in x else 'wrist ' + str(x['k'])}: {f}" for f in x["flags"]]
    return {"segments": segs, "wrist_windows": wins, "frames": len(rows), "flags": flags,
            "stack_variant": json.loads((ep / "meta.json").read_text()).get("stack_variant")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--min-share", type=float, default=0.15)
    ap.add_argument("--max-shift", type=float, default=0.09)
    ap.add_argument("--max-stack-dxy", type=float, default=0.025)
    ap.add_argument("--min-wrist-hit", type=float, default=0.5)
    args = ap.parse_args()
    root = args.dataset_root.resolve()
    out = {}
    for ep in sorted(p.parent for p in root.glob("*/episode_*/steps_molmo.jsonl")):
        out[str(ep.relative_to(root))] = q = episode_qa(ep, args)
        if q["flags"]:
            print(ep.relative_to(root), q["flags"], flush=True)
    (root / "stacking_qa.json").write_text(json.dumps(out, indent=1))
    n = sum(bool(v["flags"]) for v in out.values())
    print(f"{len(out)} episodes, {n} with flags -> {root / 'stacking_qa.json'}")


if __name__ == "__main__":
    main()
