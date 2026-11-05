#!/usr/bin/env python3
"""Automatic second-look flags for drawer chicken-search point labels (left MolmoPoint points + wrist projections).

Writes DATASET/drawer_qa.json in the stacking_qa.json format, so scripts/review_stacking.py can show it
(python scripts/review_stacking.py --dataset-root DATASET --qa drawer_qa.json --port 18766).
Checks:
  left handle   pre-open point vs the median handle point of that drawer side in the same session (> --max-handle-shift)
  left chicken  share of orange pixels around the pre-grasp / grasp point; the point should sit on the side of the
                drawer opened last (closer to that handle than to the other one, in x)
  left bin      share of blue pixels around the pre-place / place point (the held chicken can cover it: info only
                for place)
  left support  MolmoPoint inliers below the required count, needs_review
  wrist chicken share of early visible frames whose wrist point lands on orange (< --min-wrist-hit)
  wrist windows not 4 + 2 per extra drawer opened, pull shorter than 10 frames, low visibility
Usage: python scripts/drawer_qa.py --dataset-root DATASET
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

HUE = {"orange": [(4, 24)], "blue": [(88, 125)]}
TARGET = {0: "handle", 1: "handle", 2: "chicken leg", 3: "chicken leg", 4: "blue bin", 5: "blue bin"}


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
    ok &= hsv[:, 2] >= (110 if colour == "blue" else 60)  # the gripper fingers are dark blue
    return float(ok.mean())


def episode_qa(ep: Path, handle_median: dict, args) -> dict:
    ann = json.loads((ep / "molmo_point_annotations.json").read_text())
    rows = [json.loads(l) for l in (ep / "steps_molmo.jsonl").read_text().splitlines() if l.strip()]
    session = ep.parent.name
    flags, segs, sides, side = [], [], [], None
    handles = (ann["segments"][0].get("drawer_handles") or {})
    for i, s in enumerate(ann["segments"]):
        p = s["phase_index"]
        if p == 0:
            side = s.get("drawer_side")
            sides.append(side)
        used = [sm["frame_index"] for sm in s.get("samples", []) if sm.get("point_pixel")]
        pf = s.get("preview_frame", int(np.median(used)) if used else s["start_frame"])  # a frame Molmo pointed on
        img = cv2.imread(str(ep / rows[pf]["images"]["left"]))
        colour = {2: "orange", 3: "orange", 4: "blue", 5: "blue"}.get(p)
        share = colour_share(img, *s["point_pixel"], colour) if colour and img is not None else None
        info = {"i": i, "phase": p, "prompt": s["phase_prompt"], "colour": TARGET[p] + (f" ({side})" if p < 2 else ""),
                "start": s["start_frame"], "end": s["end_frame"], "preview_frame": pf, "point": s["point_pixel"],
                "colour_share": None if share is None else round(share, 2), "inliers": s.get("inliers"),
                "required": s.get("required_inliers"), "copied_from": s.get("copied_from_phase"), "flags": []}
        x, y = s["point_pixel"]
        if p == 0 and (session, side) in handle_median:
            d = float(np.hypot(*(np.array([x, y]) - handle_median[(session, side)])))
            info["handle_shift"] = round(d, 3)
            if d > args.max_handle_shift:
                info["flags"].append(f"left handle: {d:.2f} from the session's {side} handle")
        if p in (2, 3):
            if share < args.min_share and p == 2:
                info["flags"].append(f"left point colour: {share:.0%} orange")
            if "left" in handles and "right" in handles and side in ("left", "right"):
                other = "right" if side == "left" else "left"
                if abs(x - handles[side][0]) > abs(x - handles[other][0]):
                    info["flags"].append(f"left chicken point is nearer the {other} handle, last drawer opened: {side}")
        if p == 4 and share < args.min_share:
            info["flags"].append(f"left point colour: {share:.0%} blue")
        if s.get("copied_from_phase") is None and s.get("inliers") is not None and s.get("required_inliers") and s["inliers"] < s["required_inliers"]:
            info["flags"].append("left point: few MolmoPoint inliers")
        if s.get("needs_review"):
            info["flags"].append("left point: needs_review")
        segs.append(info)
    wins = []
    for k, w in enumerate(ann.get("wrist", {}).get("windows", [])):
        p0 = segs[w["segments"][0]]["phase"]
        target = TARGET[p0]
        n = w["end_frame"] - w["start_frame"]
        vis = w["visible_frames"] / max(1, n)
        role = {"object": "handle" if p0 == 0 else "grasp", "pulled": "pull", "place": "place"}[w["role"]]
        info = {"k": k, "role": w["role"], "label": f"{role} · {target}" + (f" ({segs[w['segments'][0]]['colour'].split('(')[-1]}" if p0 < 2 else ""),
                "segments": w["segments"], "start": w["start_frame"], "end": w["end_frame"],
                "contact": w["contact_frame"] if w["role"] != "pulled" else w["release_frame"], "release": w["release_frame"],
                "colour": target, "colour_hit": None, "visible": round(vis, 2), "flags": [], "target_base_m": w["target_base_m"]}
        if p0 == 2:
            end = w["contact_frame"]
            fr = np.linspace(w["start_frame"], end - 1, 10).astype(int)
            hits = []
            for f in fr:
                r = rows[f]
                if r.get("wrist_point_visible"):
                    img = cv2.imread(str(ep / r["images"]["wrist"]))
                    hits.append(colour_share(img, *r["wrist_point_pixel"], "orange", r=9) >= 0.2)
            if hits:
                info["colour_hit"] = round(float(np.mean(hits)), 2)
                if np.mean(hits) < args.min_wrist_hit:
                    info["flags"].append(f"wrist point colour: {np.mean(hits):.0%} of frames before the grasp on orange")
        if w["role"] == "pulled" and w["release_frame"] - w["start_frame"] < 10:
            info["flags"].append(f"pull only {w['release_frame'] - w['start_frame']} frames")
        if vis < (0.4 if p0 in (0, 2) else 0.8):
            info["flags"].append(f"wrist point visible on {vis:.0%} of the window")
        wins.append(info)
    expected = 2 * len(sides) + 2
    if len(wins) != expected:
        flags.append(f"{len(wins)} wrist target windows (expected {expected})")
    for x in segs + wins:
        flags += [f"{'seg ' + str(x['i']) if 'i' in x else 'wrist ' + str(x['k'])}: {f}" for f in x["flags"]]
    return {"segments": segs, "wrist_windows": wins, "frames": len(rows), "flags": flags,
            "variant": " → ".join(s or "?" for s in sides),
            "header": f"drawers opened: {' → '.join(s or '?' for s in sides)} · chicken in the {sides[-1] if sides else '?'} drawer"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--min-share", type=float, default=0.15)
    ap.add_argument("--max-handle-shift", type=float, default=0.06)
    ap.add_argument("--min-wrist-hit", type=float, default=0.5)
    args = ap.parse_args()
    root = args.dataset_root.resolve()
    eps = sorted(p.parent for p in root.glob("*/episode_*/steps_molmo.jsonl"))
    pts = collections.defaultdict(list)
    for ep in eps:
        for s in json.loads((ep / "molmo_point_annotations.json").read_text())["segments"]:
            if s["phase_index"] == 0:
                pts[(ep.parent.name, s.get("drawer_side"))].append(s["point_pixel"])
    handle_median = {k: np.median(v, axis=0) for k, v in pts.items() if len(v) >= 3}
    out = {}
    for ep in eps:
        out[str(ep.relative_to(root))] = q = episode_qa(ep, handle_median, args)
        if q["flags"]:
            print(ep.relative_to(root), q["flags"], flush=True)
    (root / "drawer_qa.json").write_text(json.dumps(out, indent=1))
    n = sum(bool(v["flags"]) for v in out.values())
    print(f"{len(out)} episodes, {n} with flags -> {root / 'drawer_qa.json'}")


if __name__ == "__main__":
    main()
