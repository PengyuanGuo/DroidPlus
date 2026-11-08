#!/usr/bin/env python3
"""Left-camera MolmoPoint points for the stacking / drawer-chicken recovery episodes (meta.json 'recovery', written
by scripts/prep_recovery_20261107.py). Same targets and queries as annotate_sorting_points.py on the full episodes:

  stacking grasp            the block to pick ("Point to the {colour} block"), first 25% of the episode
  stacking stack            the block it goes onto (top of the 2-block stack), first 25%
  drawer pre-open           that drawer's handle: both handles located on early frames (sorted by image x, as in the
                            full data), MolmoPoint on the side's handle within a window around it; if the handles are
                            not found, the median handle point of that side over the other pre-open recoveries
  drawer open-and-home      the gripper is on the handle from the first frame: median handle point of that side over
                            the pre-open recoveries of the same day (as in the full data, where open-and-home copies
                            the pre-open point = the closed handle)
  drawer grasp              chicken leg, first 25% of the segment
  drawer pre-place          inside of the blue bin, first 25% of the segment
Writes steps_molmo.jsonl, molmo_point_annotations.json and molmo_previews/ like annotate_sorting_points.py.

Usage (DroidPlus, .venv-molmo):
  python scripts/annotate_recovery_points.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase_recovery
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_sorting_points as asp  # noqa: E402

LOG = logging.getLogger("recovery")


def segments_of(rows: list[dict]) -> list[dict]:
    out, s = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["phase_index"] != rows[s]["phase_index"]:
            out.append({"phase_index": rows[s]["phase_index"], "phase_prompt": rows[s]["phase_prompt"],
                        "start_frame": s, "end_frame": i, "recovery": True})
            s = i
    return out


def fixed(seg: dict, point, method: str, n_pool: int) -> dict:
    ok = point is not None
    return {**seg, "molmo_prompt": None, "fallback_prompt": None, "sample_indices": [seg["start_frame"]], "samples": [],
            "valid_predictions": 0, "inliers": n_pool, "required_inliers": 1,
            "point_pixel": list(map(float, point)) if ok else [-1.0, -1.0], "needs_review": not ok,
            "preview_frame": seg["start_frame"], "target_method": method}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--atomic-policy-root", type=Path, default=Path("../atomic_policy"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint", default="allenai/MolmoPoint-8B")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = argparse.Namespace(roi=None, full_frame=True, checkpoint=a.checkpoint, device=a.device, dtype="bfloat16",
                              load_in_4bit=False, max_image_size=448, samples=5, skip_first_frames=2,
                              min_valid_samples=3, overwrite=True, word=None)
    eps = sorted(p.parent for p in a.dataset_root.glob("*/episode_*/meta.json") if "recovery" in json.loads(p.read_text()))
    LOG.info("%d recovery episodes", len(eps))
    import torch
    sys.path.insert(0, str(a.atomic_policy_root.resolve()))
    from atomic_policy.vlm.grounder_molmo import MolmoPointGrounder
    grounder = MolmoPointGrounder(checkpoint=a.checkpoint, device=a.device, dtype=torch.bfloat16, load_in_4bit=False,
                                  max_image_size=448)
    recs = {ep: json.loads((ep / "meta.json").read_text())["recovery"] for ep in eps}
    # pre-open first: their handle points feed open-and-home
    order = sorted(eps, key=lambda e: (recs[e]["phases"] != [0], str(e)))
    handles = collections.defaultdict(list)
    pending, failed = [], 0
    for ep in order:
        rec = recs[ep]
        rows = [json.loads(l) for l in (ep / "steps.jsonl").read_text().splitlines() if l.strip()]
        results = []
        for seg in segments_of(rows):
            p = seg["phase_index"]
            if rec["layout"] == "four_phase":  # stacking
                colour = rec["pick"] if p in (0, 1) else rec["target"]
                q = asp.block_prompts(colour)
                seg.update(molmo_prompt=q[0], fallback_prompt=q[1], moved_block=rec["pick"], target_block=rec.get("target"))
                res = {**asp.ground_segment(grounder, ep, rows, seg, args),
                       "target_method": f"{colour} block, first 25% of the segment"}
            elif p == 0:
                side = rec["drawer_side"]
                found = asp.locate_chicken_handles(grounder, ep, rows, seg, args)
                seg.update(molmo_prompt=asp.CHICKEN_SIDE_HANDLE.format(side=side), fallback_prompt=asp.CHICKEN_HANDLES,
                           drawer_side=side)
                if found:
                    res = asp.ground_segment(grounder, ep, rows, seg, args, asp.window(found[side], (-0.05, 0.05), (-0.06, 0.08)))
                    if res["needs_review"]:
                        res.update(point_pixel=found[side], needs_review=False)
                    res.update(drawer_handles=found, target_method=f"{side} handle (both handles located, sorted by x)")
                    handles[side].append(res["point_pixel"])
                else:
                    res = None  # filled from the other pre-open recoveries below
                    pending.append((ep, rows, seg))
            elif p == 1:
                side = rec["drawer_side"]
                seg.update(drawer_side=side)
                pool = handles.get(side, [])
                res = fixed(seg, np.median(pool, axis=0) if pool else None,
                            f"median {side} handle of the pre-open recoveries (n={len(pool)})", len(pool))
            else:  # 3 grasp / 4 pre-place
                key = "pre-grasp" if p == 3 else "pre-place"
                q = asp.CHICKEN_PHASES[key]
                seg.update(molmo_prompt=q[0], fallback_prompt=q[1])
                res = {**asp.ground_segment(grounder, ep, rows, seg, args),
                       "target_method": ("chicken leg" if p == 3 else "inside of the blue bin") + ", first 25% of the segment"}
            if res is not None:
                results.append(res)
        if len(results) == len(segments_of(rows)):
            failed += any(r["needs_review"] for r in results)
            for r in results:
                LOG.info("%s/%s phase %d %s: point=%s inliers=%d needs_review=%s", ep.parent.name[-6:], ep.name,
                         r["phase_index"], r["phase_prompt"], np.round(r["point_pixel"], 3).tolist(), r["inliers"], r["needs_review"])
            asp.write_episode(ep / "steps.jsonl", rows, results, args)
    for ep, rows, seg in pending:
        pool = handles.get(seg["drawer_side"], [])
        res = fixed(seg, np.median(pool, axis=0) if pool else None,
                    f"handles not located: median {seg['drawer_side']} handle of the other pre-open recoveries (n={len(pool)})", len(pool))
        failed += res["needs_review"]
        LOG.info("%s/%s phase 0 (fallback): point=%s", ep.parent.name[-6:], ep.name, np.round(res["point_pixel"], 3).tolist())
        asp.write_episode(ep / "steps.jsonl", rows, [res], args)
    LOG.info("Finished: %d episodes, %d need review", len(eps), failed)


if __name__ == "__main__":
    main()
