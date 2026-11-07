#!/usr/bin/env python3
"""Left-camera MolmoPoint points for single-phase crossword recovery episodes (meta.json 'recovery', written by
scripts/prep_crossword_bigletter.py). annotate_sorting_points.py only accepts full letter cycles, so this script
builds the one segment itself and reuses its grounding / writing code.

Targets (same definitions as the full episodes):
  phase 0 pre-grasp, 1 grasp  letter cube, MolmoPoint on the first 25% of the episode
  phase 3 place               the cube after release: MolmoPoint on the last 15% of the episode
  phase 2 pre-place           no release inside the episode, so the slot's placed-cube position is taken from the
                              place recoveries: median per slot in the same session, else over all sessions
Writes steps_molmo.jsonl, molmo_point_annotations.json and molmo_previews/ like annotate_sorting_points.py; segments
carry letter / grid_cell for the prompt relabel.

Usage (DroidPlus, .venv-molmo):
  python scripts/annotate_crossword_recovery.py --dataset-root data/annotated/spacemouse_crossword_bigletter_4phase \
      --atomic-policy-root ../atomic_policy [--device cuda:0] [--skip-existing]
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
PHASES = ["pre-grasp", "grasp", "pre-place", "place"]


def load(ep: Path):
    rows = [json.loads(l) for l in (ep / "steps.jsonl").read_text().splitlines() if l.strip()]
    rec = json.loads((ep / "meta.json").read_text())["recovery"]
    primary, fallback = asp.letter_prompts(rec["letter"])
    seg = {"phase_index": rec["phase"], "phase_prompt": PHASES[rec["phase"]], "start_frame": 0, "end_frame": len(rows),
           "molmo_prompt": primary, "fallback_prompt": fallback, "letter": rec["letter"], "grid_cell": rec["slot"],
           "recovery": True}
    return rows, seg, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--atomic-policy-root", type=Path, default=Path("../atomic_policy"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint", default="allenai/MolmoPoint-8B")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = argparse.Namespace(roi=None, full_frame=True, checkpoint=a.checkpoint, device=a.device, dtype="bfloat16",
                              load_in_4bit=False, max_image_size=448, samples=5, skip_first_frames=2,
                              min_valid_samples=3, overwrite=True, word="MARS")
    eps = sorted(p.parent for p in a.dataset_root.glob("*/episode_*/meta.json")
                 if "recovery" in json.loads(p.read_text()))
    if a.skip_existing:
        eps = [e for e in eps if not (e / "steps_molmo.jsonl").is_file()]
    LOG.info("%d recovery episodes", len(eps))
    import torch
    sys.path.insert(0, str(a.atomic_policy_root.resolve()))
    from atomic_policy.vlm.grounder_molmo import MolmoPointGrounder
    grounder = MolmoPointGrounder(checkpoint=a.checkpoint, device=a.device, dtype=torch.bfloat16, load_in_4bit=False,
                                  max_image_size=448)
    by_phase = collections.defaultdict(list)
    for ep in eps:
        by_phase[json.loads((ep / "meta.json").read_text())["recovery"]["phase"]].append(ep)
    slot_points = collections.defaultdict(list)  # (session, slot) -> placed-cube points
    # existing place results count too (for --skip-existing reruns)
    for p in a.dataset_root.glob("*/episode_*/molmo_point_annotations.json"):
        s = json.loads(p.read_text())["segments"][0]
        if s.get("recovery") and s["phase_index"] == 3 and not s["needs_review"]:
            slot_points[(p.parent.parent.name, s["grid_cell"])].append(s["point_pixel"])
    failed = 0
    for phase in (3, 0, 1, 2):
        for ep in by_phase.get(phase, []):
            rows, seg, rec = load(ep)
            n = len(rows)
            if phase == 3:
                frames = np.linspace(int(n * 0.85), n - 1, 8, dtype=int).tolist()
                res = asp.ground_segment(grounder, ep, rows, seg, args, samples=sorted(set(frames)))
                res["target_method"] = "placed cube after release (end of episode)"
                if not res["needs_review"]:
                    slot_points[(ep.parent.name, rec["slot"])].append(res["point_pixel"])
            elif phase == 2:
                same = slot_points.get((ep.parent.name, rec["slot"]))
                pool = same or [q for (s, k), v in slot_points.items() if k == rec["slot"] for q in v]
                ok = bool(pool)
                res = {**seg, "sample_indices": [0], "samples": [], "valid_predictions": 0, "inliers": len(pool),
                       "required_inliers": 1, "point_pixel": np.median(pool, axis=0).tolist() if ok else [-1.0, -1.0],
                       "needs_review": not ok, "preview_frame": n - 1,
                       "target_method": f"slot {rec['slot']} placed-cube median of place recoveries "
                                        f"({'same session' if same else 'all sessions'}, n={len(pool)})"}
            else:
                res = asp.ground_segment(grounder, ep, rows, seg, args)
                res["target_method"] = "letter cube, first 25% of the episode"
            failed += res["needs_review"]
            LOG.info("%s/%s phase %d %s slot %d: point=%s inliers=%d needs_review=%s", ep.parent.name[-6:], ep.name,
                     phase, rec["letter"], rec["slot"], np.round(res["point_pixel"], 3).tolist(), res["inliers"],
                     res["needs_review"])
            asp.write_episode(ep / "steps.jsonl", rows, [res], args)
    LOG.info("Finished: %d episodes, %d need review", len(eps), failed)


if __name__ == "__main__":
    main()
