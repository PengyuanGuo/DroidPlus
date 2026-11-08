#!/usr/bin/env python3
"""QA file for the review page (scripts/review_stacking.py) on recovery datasets (meta.json 'recovery', written by
scripts/prep_recovery_20261107.py): one entry per episode with its phase points, wrist targets and the prep facts
(recording, start state, trim). Flags: a hand-corrected left point, a wrist target visible on < 40% of its frames.

    python scripts/recovery_qa.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase_recovery
    python scripts/review_stacking.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase_recovery \
        --qa recovery_qa.json --title "Drawer recovery review" --port 18768
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KIND = {"stacking_4phase_recovery_phase1": "grasp recovery", "stacking_4phase_recovery_phase3": "stack recovery (layer 2)",
        "recovery_drawer_chicken_phase0": "pre-open recovery", "recovery_drawer_chicken_phase1": "open-and-home recovery",
        "recovery_drawer_chicken_phase34": "grasp + pre-place recovery"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for ann_path in sorted(args.dataset_root.glob("*/episode_*/molmo_point_annotations.json")):
        ep = ann_path.parent
        rec = json.loads((ep / "meta.json").read_text())["recovery"]
        ann = json.loads(ann_path.read_text())
        corrected = {c["segment"] for c in ann.get("corrections", [])}
        segs = []
        for i, s in enumerate(ann["segments"]):
            flags = ["left point corrected by hand: " + s.get("target_method", "")] if i in corrected else []
            segs.append({"i": i, "phase": s["phase_index"], "prompt": s["phase_prompt"], "colour": None,
                         "start": s["start_frame"], "end": s["end_frame"],
                         "preview_frame": min(max(s.get("preview_frame", s["start_frame"]), s["start_frame"]), s["end_frame"] - 1),
                         "point": s["point_pixel"], "colour_share": None, "inliers": s.get("inliers"),
                         "required": s.get("required_inliers"), "copied_from": None, "flags": flags,
                         "handle_note": s.get("target_method")})
        wins = []
        for k, w in enumerate(ann.get("wrist", {}).get("windows", [])):
            n = max(1, w["end_frame"] - w["start_frame"])
            vis = w["visible_frames"] / n
            wins.append({"k": k, "role": w["role"], "segments": w["segments"], "start": w["start_frame"], "end": w["end_frame"],
                         "contact": w["contact_frame"], "release": w["release_frame"], "colour": None, "colour_hit": None,
                         "visible": vis, "flags": ["wrist target visible on < 40% of frames"] if vis < 0.4 else [],
                         "target_base_m": w["target_base_m"]})
        kind = KIND.get(re.sub(r"_\d+$", "", rec["recording"]), rec["recording"])
        start = "starts closed (command fixed to 1 until the first change)" if rec["start_held"] else "starts open"
        trim = (f"kept from frame {rec['trim_start']} of {rec['frames_orig']} ({rec['trim_reason']})" if rec["trim_start"]
                else f"all {rec['frames_orig']} frames kept")
        target = ", ".join(f"{k} {rec[k]}" for k in ("pick", "target", "drawer_side") if rec.get(k))
        out[str(ep.relative_to(args.dataset_root))] = {
            "segments": segs, "wrist_windows": wins, "frames": rec["frames"], "variant": kind,
            "flags": [f for s in segs for f in s["flags"]] + [f for w in wins for f in w["flags"]],
            "header": f"{rec['recording']} · {kind}{' · ' + target if target else ''} · {start} · {trim}"}
    (args.dataset_root / "recovery_qa.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} episodes, {sum(bool(v['flags']) for v in out.values())} flagged -> {args.dataset_root / 'recovery_qa.json'}")


if __name__ == "__main__":
    main()
