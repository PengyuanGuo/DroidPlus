#!/usr/bin/env python3
"""Fill crossword pre-grasp points that MolmoPoint could not confirm with the grasp point of the same cycle.

annotate_sorting_points.py --word rejects pre-grasp hits that are not near a cube found by the "all cubes" query
on the first frames; when that query misses the cube, the pre-grasp segment fails although the grasp segment of
the same cycle (same cube, gripper next to it) finds it. Both phases target the same cube, so the grasp point is
copied. Only segments with needs_review, a following grasp segment of the same letter that passed, are changed;
each change is recorded in 'corrections' (method copied_from_grasp) and steps_molmo.jsonl is updated.

Usage: python scripts/fix_crossword_pregrasp_from_grasp.py --dataset-root DATASET [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    fixed = left = 0
    for p in sorted(args.dataset_root.glob("*/episode_*/molmo_point_annotations.json")):
        ann = json.loads(p.read_text())
        segs = ann["segments"]
        changed = []
        for i, s in enumerate(segs):
            if not s["needs_review"]:
                continue
            g = segs[i + 1] if i + 1 < len(segs) else None
            if s["phase_index"] == 0 and g and g["phase_index"] == 1 and g.get("letter") == s.get("letter") \
                    and not g["needs_review"] and g["point_pixel"][0] >= 0:
                old = s["point_pixel"]
                s.update(point_pixel=list(g["point_pixel"]), needs_review=False, method="copied_from_grasp",
                         preview_frame=s["start_frame"])
                ann.setdefault("corrections", []).append({
                    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "segment_index": i,
                    "method": "copied_from_grasp", "previous_point_pixel": old, "new_point_pixel": s["point_pixel"]})
                changed.append(i)
            else:
                left += 1
                print(f"not fixable: {p.parent} segment {i} phase {s['phase_index']}")
        if not changed:
            continue
        fixed += len(changed)
        print(f"{p.parent.relative_to(args.dataset_root)}: segments {changed} <- grasp point")
        if args.dry_run:
            continue
        ann["needs_review"] = any(s["needs_review"] for s in segs)
        rows = [json.loads(l) for l in (p.parent / "steps_molmo.jsonl").read_text().splitlines() if l.strip()]
        for i in changed:
            for r in rows[segs[i]["start_frame"]:segs[i]["end_frame"]]:
                r["point_pixel"] = segs[i]["point_pixel"]
        (p.parent / "steps_molmo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        p.write_text(json.dumps(ann, indent=2) + "\n")
    print(f"fixed {fixed}, still needing review {left}")


if __name__ == "__main__":
    main()
