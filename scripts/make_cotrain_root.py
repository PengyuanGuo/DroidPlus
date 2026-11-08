#!/usr/bin/env python3
"""Combine annotated datasets into one dataset root of symlinks (for converters that read a single --dataset-root,
e.g. atomic_policy real_deploy/droidplus_to_lerobot_droid.py for pi0.5 or droidplus_to_molmobot.py).

Session folders are real directories; every episode_* is a symlink to the episode in its source dataset, so
annotations are edited there, not here (same layout as spacemouse_crossword_mars_robot_4phase). SRC:K repeats a
source K times: its sessions appear K times (suffix _rep2, _rep3, ...), so its episodes are seen K times as often.

Usage:
  python scripts/make_cotrain_root.py --out data/annotated/spacemouse_drawer_chicken_6phase_recovery_cotrain \
      data/annotated/spacemouse_drawer_chicken_6phase data/annotated/spacemouse_drawer_chicken_6phase_recovery:2
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("sources", nargs="+", help="DATASET_ROOT[:REPEAT]")
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    lines, total = [], 0
    for spec in args.sources:
        path, _, rep = spec.partition(":")
        src, rep = Path(path).resolve(), int(rep or 1)
        eps = sorted(p.parent for p in src.glob("*/episode_*/steps_molmo.jsonl"))
        if not eps:
            raise SystemExit(f"no annotated episodes under {src}")
        for k in range(1, rep + 1):
            for ep in eps:
                session = args.out / (ep.parent.name + ("" if k == 1 else f"_rep{k}"))
                session.mkdir(exist_ok=True)
                (session / ep.name).symlink_to(ep.resolve())
        total += len(eps) * rep
        lines.append(f"- {src.name}: {len(eps)} episodes x{rep}")
    (args.out / "README.md").write_text(
        "Combined training set (scripts/make_cotrain_root.py). Session folders are real directories; every episode_* is a\n"
        "symlink to the episode in its source dataset, so annotations are edited there, not here. Sources:\n"
        + "\n".join(lines) + f"\nTotal {total} episode entries.\n")
    print(f"{args.out}: {total} episode entries\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
