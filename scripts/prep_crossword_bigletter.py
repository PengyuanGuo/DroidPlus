#!/usr/bin/env python3
"""Build data/annotated/spacemouse_crossword_bigletter_4phase from the 2026-11-06 big-letter MARS recordings.

Full episodes (task Bigletter_mars_4phase_*): kept when valid is not False and the phases are exactly
[0,1,2,3] x 4 (M, A, R, S into slots 1-4). Others go to _excluded/.

Recovery episodes (task recover_Bigletter_mars_phase{N}_{k}): one atomic phase each, starting from a failed state.
  * phase label: the whole episode becomes phase N (the recorder starts in phase 0 and the operator often switched
    late or not at all);
  * letter / slot: from the visual check of the wrist frames (cube between the fingers), written to meta.json
    'recovery' (slot = position of the letter in MARS);
  * gripper command: the recorder resets it to open (0) at the start of every episode. Episodes that start with the
    cube in the gripper (all pre-place recoveries; place recoveries listed in PLACE_HELD) get closed (1) up to the
    operator's first real change;
  * deliberate misses: grasp recoveries contain several staged misses (close next to the cube, reopen). Only the last
    failed state onward is kept: start 8 frames before the reopen after the last failed close, so the data shows a
    closed, empty gripper -> open -> adjust -> grasp, and never the approach to a staged miss. Place recoveries that
    start held and drop the cube on purpose before re-grasping are cut the same way: start 8 frames after that drop.
Originals are kept next to the edited files as *.orig. Rerunning starts from the staging copy, so it is repeatable.

Usage: python scripts/prep_crossword_bigletter.py --staging data/staging_crossword_20261106 \
           --out data/annotated/spacemouse_crossword_bigletter_4phase [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

WORD = "MARS"
PHASE_NAMES = ["pre-grasp", "grasp", "pre-place", "place"]
MIN_FRAMES = 30
LEAD = 8  # frames kept before the reopen (grasp) / after the drop (place)

# letter per recovery number, from the wrist-frame sheets (checked by eye 2026-11-07)
LETTER_RANGES = {
    0: [(1, 8, "M"), (9, 15, "A"), (16, 24, "R"), (25, 31, "S")],
    1: [(1, 5, "M"), (7, 11, "A"), (12, 18, "R"), (19, 24, "S")],
    2: [(1, 5, "M"), (6, 10, "A"), (11, 15, "R"), (16, 20, "S")],
    3: [(1, 5, "M"), (6, 10, "A"), (11, 15, "R"), (16, 20, "S")],
}
# recover_..._phase1_06 was recorded twice: episode_005 is M, episode_006 is A
LETTER_OVERRIDE = {("213316", "episode_005"): "M", ("213316", "episode_006"): "A"}
# place recoveries that start with the cube in the gripper (both fingertips on the cube in the first wrist frame);
# the others (215652 episodes 017-026) start open with the cube misplaced on the board
PLACE_HELD = {("215652", f"episode_{i:03d}") for i in (28, 30, 32, 34, 36, 38, 40, 44, 46)} | {("221718", "*")}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def drop_stale_take(rows: list[dict], meta: dict) -> tuple[list[dict], int]:
    """A re-recorded episode can keep the rows of an aborted earlier take in front (seq restarts at 0; their images
    were overwritten). Keep the rows after the last seq reset when that matches meta step_count."""
    resets = [i for i in range(1, len(rows)) if rows[i]["seq"] != rows[i - 1]["seq"] + 1]
    if resets and len(rows) - resets[-1] == meta.get("step_count"):
        return rows[resets[-1]:], resets[-1]
    if resets:
        raise ValueError(f"seq resets at {resets} that do not match step_count {meta.get('step_count')}")
    return rows, 0


def flips(cmd: list[int]) -> list[tuple[int, int]]:
    return [(i, cmd[i]) for i in range(1, len(cmd)) if cmd[i] != cmd[i - 1]]


def letter_for(phase: int, num: int, session: str, ep: str) -> str:
    if (session, ep) in LETTER_OVERRIDE:
        return LETTER_OVERRIDE[(session, ep)]
    for lo, hi, letter in LETTER_RANGES[phase]:
        if lo <= num <= hi:
            return letter
    raise ValueError(f"no letter for phase {phase} #{num} ({session}/{ep})")


def process_recovery(src: Path, dst: Path, phase: int, num: int, dry: bool) -> dict:
    session = src.parent.name.split("_")[-1]
    rows, stale = drop_stale_take(read_jsonl(src / "steps.jsonl"), json.loads((src / "meta.json").read_text()))
    letter = letter_for(phase, num, session, src.name)
    cmd = [1 if r["action"]["positions"][7] > 0.5 else 0 for r in rows]
    held = phase == 2 or (phase == 3 and ((session, src.name) in PLACE_HELD or (session, "*") in PLACE_HELD))
    fixed_until = 0
    if held:
        f = flips(cmd)
        fixed_until = f[0][0] if f else len(cmd)  # first real change (to 1: a no-op on a closed gripper)
        cmd[:fixed_until] = [1] * fixed_until
    start, cut_reason = 0, None
    f = flips(cmd)
    if phase == 1:
        closes = [i for i, c in f if c == 1]
        if len(closes) >= 2:
            reopen = next(i for i, c in f if c == 0 and i > closes[-2])
            start, cut_reason = max(closes[-2] + 1, reopen - LEAD), f"last staged miss: closed {closes[-2]}, reopened {reopen}"
    elif phase == 3 and held:
        opens = [i for i, c in f if c == 0]
        if len(opens) >= 2:  # drop on purpose, re-grasp, final release
            start, cut_reason = opens[-2] + LEAD, f"staged drop at {opens[-2]}"
    info = {"phase": phase, "phase_name": PHASE_NAMES[phase], "letter": letter, "slot": WORD.index(letter) + 1,
            "recording": f"recover_Bigletter_mars_phase{phase}_{num:02d}", "start_held": held,
            "gripper_closed_until": fixed_until, "trim_start": start, "trim_reason": cut_reason,
            "frames_orig": len(rows), "frames": len(rows) - start, "stale_rows_dropped": stale}
    if dry:
        return info
    shutil.copytree(src, dst, dirs_exist_ok=True)
    out = []
    for i in range(start, len(rows)):
        r = dict(rows[i])
        r["action"] = {**r["action"], "positions": list(r["action"]["positions"])}
        r["action"]["positions"][7] = float(cmd[i])
        r["phase_index"], r["phase_prompt"] = phase, PHASE_NAMES[phase]
        out.append(r)
    for name in ("steps.jsonl", "ee_trajectory.json", "meta.json"):
        if (dst / name).is_file() and not (dst / f"{name}.orig").exists():
            shutil.copy2(dst / name, dst / f"{name}.orig")
    (dst / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    ee = json.loads((dst / "ee_trajectory.json.orig").read_text())
    for k, v in ee.items():
        if isinstance(v, list) and len(v) in (len(rows), len(rows) + stale):
            ee[k] = v[len(v) - len(rows):][start:]
    (dst / "ee_trajectory.json").write_text(json.dumps(ee))
    meta = json.loads((dst / "meta.json.orig").read_text())
    meta.update(step_count=len(out), recovery=info, word=WORD)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    report = {"full": [], "recovery": [], "excluded": []}
    for src in sorted(args.staging.glob("*/episode_*")):
        meta = json.loads((src / "meta.json").read_text())
        rows, _ = drop_stale_take(read_jsonl(src / "steps.jsonl"), meta)
        rel = Path(src.parent.name) / src.name
        task = meta.get("task") or ""
        runs = [r["phase_index"] for i, r in enumerate(rows) if i == 0 or r["phase_index"] != rows[i - 1]["phase_index"]]
        m = re.match(r"recover_Bigletter_mars_phase(\d)_(\d+)$", task)
        reason = None
        if meta.get("valid") is False:
            reason = "valid: false"
        elif len(rows) < MIN_FRAMES:
            reason = f"{len(rows)} frames"
        elif not m and runs != [0, 1, 2, 3] * len(WORD):
            reason = f"phases {runs}"
        if reason:
            report["excluded"].append({"episode": str(rel), "task": task, "reason": reason})
            if not args.dry_run:
                shutil.copytree(src, args.out.parent / f"_excluded_{args.out.name}" / rel, dirs_exist_ok=True)
            continue
        if m:
            info = process_recovery(src, args.out / rel, int(m[1]), int(m[2]), args.dry_run)
            report["recovery"].append({"episode": str(rel), **info})
        else:
            report["full"].append({"episode": str(rel), "task": task, "frames": len(rows)})
            if not args.dry_run:
                shutil.copytree(src, args.out / rel, dirs_exist_ok=True)
                meta["word"] = WORD
                (args.out / rel / "meta.json").write_text(json.dumps(meta, indent=2))
    rec = report["recovery"]
    print(f"full {len(report['full'])}, recovery {len(rec)}, excluded {len(report['excluded'])}")
    for p in range(4):
        sub = [r for r in rec if r["phase"] == p]
        letters = {l: sum(r["letter"] == l for r in sub) for l in WORD}
        print(f"  recovery phase {p} ({PHASE_NAMES[p]}): {len(sub)} {letters}, start held {sum(r['start_held'] for r in sub)},"
              f" trimmed {sum(r['trim_start'] > 0 for r in sub)}, frames {sum(r['frames'] for r in sub)} of {sum(r['frames_orig'] for r in sub)}")
    for e in report["excluded"]:
        print("  excluded", e)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "prep_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
