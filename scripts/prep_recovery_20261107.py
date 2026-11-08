#!/usr/bin/env python3
"""Build the stacking and drawer-chicken recovery datasets from the 2026-11-07 recordings.

    data/annotated/spacemouse_stacking_4phase_recovery        stacking_4phase_recovery_phase{1,3}_NN
    data/annotated/spacemouse_drawer_chicken_6phase_recovery  recovery_drawer_chicken_phase{0,1,34}_NN

Each recording is one atomic phase (drawer: grasp + pre-place) that starts from a failed state. Kept: meta
valid == true and >= 30 frames (the operator marked retakes valid: false; valid: null = unmarked attempts, dropped).

Fixes (originals kept as *.orig):
  * phase label: the recorder starts in phase 0 and the operator switched late or not at all. Single-phase
    recordings become that phase throughout; drawer phase34 recordings are grasp (3) up to the operator's switch to
    pre-place (4) and pre-place after (47-88 frames after the final close, as in the full data, median 57).
  * gripper command: the recorder resets it to 0 (open) at every episode start while the gripper keeps its state.
    Episodes that start closed (read by eye from the first wrist frame; the recorded width is stale, it carries the
    previous episode's value) get 1 up to the operator's first change: all stack recoveries (block in hand), all
    open-and-home recoveries (fingers hooked on the handle, closed throughout) and pre-open 02 / 09.
  * staged failures: grasp recoveries (stacking grasp, drawer grasp) and pre-open 03-08 (close away from the handle)
    contain staged failures. Only the last failure onward is kept: start 8 frames before the reopen after the last
    failed close (closed gripper on a misaligned block / chicken tip -> open -> regrasp). Drawer 164827/episode_015
    lifts the chicken before the staged drop, so it starts 8 frames after the drop instead.
  * stack prompts carry the layer, as relabel_stacking_prompts.py (commit 62d2cbe): every stack recovery puts the
    third block on the 2-block stack -> "stack {pick} block on {target} block on layer 2".
  * targets (by eye, checked on wrist / left frames 2026-11-08): stacking block colours, stack target = the middle
    block of the final 3-block stack; drawer side from the fingertip vs the full data's handle positions (3-9 cm from
    one handle, >= 22 cm from the other).
Written to meta.json 'recovery' (phases, layout, prompts, targets, trim) and to phase_prompt in steps.jsonl.

Usage: python scripts/prep_recovery_20261107.py --staging data/staging_recovery_20261107 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

MIN_FRAMES = 30
LEAD = 8

STACK_GRASP = {**{n: "yellow" for n in (1, 2, 3, 4)}, **{n: "red" for n in (5, 6, 7)},
               **{n: "blue" for n in (8, 9, 10, 11, 12)}}
# stack recovery number -> (carried block, block it goes onto = middle of the final stack)
STACK_PLACE = {1: ("blue", "yellow"), 2: ("blue", "yellow"), 3: ("blue", "yellow"), 4: ("blue", "red"),
               5: ("red", "blue"), 6: ("blue", "red"), 7: ("red", "blue"), 8: ("red", "blue"),
               9: ("red", "yellow"), 10: ("red", "yellow"), 11: ("yellow", "red"), 12: ("yellow", "red"),
               13: ("yellow", "blue"), 14: ("yellow", "blue"), 15: ("yellow", "blue")}
PREOPEN_SIDE = {n: "left" if n <= 9 else "right" for n in range(1, 21)}
PREOPEN_HELD = {2, 9}
# open-and-home: two recordings carry number 10 (left, then right) and two 15 (both right): side per episode
HOME_RIGHT_FROM = ("163143", "episode_011")
DROP_INSTEAD = {("164827", "episode_015")}  # lifted before the staged drop: start after the drop


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def flips(cmd: list[int]) -> list[tuple[int, int]]:
    return [(i, cmd[i]) for i in range(1, len(cmd)) if cmd[i] != cmd[i - 1]]


def last_failure_start(cmd: list[int], key: tuple[str, str]) -> tuple[int, str | None]:
    f = flips(cmd)
    closes = [i for i, c in f if c == 1]
    if len(closes) < 2:
        return 0, None
    reopen = next(i for i, c in f if c == 0 and i > closes[-2])
    if key in DROP_INSTEAD:
        return reopen + LEAD, f"staged drop of the lifted object at {reopen}"
    return max(closes[-2] + 1, reopen - LEAD), f"last staged failure: closed {closes[-2]}, reopened {reopen}"


def plan(task: str, session: str, ep: str, rows: list[dict]) -> dict:
    """Phase labels, prompts, held start, trim for one recording."""
    n = len(rows)
    m = re.match(r"stacking_4phase_recovery_phase(\d)_(\d+)$", task)
    if m:
        phase, num = int(m[1]), int(m[2])
        if phase == 1:
            c = STACK_GRASP[num]
            return {"dataset": "stacking", "phases": [1] * n, "prompts": {1: f"grasp {c} block"}, "held": False,
                    "targets": {"pick": c}, "trim": True, "layout": "four_phase"}
        c, t = STACK_PLACE[num]
        return {"dataset": "stacking", "phases": [3] * n, "prompts": {3: f"stack {c} block on {t} block on layer 2"},
                "held": True, "targets": {"pick": c, "target": t}, "trim": False, "layout": "four_phase"}
    m = re.match(r"recovery_drawer_chicken_phase(\d+)_(\d+)$", task)
    if not m:
        raise ValueError(f"unknown task {task}")
    kind, num = m[1], int(m[2])
    if kind == "0":
        s = PREOPEN_SIDE[num]
        return {"dataset": "drawer", "phases": [0] * n, "prompts": {0: f"pre-open the {s} drawer"},
                "held": num in PREOPEN_HELD, "targets": {"drawer_side": s}, "trim": True, "layout": "chicken_six_phase"}
    if kind == "1":
        s = "right" if (session, ep) >= HOME_RIGHT_FROM else "left"
        return {"dataset": "drawer", "phases": [1] * n, "prompts": {1: f"open the {s} drawer and home"}, "held": True,
                "targets": {"drawer_side": s}, "trim": False, "layout": "chicken_six_phase"}
    if kind == "34":
        switch = next(i for i, r in enumerate(rows) if r["phase_index"] == 4)
        return {"dataset": "drawer", "phases": [3] * switch + [4] * (n - switch),
                "prompts": {3: "grasp chicken leg", 4: "pre-place chicken leg to the blue bin"}, "held": False,
                "targets": {"object": "chicken leg", "place": "blue bin"}, "trim": True, "layout": "chicken_six_phase"}
    raise ValueError(f"unknown drawer recovery kind {kind}")


def process(src: Path, dst: Path, dry: bool) -> dict:
    session, ep = src.parent.name.split("_")[-1], src.name
    meta = json.loads((src / "meta.json").read_text())
    rows = read_jsonl(src / "steps.jsonl")
    p = plan(meta["task"], session, ep, rows)
    cmd = [1 if r["action"]["positions"][7] > 0.5 else 0 for r in rows]
    fixed_until = 0
    if p["held"]:
        f = flips(cmd)
        fixed_until = f[0][0] if f else len(cmd)  # first real change (to 1 = a no-op on a closed gripper)
        cmd[:fixed_until] = [1] * fixed_until
    start, why = last_failure_start(cmd, (session, ep)) if p["trim"] else (0, None)
    phases = p["phases"][start:]
    labels = [ph for i, ph in enumerate(phases) if i == 0 or ph != phases[i - 1]]
    info = {"phases": labels, "phase": labels[0] if len(labels) == 1 else None, "layout": p["layout"],
            "prompts": {str(k): v for k, v in p["prompts"].items()}, **p["targets"],
            "recording": meta["task"], "start_held": p["held"], "gripper_closed_until": fixed_until,
            "trim_start": start, "trim_reason": why, "frames_orig": len(rows), "frames": len(rows) - start}
    if dry:
        return info
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for name in ("steps.jsonl", "ee_trajectory.json", "meta.json"):
        if (dst / name).is_file() and not (dst / f"{name}.orig").exists():
            shutil.copy2(dst / name, dst / f"{name}.orig")
    out = []
    for i in range(start, len(rows)):
        r = dict(rows[i])
        r["action"] = {**r["action"], "positions": list(r["action"]["positions"])}
        r["action"]["positions"][7] = float(cmd[i])
        r["phase_index"] = p["phases"][i]
        r["phase_prompt"] = p["prompts"][p["phases"][i]]
        out.append(r)
    (dst / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    if (dst / "ee_trajectory.json.orig").is_file():
        ee = json.loads((dst / "ee_trajectory.json.orig").read_text())
        for k, v in ee.items():
            if isinstance(v, list) and len(v) == len(rows):
                ee[k] = v[start:]
        (dst / "ee_trajectory.json").write_text(json.dumps(ee))
    meta = json.loads((dst / "meta.json.orig").read_text())
    meta.update(step_count=len(out), recovery=info)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--annotated", type=Path, default=Path("data/annotated"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    outs = {"stacking": args.annotated / "spacemouse_stacking_4phase_recovery",
            "drawer": args.annotated / "spacemouse_drawer_chicken_6phase_recovery"}
    report = {k: {"kept": [], "excluded": []} for k in outs}
    for src in sorted(args.staging.glob("*/*/episode_*")):
        meta = json.loads((src / "meta.json").read_text())
        key = "stacking" if src.parts[-3].startswith("spacemouse_stacking") else "drawer"
        rel = Path(src.parent.name) / src.name
        n = len(read_jsonl(src / "steps.jsonl"))
        reason = None
        if meta.get("valid") is not True:
            reason = f"valid: {meta.get('valid')}"
        elif n < MIN_FRAMES:
            reason = f"{n} frames"
        if reason:
            report[key]["excluded"].append({"episode": str(rel), "task": meta.get("task"), "reason": reason})
            continue
        info = process(src, outs[key] / rel, args.dry_run)
        report[key]["kept"].append({"episode": str(rel), **info})
    for key, r in report.items():
        kinds = {}
        for e in r["kept"]:
            k = re.sub(r"_\d+$", "", e["recording"])
            d = kinds.setdefault(k, {"n": 0, "held": 0, "trimmed": 0, "frames": 0, "frames_orig": 0})
            d["n"] += 1; d["held"] += e["start_held"]; d["trimmed"] += e["trim_start"] > 0
            d["frames"] += e["frames"]; d["frames_orig"] += e["frames_orig"]
        print(f"{key}: kept {len(r['kept'])}, excluded {len(r['excluded'])}")
        for k, d in kinds.items():
            print(f"  {k}: {d}")
        if not args.dry_run:
            outs[key].mkdir(parents=True, exist_ok=True)
            (outs[key] / "prep_report.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
