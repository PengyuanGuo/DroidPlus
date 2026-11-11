#!/usr/bin/env python3
"""Drop the ' on layer N' suffix from stacking prompts in an annotated dataset (user, 2026-11-11: train stacking
without layer prompts). Rewrites steps.jsonl / steps_molmo.jsonl / meta.json / molmo_point_annotations.json and their
*.pretrim copies (so scripts/trim_recovery_start.py reruns keep the plain prompts). Idempotent.

    python scripts/strip_layer_prompts.py data/annotated/spacemouse_stacking_4phase_recovery
"""
import json
import re
import sys
from pathlib import Path

PAT = re.compile(r" on layer \d+")
root = Path(sys.argv[1])
n = 0
for ep in sorted(p.parent for p in root.glob("*/episode_*/meta.json")):
    for name in ("steps.jsonl", "steps_molmo.jsonl", "meta.json", "molmo_point_annotations.json"):
        for p in (ep / name, ep / f"{name}.pretrim"):
            if not p.exists():
                continue
            t = p.read_text()
            new = PAT.sub("", t)
            if new != t:
                p.write_text(new)
                n += 1
print(f"{n} files changed")
