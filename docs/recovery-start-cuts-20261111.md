# Recovery episodes cut at the failure (2026-11-11)

Recovery recordings showed the operator driving into the failure (approach to a staged miss, carrying the cube to a
wrong slot, lowering the block off the stack) before the recovery. Behaviour cloning learns those frames too, so every
recovery episode now starts at the failed state: failure -> success only.

| Dataset | Episodes cut | Start frames |
|---|---|---|
| `spacemouse_crossword_bigletter_4phase` (recovery part) | 72 of 96 (pre-grasp 26, grasp 20, pre-place 18, place 8); the rest keep the prep cut | `scripts/recovery_starts/crossword_bigletter.json`, picked by eye |
| `spacemouse_stacking_4phase_recovery` | 27 (grasp 12, stack 15); 2 keep all frames (start 0) | `scripts/recovery_starts/stacking.json`, rule proposals checked and edited in the review page |
| `spacemouse_drawer_chicken_6phase_recovery` (2026-11-12) | 51 of 52 picked in the review page (pre-open 20, open-and-home 22, grasp + pre-place 9); 164827/020 not picked, keeps the prep cut | `scripts/recovery_starts/drawer_chicken.json` |

Start frames are raw image numbers (`wrist/000000100.jpg` -> 100), so they hold whether or not the prep step already
cut the episode.

    python scripts/trim_recovery_start.py scripts/recovery_starts/crossword_bigletter.json
    python scripts/trim_recovery_start.py scripts/recovery_starts/stacking.json
    python scripts/strip_layer_prompts.py data/annotated/spacemouse_stacking_4phase_recovery   # plain stacking prompts

`trim_recovery_start.py` keeps the files as they were before the first cut as `*.pretrim` and always starts from them.
A start earlier than the prep cut restores the raw rows (`steps.jsonl.orig`, labelled like the episode) and recomputes
the wrist labels with the same rule as `annotate_wrist_points.py` (checked: 0 px difference on untouched episodes).
Left points and wrist targets do not depend on the start, so later starts only shift frame indices.

## Review page

    python scripts/recovery_cut_ui.py --dataset-root data/annotated/spacemouse_stacking_4phase_recovery \
        --starts scripts/recovery_starts/stacking.json --port 18769

Both cameras over the whole raw recording, gripper command, arm height, the prep cut, the proposal and the saved start.
Proposals, from the crossword picks:

- grasp: the last failed close + 20 frames, capped at the reopen (user's crossword picks: close + 17..21 in 16 of 20);
- held stack / place: the first frame the held object is down at release height (bad placement reached);
- place with a staged drop: 15 frames before the re-grasp close.

Pre-grasp / pre-place picks (hovering over the wrong cube or slot, 5-12 cm from the end pose) are not predicted well
from the arm path (16 of 44 within 15 frames), so those stay manual.

Drawer chicken-search proposals (`drawer_proposal`, layout `chicken_six_phase`):

- grasp + pre-place: a missed grasp = failed close + 20 frames (capped at the reopen); a staged drop (the chicken
  rose > 3 cm in the hand before the reopen) = 15 frames before the re-grasp; flagged when the search before the
  re-grasp is > 150 frames;
- pre-open with a staged close away from the handle: close + 20, capped 5 frames before the reopen (short closes);
- other pre-open / open-and-home: the recording starts at the offset or mis-hooked pose (the failed state), so all
  frames are kept; flagged when the arm reaches a second bad pose and backs out (>= 3 cm).

The user moved most pre-open and open-and-home starts later (to where the arm leaves the start pose). A start before
the prep cut restores raw rows and recomputes the drawer wrist labels (`drawer_chicken_windows`, place z 0.08 m;
checked: 0 px difference on the 10 grasp episodes). Backup before the cut: box 2
`runs/backup_drawer_recovery_pretrim_20261111`.

    python scripts/recovery_cut_ui.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase_recovery \
        --starts scripts/recovery_starts/drawer_chicken.json --port 18770
    python scripts/trim_recovery_start.py scripts/recovery_starts/drawer_chicken.json

## Training (2026-11-11)

Stacking is trained with plain prompts ("stack red block on blue block"), no "on layer N"; the LeRobot converter takes
`--strip-layer` for the main stacking data (`PI05_CONVERT_ARGS=--strip-layer`).
