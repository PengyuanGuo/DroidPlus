# ROBOT crossword: filtering and annotation (2026-11-04)

Dataset `data/annotated/spacemouse_crossword_robot_4phase`: spell ROBOT, letter k -> slot k of the blue grid board,
5 letters x 4 phases (pre-grasp / grasp / pre-place / place), spacemouse teleop at 15 Hz, collected 2026-11-03 in
4 sessions (200548, 202621, 220032, 231543).

## Current stage

| Step | State |
|---|---|
| Filter | Done: 67 recorded -> 50 kept (one complete episode per task 01-50). 17 aborted restarts moved to `spacemouse_crossword_robot_4phase_excluded/` with reasons in `filter_log.json`. |
| Phase fixes | Done: 202621/005 frames 790-803 grasp -> pre-grasp; 220032/008 letter-R boundaries aligned to the arm motion (`steps.jsonl.orig` kept). Re-grasps inside place (202621/015, 220032/008) are kept as place. |
| Letter points (left) | Done: MolmoPoint, then 35 hand fixes in 30 episodes (mostly the second O, where MolmoPoint points at the first O already in slot 2). |
| Slot points (left) | Done: hand-clicked once per board position (MolmoPoint failed on 33% of slots). |
| Prompts | Done: `relabel_crossword_mars_prompts.py --word ROBOT` -> `pre-grasp letter R`, `place to slot 5`, ... (same vocabulary as MARS). |
| Wrist points | Done: FK projection with the MARS wrist calibration (14 px median on ROBOT grasp tracks), `--place-drop 0.03`; 98% visible; 7 flagged windows checked by eye. See [wrist-camera-points.md](wrist-camera-points.md). |
| Combined with MARS | `spacemouse_crossword_mars_robot_4phase/`: per-session folders of episode symlinks (MARS 55 + ROBOT 50). Edit annotations in the source folders, never through the symlinks. |
| Training | dp + MolmoBot on MARS+ROBOT running on box 2; π0.5 (atomic, per-phase prompts) to train on the old A100 box. Details in atomic_policy `real_deploy/train_realworld.md`. |

## Static slot points

The board does not move within a stretch of episodes, so slots are clicked once per board position and copied into
every pre-place / place segment (`scripts/annotate_static_slots.py`, `.venv-molmo`, no GPU):

    python scripts/annotate_static_slots.py click --dataset-root DATASET --slots 5 --port 29876
    python scripts/annotate_static_slots.py apply --dataset-root DATASET [--dry-run]

Points are saved in `DATASET/static_slot_points.json` (`sessions` + `groups`; a group overrides its session for the
listed episodes). For ROBOT the board moved, so the sets are:

| Set | Episodes |
|---|---|
| A | 200548/* and 202621/003-022 |
| own | 202621/025 (board slid mid-episode) |
| B | 220032/001-006 |
| C | 220032/007-029 |
| own | 231543/* |

Check the board position (blue grid in the first and last frame of each episode) before trusting one set per
session: within a set it stays within a few px, between sets it shifts up to ~35 px (about one cell).

## Letter review

    python scripts/annotate_static_slots.py letters --dataset-root DATASET --port 18765   # -> letter_edits.json
    python scripts/annotate_static_slots.py apply-letters --dataset-root DATASET [--dry-run]

Shows all 5 letter points on each episode's first frame (failed ones marked ?); drag or click to fix, then apply
writes the edits to the pre-grasp and grasp segments.

## Order matters

Apply slot / letter corrections **before** `relabel_crossword_mars_prompts.py`: the correction tool checks the
steps.jsonl hash and refuses a file that changed since the preview. If prompts were already relabelled, restore the
`*.prompt_orig` files, apply, then relabel again.

## Serving the pages through Cursor

Cursor Remote-SSH forwards ports it sees; an `ssh -L` to the same port then fails with "Address already in use".
Serve on a port Cursor already forwards and open `http://localhost:<port>` directly.
