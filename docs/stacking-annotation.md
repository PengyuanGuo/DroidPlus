# Stacking: annotation, review and conversion (2026-11-05)

Dataset `data/annotated/spacemouse_stacking_4phase`: blue / red / yellow blocks, spacemouse teleop at ~14 Hz, recorded
2026-11-02 in 7 sessions. The bottom block stays where it starts; two blocks are stacked on it, so every episode has
2 cycles x 4 phases (pre-grasp / grasp / pre-place / place). The order is in `meta.json` `instruction`
("... in the order of red, yellow, blue from bottom to top").

## Current stage

| Step | State |
|---|---|
| Episodes | 171 complete; 6 orders, 25-31 each (blue<red<yellow 31, blue<yellow<red 30, red<blue<yellow 29, yellow<red<blue 29, red<yellow<blue 27, yellow<blue<red 25) |
| Left points | Done: `annotate_sorting_points.py` stacking mode (order from the instruction; "Point to the {colour} block": grasp phases -> the block to move, place phases -> the block it goes onto, constrained to the stack). Every preview reviewed, ~57 fixed. |
| Prompts | Done: atomic_policy `real_deploy/relabel_stacking_prompts.py`: `pre-grasp red block`, `grasp red block`, `pre-stack red block on blue block`, `stack red block on blue block`. Adds `stack_order` / `stack_variant` to meta.json; checked against the instruction for all 171. |
| Wrist points | Done: `annotate_wrist_points.py --place-drop 0.03` with the MARS calibration (wrist camera position confirmed). 98% visible, 683 windows. |
| QA + review | `scripts/stacking_qa.py` flags 31 episodes; `scripts/review_stacking.py` page; user reviewed: labels good. |
| Conversion | zarr `data/convert_zarr/droidplus_stacking_4phase_wrist.zarr` (agent ROI [94, 0, 376], wrist ROI [148, 0, 376]); MolmoBot data `runs/molmobot_data/stacking_4phase`. |
| Training | dp on GPU 0, MolmoBot on GPU 1 (atomic_policy `real_deploy/launch_stacking.sh`, `train_realworld.md`). |

## Wrist place target

The place target is the top of the target block: the fingertip centre at release, lowered 3 cm. Blocks are 5.2 cm tall
(the release height rises 5.3 cm for the first stack and 5.2 cm for the second); drops of 2-5 cm all project onto the
target block's top within a few px, so the exact value hardly matters. 172549/episode_080 closes the gripper again
after its last release and keeps it closed, so that episode has no 4th (place) window.

## Automatic flags (`scripts/stacking_qa.py`)

    python scripts/stacking_qa.py --dataset-root DATASET     # -> DATASET/stacking_qa.json

- left colour: share of block-coloured pixels around each pre-grasp / pre-place point (all pass)
- left target: cycle-2 stack point vs cycle-1 stack point in the image (> 0.09)
- left support: fewer MolmoPoint inliers than required
- wrist colour, grasp targets only: share of early-window frames whose wrist point lands on the block colour (< 50%).
  Not used for place targets: early in pre-place the target block hides behind the carried block.
- wrist target: 2nd stack's 3D target vs the 1st (> 2.5 cm sideways, or not 3.5-7 cm higher)
- not 4 wrist windows, or a window visible on < 80% of its frames

## Review page (`scripts/review_stacking.py`)

    python scripts/review_stacking.py --dataset-root DATASET --port 18765    # open http://localhost:18765

Episode list (flagged / all / not reviewed / needs fix, filter by order, `j`/`k`), frame player (left | wrist, space),
the 8 left points on their preview frames (cyan), the 4 wrist targets at start / middle / before contact (orange while
held). OK / Needs fix per point; in fix mode a click proposes a new left point. Saved to `DATASET/stacking_review.json`;
the annotation files are not changed. To apply proposed points: restore `*.prompt_orig`, apply with
`annotate_sorting_points_correction.py`, rerun `relabel_stacking_prompts.py` and `annotate_wrist_points.py`.
