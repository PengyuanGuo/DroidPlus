# Drawer chicken search: annotation, review and conversion (2026-11-05)

Dataset `data/annotated/spacemouse_drawer_chicken_6phase`: two white drawer units side by side, a chicken leg hidden in
one of them, a blue bin in front. Spacemouse teleop at ~14 Hz. The operator opens the left drawer; if it is empty, the
right one as well; then picks the chicken leg and puts it in the bin. Phase labels are recorded during collection:

| phase_index | recorded | training prompt (`relabel_drawer_chicken_prompts.py`) | left point | wrist point |
|---|---|---|---|---|
| 0 | `pre-open` | `pre-open the {left/right} drawer` | that drawer's handle | fingertip at the handle (end of pre-open) |
| 1 | `open-and-home` | `open the {left/right} drawer and home` | same as phase 0 (copied) | the handle, pulled with the gripper, then left at the open position |
| 2 | `pre-grasp` | `pre-grasp chicken leg` | chicken leg | chicken leg (fingertip at the grasp) |
| 3 | `grasp` | `grasp chicken leg` | chicken leg | same, held after the close |
| 4 | `pre-place` | `pre-place chicken leg to the blue bin` | inside of the blue bin | bin: release point, 3 cm lower |
| 5 | `place` | `place chicken leg to the blue bin` | same | same |

Phases 0,1 repeat when the first drawer is empty: `[0,1,0,1,2,3,4,5]` (left, then right) or `[0,1,2,3,4,5]` (left only).

## Current stage

| Step | State |
|---|---|
| Episodes | 99: 50 from 2026-10-31 (6 sessions, from the A100 box) + 49 from 2026-11-04 (sessions 185346, 190812). 49 open both drawers, 50 the left only. 7 invalid 11-04 episodes (`valid: false`, stopped early) moved to `data/annotated/_invalid_drawer_chicken_20261104/`. 190812/episode_017 had one frame labelled `open-and-homne`: fixed, original in `steps.jsonl.typo_orig`. |
| Left points | `annotate_sorting_points.py --full-frame --skip-existing` (chicken mode, drawer side from MolmoPoint + gripper position). New episodes: 0 failed segments; 190812/episode_003 pre-open (left) hit the gripper (it hides the left drawer at frame 0): re-pointed from frame 30 with `annotate_sorting_points_correction.py`, copied to its open-and-home segment (backups in `molmo_correction_backups/`). |
| Prompts | atomic_policy `real_deploy/relabel_drawer_chicken_prompts.py` (rerun on all 99; safe to rerun). |
| Wrist points | `annotate_wrist_points.py --task drawer_chicken --place-drop 0.03`, MARS calibration (see `docs/wrist-camera-points.md`): 494 windows, 67,921 frames, 89% visible. |
| QA + review | `scripts/drawer_qa.py` (1 flag) and `scripts/review_stacking.py --qa drawer_qa.json` (port 18766). Reviewed by the user: points accepted as they are. |
| Conversion | zarr `data/convert_zarr/droidplus_multidrawer_chicken_6phase_wrist.zarr` (layout `chicken_six_phase`, agent ROI [148, 0, 376], wrist ROI centred); MolmoBot data `runs/molmobot_data/multidrawer_chicken_6phase`. |
| Training | not started (to run on an extra Vast instance); atomic_policy `real_deploy/train_realworld.md`. |

## Commands (from DroidPlus, `.venv-molmo`)

    D=data/annotated/spacemouse_drawer_chicken_6phase
    python scripts/annotate_sorting_points.py --dataset-root $D --atomic-policy-root ../atomic_policy --full-frame --skip-existing
    python ../atomic_policy/real_deploy/relabel_drawer_chicken_prompts.py $D
    python scripts/annotate_wrist_points.py --dataset-root $D --atomic-policy-root ../atomic_policy --task drawer_chicken --place-drop 0.03
    python scripts/drawer_qa.py --dataset-root $D
    python scripts/review_stacking.py --dataset-root $D --qa drawer_qa.json --title "Drawer label review" --port 18766

New recordings: unzip into `$D`, move `valid: false` episodes out, then run the same commands (`--skip-existing` keeps
the old left points; the relabel and wrist steps recompute every episode).

## Automatic flags (`scripts/drawer_qa.py`)

- left handle: pre-open point > 0.06 from the session's median handle point for that side
- left chicken / bin: orange / blue pixel share around the point (pre-grasp, pre-place) < 15%; chicken point nearer the
  other drawer's handle than the one opened last
- left support: fewer MolmoPoint inliers than required, needs_review
- wrist chicken: share of frames before the grasp whose wrist point lands on orange < 50%
- wrist windows: not 4 + 2 per extra drawer, pull shorter than 10 frames, visibility < 40% (handle, chicken) / 80% (rest)

On the 99 episodes: left orange share >= 0.8 for 95% of pre-grasp points, bin 100%; handle shift p99 3.4 cm;
wrist chicken hit 100% at the 5th percentile. One flag: 190812/episode_003 right handle (0.09 off, 0/2 inliers),
accepted on review.
