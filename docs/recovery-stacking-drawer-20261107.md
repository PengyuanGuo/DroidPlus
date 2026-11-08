# Stacking and drawer-chicken recovery episodes (recorded 2026-11-07, prepared 2026-11-08)

Recordings: `spacemouse_stacking_4phase_recovery.zip`, `spacemouse_drawer_chicken_6phase_recovery.zip` (unzipped in
`data/staging_recovery_20261107/`). Each recording is one atomic phase (drawer: grasp + pre-place) that starts from a
failed state. Datasets:

| Dataset | Kind (task name) | Episodes | Frames kept | Start | Fix |
|---|---|---|---|---|---|
| `spacemouse_stacking_4phase_recovery` | grasp (`..._phase1_NN`) | 12 (yellow 4, red 3, blue 5) | 1,655 of 2,465 | open, next to the block | last staged miss onward (11 trimmed) |
| | stack on the 2nd layer (`..._phase3_NN`) | 15 | 2,562 | block in hand above the 2-block stack | command closed until the first change |
| `spacemouse_drawer_chicken_6phase_recovery` | pre-open (`..._phase0_NN`) | 20 (left 9, right 11) | 3,269 of 3,519 | 02, 09 closed; the rest open | 02, 09 closed start; 03-08 staged close away from the handle: last one onward |
| | open-and-home (`..._phase1_NN`) | 22 (left 10, right 12) | 3,130 | fingers hooked on the handle | command closed throughout |
| | grasp + pre-place (`..._phase34_NN`) | 10 | 2,532 of 4,089 | open above the chicken | last staged miss onward |

Excluded: `valid: false` retakes (stacking 29, drawer 27), `valid: null` attempts (drawer 162917/001, 163020/000),
empty / < 30 frames. `prep_report.json` in each dataset lists every episode.

## Fixes (`scripts/prep_recovery_20261107.py`, originals kept as `*.orig`)

- **Phase label.** The recorder starts in phase 0. Single-phase recordings become that phase throughout; grasp +
  pre-place recordings are grasp (3) up to the operator's switch to pre-place (4), 47-88 frames after the final close,
  as in the full data (median 57).
- **Gripper command.** The recorder resets it to 0 at every episode start; the gripper keeps its state. The recorded
  width can't tell (it carries the previous episode's value: pre-open 04 reads 1.0 with open fingers, 09 reads 0 with
  closed ones), so the start state was read by eye on the first wrist frame of every episode. Closed starts get 1 up to
  the operator's first change: all stack recoveries (2 of them read width 0), all open-and-home, pre-open 02 and 09.
- **Staged failures.** Kept from 8 frames before the reopen after the last failed close: a closed gripper on a
  misaligned block / the chicken tip -> open -> regrasp. 164827/episode_015 lifts the chicken before the staged drop,
  so it starts 8 frames after the drop (no "open while holding" frames).
- **Targets** (by eye): stacking colours on the wrist frames; the stack target is the middle block of the final
  3-block stack (left camera). Drawer side from the fingertip (end of pre-open, start of open-and-home) vs the full
  data's handle positions: 3-9 cm from one handle, >= 22 cm from the other. Prompts as in the full data:
  `grasp red block`, `stack red block on blue block on layer 2` (all stack recoveries are the second layer; layer
  prompts as in `relabel_stacking_prompts.py`), `pre-open the left drawer`, `open the left drawer and home`,
  `grasp chicken leg`, `pre-place chicken leg to the blue bin`.

## Left-camera points (`scripts/annotate_recovery_points.py`, MolmoPoint, full frame)

    python scripts/annotate_recovery_points.py --dataset-root data/annotated/spacemouse_stacking_4phase_recovery
    python scripts/annotate_recovery_points.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase_recovery

Stacking: the block to pick / the block it goes onto, first 25% of the episode. Drawer: pre-open = that side's handle
(both handles located, sorted by image x, as in the full data); open-and-home = the median handle point of that side
over the pre-open recoveries (the gripper covers the handle from frame 0; the full data copies the closed-handle point
too); grasp = chicken leg; pre-place = inside of the blue bin. Every point checked on zoomed crops: all correct but one,
pre-open 161052/episode_008 (hit the robot arm), set to the median left handle (logged in `corrections`). Left handles
at x 0.41, right 0.57 (full data 0.43 / 0.58).

## Wrist points (`annotate_wrist_points.py --place-drop 0.03`, MARS calibration)

    python scripts/annotate_wrist_points.py --dataset-root data/annotated/spacemouse_stacking_4phase_recovery --place-drop 0.03
    python scripts/annotate_wrist_points.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase_recovery \
        --place-drop 0.03 --place-z 0.08

Stacking: `recovery_windows` (grasp = last durable close, stack = last durable open, 3 cm lower). Drawer:
`drawer_chicken_windows` (pre-open = fingertip at the last frame, open-and-home = handle pulled with the fingertip),
extended for recoveries: a grasp without its pre-grasp = the close still held at the end; a pre-place without its
release = fingertip xy at the last frame at the full data's median bin target height (0.08 m). Stacking 100% visible;
drawer 88% (pre-open from offset start poses 29-100%, as in the full data's early pre-open). Checked on wrist frames.

## Conversion (same crops as the full zarrs, so they can be trained together)

    bash ../atomic_policy/real_deploy/convert_zarr.sh data/annotated/spacemouse_stacking_4phase_recovery \
        data/convert_zarr/droidplus_stacking_4phase_recovery_wrist.zarr 94 0 376
    bash ../atomic_policy/real_deploy/convert_zarr.sh data/annotated/spacemouse_drawer_chicken_6phase_recovery \
        data/convert_zarr/droidplus_multidrawer_chicken_6phase_recovery_wrist.zarr 148 0 376

`convert_droidplus_to_zarr.py` takes the phases and layout of a recovery episode from `meta.json` `recovery`.

## Review page and co-training roots

    python scripts/recovery_qa.py --dataset-root data/annotated/spacemouse_stacking_4phase_recovery
    python scripts/review_stacking.py --dataset-root data/annotated/spacemouse_stacking_4phase_recovery \
        --qa recovery_qa.json --title "Stacking recovery review" --port 18767        # drawer: --port 18768

The episode header shows the recording, kind, target, start state and the kept frames; reviews are saved to
`recovery_review.json`. pi0.5 reads one dataset root, so co-training sets are symlink folders:

    python scripts/make_cotrain_root.py --out data/annotated/spacemouse_stacking_4phase_recovery_cotrain \
        data/annotated/spacemouse_stacking_4phase data/annotated/spacemouse_stacking_4phase_recovery:2
