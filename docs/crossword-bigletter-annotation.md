# Crossword big letters + recovery episodes: data prep, annotation, conversion (2026-11-07)

Recordings of 2026-11-06 (`crossword_mars_4phase_teleop_20261106_.zip`, 12 sessions, spacemouse ~14 Hz): the MARS
crossword with **bigger letter cubes and a new slot board** (a white strip with large printed numbers 1-5 instead
of the blue grid board of `spacemouse_crossword_mars_4phase`), plus **recovery episodes** that each record one
atomic phase starting from a failed state. Dataset: `data/annotated/spacemouse_crossword_bigletter_4phase`.
Training: atomic_policy `real_deploy/train_realworld.md`, "Crossword big-letter co-training".

## Episodes

| Kind | Episodes | Frames | Letters M/A/R/S |
|---|---|---|---|
| Full (`Bigletter_mars_4phase_*`), M,A,R,S -> slots 1-4, phases [0,1,2,3] x 4 | 48 | 79,362 | |
| Recovery pre-grasp (`recover_..._phase0_*`): offset start pose -> above the cube | 31 | 6,504 | 8/7/9/7 |
| Recovery grasp (`..._phase1_*`): staged misses, then grasp | 24 | 2,901 (of 6,212) | 6/6/7/5 |
| Recovery pre-place (`..._phase2_*`): cube held, offset -> above the slot | 20 | 3,147 | 5/5/5/5 |
| Recovery place (`..._phase3_*`): cube misplaced or held off-slot -> placed | 21 | 3,844 (of 4,540) | 5/6/5/5 |

Set aside in `data/annotated/_excluded_spacemouse_crossword_bigletter_4phase/` (`prep_report.json` lists why):
32 `valid: false` / < 30 frames, 7 full episodes with incomplete or mislabelled cycles.

## Fixes (`scripts/prep_crossword_bigletter.py`, originals kept as `*.orig`)

The recorder resets two things at the start of every episode, which is wrong for recovery episodes:

- **Phase label.** It starts in phase 0; the operator switched late (9-93 frames) or never. Every recovery episode is
  relabelled to the phase in its task name.
- **Gripper command.** It starts at 0 (open). Pre-place recoveries and 12 place recoveries start with the cube in the
  gripper (both fingertips on the cube in the first wrist frame), so their command is set to closed (1) until the
  operator's first change. Grasp recoveries start open (the recorded width reads 1.0 = open), unchanged. The other 9
  place recoveries (215652 episodes 017-026) start with the cube dropped on the board and the gripper open, unchanged.
- **Staged failures.** Grasp recoveries contain 1-3 deliberate misses. Only the last failure onward is kept: start 8
  frames before the reopen after the last failed close, so the data shows closed empty gripper -> open -> adjust ->
  grasp and never the approach to a staged miss. Place recoveries that start held and drop the cube on purpose before
  re-grasping start 8 frames after that drop.
- **Aborted take.** 212149/episode_004 (pre-grasp R) was re-recorded and its `steps.jsonl` still begins with 139
  rows of the aborted first take (seq restarts at 0; their images were overwritten). Rows before the last seq reset
  are dropped when the rest matches `meta.json` step_count (here 189); ee_trajectory is cut the same way.
- **Letter / slot.** Read by eye from the wrist frames for every recovery episode (cube between the fingers at the end
  of pre-grasp/grasp, in the gripper at the start of pre-place/place) and written to `meta.json` `recovery`
  (`recover_..._phase1_06` was recorded twice: episode_005 = M, episode_006 = A). Slot = position in MARS.

## Left-camera points (MolmoPoint, full frame)

    D=data/annotated/spacemouse_crossword_bigletter_4phase
    # full episodes, one session at a time (the annotator stops on non-cycle episodes)
    for s in 141343 144222 194230 203002; do
      python scripts/annotate_sorting_points.py --dataset-root $D/crossword_mars_4phase_teleop_20261106_$s \
          --atomic-policy-root ../atomic_policy --full-frame --word MARS --skip-existing; done
    python scripts/fix_crossword_pregrasp_from_grasp.py --dataset-root $D
    # recovery episodes
    python scripts/annotate_crossword_recovery.py --dataset-root $D --atomic-policy-root ../atomic_policy
    python ../atomic_policy/real_deploy/relabel_crossword_mars_prompts.py $D --word MARS

Targets are the same as for the MARS data: the letter cube for pre-grasp/grasp, the cube after release (in its slot)
for pre-place/place. Recovery specifics (`annotate_crossword_recovery.py`): pre-grasp / grasp ground the letter on the
first 25% of the episode; place grounds the placed cube on the last 15%; pre-place never releases, so it takes the
median placed-cube position of that slot from the place recoveries (same session, else all sessions; the board did
not move).

Checks: every recovery point was checked on full-resolution crops (right letter under the point for all 76
pre-grasp/grasp/place episodes, right slot for all 20 pre-place). Full episodes: 6 pre-grasp segments in session
144222 failed because the "all cubes" query of the first frames missed that cube (the rejected hit was often exactly
the cube); `fix_crossword_pregrasp_from_grasp.py` copies the grasp point of the same cycle (logged in
`corrections`; 7 segments, 6 in 144222 and 1 in 203002). All 192 full-episode letter points were then checked on
crops of the episode's first frame (cubes do not move before their grasp) and all 192 slot points on the end of
pre-place. Two letter points were on a neighbouring cube (144222/episode_005 M on the A, 203002/episode_009 S on an
O; the gripper lifts M / S and S ends in slot 4): re-grounded with MolmoPoint, accepting only hits near the lifted
cube (method `regrounded_near_grasped_cube`, pre-grasp and grasp). One slot-1 point (194230/episode_014, x = 143.5
px) lay just left of the shared crop and was set to the session's slot-1 median (155.7 px; method
`slot_median_of_session`).

## Conversion

Same crops as `droidplus_crossword_mars_robot_4phase_wrist.zarr`, so both can be trained together:

    bash ../atomic_policy/real_deploy/convert_zarr.sh $D data/convert_zarr/droidplus_crossword_bigletter_4phase.zarr 148 0 376

`convert_droidplus_to_zarr.py` accepts a single-phase episode when its `meta.json` has `recovery` (layout
`four_phase`). Wrist points are not annotated yet (the co-training uses the left point only); recovery episodes will
need an end-of-episode anchor for pre-grasp / pre-place, since they contain no gripper change to anchor on.
