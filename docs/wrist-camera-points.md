# Wrist-camera target points

Per-frame target points in the wrist camera, for training (dp / MolmoBot) and for real-time use during deployment.
They complement the left-camera points (`point_pixel`, one per phase, from `annotate_sorting_points.py`).

## Why projection, not tracking

The left camera is static, so one point per phase is enough. The wrist camera moves, so the target moves in its
image on every frame. SmartData (phone on a handheld gripper) and UMI_underwater (ROV) re-find the object
visually on every frame (SAM 2 / Grounded-SAM-2). On our recordings that fails often:

- **Cubes:** streaming SAM 2 alone was more than 25 px off on 11–12% of frames. It jumps to look-alike cubes and
  to the fingers.
- **Slots:** it can't follow a grid cell at all; it segments the whole board.

The Franka, unlike a phone or an ROV, knows where its wrist camera is on every frame (forward kinematics of the
recorded joint angles), and each target stays still until it is touched. So the point is computed:

    wrist point(t) = project( 3D target , FK(q_t) )

The 3D target comes from the contact, in hindsight. A target window is made of consecutive phases and ends with
the first durable gripper-command change (it then holds for at least 2 s):

- **Close** gives an object target (cube, chicken leg, drawer handle). It sits at the fingertip centre at the
  close, rides with the gripper while held, and stays where it is released.
- **Open** gives a place target (slot, bin): the fingertip centre at release, lowered by `--place-drop` (3 cm for
  the crossword slots, which matches where MolmoPoint points: the number printed on the board).

So crossword pre-grasp + grasp map to the letter cube, and pre-place + place to the slot. Occlusion by the
fingers or the held cube doesn't matter, and a held object is exactly at the fingertip pixel.

## Pipeline

Run from DroidPlus with `.venv-molmo` (SAM 2 is installed there: `/workspace/pengyuan/third_party/sam2`,
checkpoints in `/workspace/pengyuan/checkpoints/sam2`). The geometry code lives in
`atomic_policy/atomic_policy/wrist/` (pass `--atomic-policy-root`).

```bash
DATASET=data/annotated/test_spacemouse_crossword_mars_4phase_wrist_cam
# 1. calibrate the wrist camera from the data (no board): SAM 2 large tracks each grasped cube backward from the
#    grasp moment; ~40 min for 17 episodes on an A100, tracks cached in $DATASET/wrist_calibration_tracks.json
python scripts/calibrate_wrist_camera.py --dataset-root $DATASET --atomic-policy-root ../atomic_policy --per-session 2
# 2. per-frame labels (CPU, seconds); QA against the cached SAM 2 tracks; preview videos
python scripts/annotate_wrist_points.py --dataset-root $DATASET --atomic-policy-root ../atomic_policy \
    --place-drop 0.03 --sam2-check --previews 3
# 3. zarr for dp (wrist points are picked up automatically)
python scripts/convert_droidplus_to_zarr.py --dataset-root $DATASET --output data/convert_zarr/<name>.zarr
```

Calibration assumptions:

- the grasped cube sits between the fingertips at a fixed wrist pixel (`--seed-pixel 0.62 0.45` on this rig);
- the fingertip centre lies on the flange axis;
- the fingertip distance is fixed (`--tcp-z 0.15`). With `xyz_yaw` spacemouse teleop the gripper only rotates
  about that axis, so camera and fingertip offsets along it can't be separated from the data. Labels and
  deployment are unaffected; absolute heights shift with the choice.

Recalibrate if the wrist camera mount is touched. The per-session residuals in the output show a moved camera.

## Formats

| Where | Field | Meaning |
|---|---|---|
| `steps_molmo.jsonl` rows | `wrist_point_pixel` | `[x, y]` normalized on the raw 672×376 wrist image; `[-1, -1]` when not visible |
| | `wrist_point_visible` | target in front of the camera and inside the image |
| | `wrist_point_held` | target object in the gripper |
| `molmo_point_annotations.json` | `wrist` | calibration path + sha256, `place_drop_m`, per target window: phases, role, contact / release frames, 3D target (base frame), `sam2_check_median_px`, `needs_review` |
| zarr `data/` | `wrist_point_pixel` (N, 2) float32 | normalized on the **wrist_cam crop** (centred square by default), `-1` when not visible or outside the crop |
| | `wrist_point_visible` (N,) bool | |
| MolmoBot h5 | `obs/extra/object_image_points/<target>/wrist_camera/{points, num_points}` | per frame, full wrist frame, `num_points` 0 when not visible (atomic_policy `real_deploy/droidplus_to_molmobot.py`) |

The first annotation run keeps `steps_molmo.jsonl.wrist_orig`; rerunning is safe (labels are recomputed).

## Training (atomic_policy)

dp: set `dataset.wrist_point_key: wrist_point_pixel` and `point_policy.point_input_dim: 5`. Point-conditioned
policies then get `[left x, left y, wrist x, wrist y, wrist visible]`:

- the left point is from the window start, as before;
- the wrist point is from the current (last) observation frame, exactly what the deployment tracker reports;
- not-visible frames give wrist `(0, 0, 0)`;
- training-only augmentation: `dataset.wrist_point_jitter` (e.g. 0.02) and `dataset.wrist_point_dropout`
  (e.g. 0.15). The latter reports "not visible" so the policy copes when deployment has no wrist point.

MolmoBot: `--use_point_prompts --point_prompt_camera wrist_camera`; the prompt then marks the target in the
phase's first (conditioning) wrist frame.

## Deployment (atomic_policy `deploy/wrist.py`, `wrist/tracker.py`)

Deploy config:

```yaml
wrist:
  calibration: /path/wrist_camera_calibration.json   # also gives the target height per phase
  sam2_model: null          # kinematic only (recommended, see results); 'tiny' enables SAM 2 refinement
  reground_every: 30        # steps between asynchronous wrist MolmoPoint calls
  reground_gate_m: 0.04
```

At each phase start the runner already pauses the robot for left-camera MolmoPoint. In the same pause it calls
MolmoPoint on the wrist image, then:

1. lifts the pixel along its ray to the phase's target height, giving the 3D target;
2. every control step, projects it through FK of the current joints (about 0.1 ms; can't drift; works through
   occlusion);
3. every 30 steps, asks MolmoPoint again asynchronously (about 0.6 s; the robot keeps moving). Each answer is
   lifted with the pose at capture time and joins a median consensus (answers more than 4 cm off are outvoted,
   lifts over 0.6 m from the camera are rejected);
4. once the robot commands the gripper closed on an object target, snaps the point to the fingertip, as in the
   labels;
5. hands the policy `[left, wrist (crop-normalized), visible]`, and logs every point (`WRIST POINT` / `GROUND WRIST`).

Offline test on recorded episodes:

```bash
python -m atomic_policy.wrist.replay --dataset-root $DATASET --calibration $DATASET/wrist_camera_calibration.json \
    --out runs/wrist_replay/v2 --episodes <episode dirs> --videos 3
python -m atomic_policy.wrist.sweep_place_drop ...   # choose the place-target height from cached MolmoPoint answers
```

## Results on the crossword test set

Test set: 42 episodes, 10 sessions.

**Calibration:** 68 grasps from 17 episodes (8,791 frames), median reprojection error 8.0 px (p90 20.5 px). A
session held out of the fit came in at 11.5 px. The SAM 2 tracks being fitted also contain tracking errors.

**Labels:** 65,944 frames; the target is visible in the full wrist image on 97% of them. Against independent SAM 2
tracks of the 68 cubes the median difference is 8 px. The 7 windows flagged for review were checked by eye, and in
those SAM 2 was wrong (it drifted to a neighbouring cube), not the label.

**Replay of the deployment tracker:** causal, MolmoPoint wrist seeds at phase starts, async re-grounds every 2 s applied 0.6 s
late, 7 episodes from 7 sessions not used in the calibration (`/workspace/pengyuan/runs/wrist_replay/v2`: summary JSON,
3 videos, `replay_error_cdf.png`, `replay_strip.jpg`).

| Method | Cube before grasp: median / p90 / >25 px | Held cube | Slot: median / p90 / >25 px |
|---|---|---|---|
| **Kinematic + MolmoPoint consensus (default)** | **11 / 19 px / 4%** | **0 px** | **13 / 58 px / 19%** |
| + SAM 2 tiny refinement (gated, anchored) | 10 / 20 px / 8% | 0 px | same (no SAM 2 on slots) |
| SAM 2 only, MolmoPoint re-detection (UMI-style) | 10 / 32 px / 11% | 9 px | 26 / 154 px / 51% |

Slot errors sit at the start of the window, when the slot is far away and MolmoPoint has answered once or not yet:
in the place phase the median is 12 px with 7% of frames over 25 px, and in the last quarter the p90 is 19 px.
MolmoPoint on the wrist image: 0.6 s per call (A100, bf16, 18.5 GB); cubes within 25 px on 98% of calls,
slots 72%. SAM 2 tiny streaming: 33 ms per frame (base_plus 42 ms).

## Full crossword set

`data/annotated/spacemouse_crossword_mars_4phase` (55 episodes, 84,282 frames) is labelled with the test-set calibration
(copied to its root): 98% visible, 440 target windows. The two sessions not in the test set (171405, 172050) were
checked by eye. zarr: `data/convert_zarr/droidplus_crossword_mars_4phase_wrist.zarr`; dp run
`runs/dp_crossword_mars_4phase_wrist` (atomic_policy `real_deploy/train_realworld.md`).

## Limits

- **Static targets until contact.** A cube pushed during the approach is labelled where it was grasped, i.e. at
  its final position.
- **Table-plane lift.** MolmoPoint pixels are lifted to the median target height of the phase (crossword: cubes
  and slots within 1 cm of the same height). Targets at varying heights need per-target heights.
- **Gripper command, not state.** Contacts come from the commanded gripper. The recorded gripper *state* is stale
  while closed in the spacemouse data, so don't use it.
- **Other tasks:** only crossword has been calibrated and replayed so far; stacking and drawer use its calibration.

## Drawer chicken search (`--task drawer_chicken`, 2026-11-05)

`data/annotated/spacemouse_drawer_chicken_6phase` (99 episodes: 50 from 2026-10-31, 49 from 2026-11-04) with the MARS
calibration (copied to its root, sha256 b1e20b782317):

    python scripts/annotate_wrist_points.py --dataset-root data/annotated/spacemouse_drawer_chicken_6phase \
        --atomic-policy-root ../atomic_policy --task drawer_chicken --place-drop 0.03 --previews 3

The generic close/open rule does not fit this task: the operator closes the gripper early in pre-open (16-45 cm before
the handle), hooks the handle with closed fingers, pulls, and only reopens at the start of pre-grasp.
`atomic_policy.wrist.targets.drawer_chicken_windows` sets the anchors explicitly:

| Phase | Target | Anchor |
|---|---|---|
| pre-open | handle | fingertip at the last pre-open frame (the arm is stopped at the handle) |
| open-and-home | handle (`pulled`) | slides with the fingertip along the pull (~15 cm along +y) until the largest horizontal travel before a 5 cm lift, then stays |
| pre-grasp + grasp | chicken leg | the close in grasp that still holds at its end (3 episodes miss a first grasp and regrasp) |
| pre-place + place | blue bin | first open in place, 3 cm below the fingertip |

Result: 494 windows, 67,921 frames, visible 89%. Not visible only early in pre-open (handle out of view from home,
68% of the phase visible) and early in pre-grasp (79%); every other phase 98-100%. Checked by eye on contact sheets of
two- and one-drawer episodes from both days. Deployment: the tracker does not know the `pulled` role yet.
Review page: `scripts/drawer_qa.py` (1 flag) + `scripts/review_stacking.py --qa drawer_qa.json --port 18766`; reviewed by the
user 2026-11-05: left and wrist points accepted as they are (pre-open stays on the visible handle bar).
