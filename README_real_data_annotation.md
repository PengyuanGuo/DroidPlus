# Annotating DROID Plus (LeRobot) data for atomic training

Guide for adapting `panda_tabletop_phase_annotator.py` (same folder) to real-robot data
collected with DROID Plus. The annotator was written for the simulated PandaTabletop task;
most of its logic transfers, but some parts must be adapted. Each section marks what is
**[existing]** in the annotator and what is **[adapt]** / **[new]**.

Dependency: `atomic_policy/vlm/grounder_molmo.py` (MolmoPoint wrapper).

For the real sorting recordings in `output_1017_phase_sorting`, use
[`scripts/annotate_sorting_points.py`](scripts/annotate_sorting_points.py) to
annotate JPEG images directly before zarr conversion. See the
[setup and commands](docs/sorting-point-annotation.md).

---
nda_tabletop_10hz_annotated.zar
## 1. Target format (what training expects)

The sim reference dataset (`panda_tabletop_10hz_annotated.zarr`: 2000 episodes, 1.28 M frames,
10 Hz) is not needed — this section is the full spec.

**Container**
- zarr **format v2** (training uses zarr-python 2.12; zarr-python ≥ 3 writes v3 by default →
  pass `zarr_format=2`), directory store, two groups: `data/` and `meta/`.
- All episodes are concatenated in order along axis 0; every `data/*` array has the same length N.
- `meta/episode_ends[i]` = exclusive end index of episode i (cumulative length); episode 0
  starts at 0. Example: lengths 600, 650 → `[600, 1250]`.
- Compression: blosc (zstd, clevel 5, bit-shuffle). Chunk along time only (e.g. 100 frames for
  images, ~1000 for low-dim arrays); avoid one-frame chunks (one file per chunk).

**Arrays** (frame t = one synchronized timestep)

| Key | Shape / dtype | Content |
|---|---|---|
| `data/agent_view` | (N, 224, 224, 3) uint8 | front camera, **RGB, HWC**; fixed camera, fixed square ROI then resize (Section 2). `point_pixel` refers to this image |
| `data/wrist_cam` | (N, 224, 224, 3) uint8 | wrist camera, RGB, HWC |
| `data/state` | (N, 8) float32 | observed joint positions [rad] (7) + gripper opening (1). **Gripper = Franka Hand width / 0.08 m → ≈1 open, ≈0 closed** (in the sim, the same hand reads ~0.15 on a thin handle, 0.5–0.7 on a 5 cm cube) |
| `data/action` | (N, 8) float32 | **absolute** target joint positions [rad] for step t (≈ `state[t+1]` joints; sim: mean error 0.004 rad) + gripper cmd in [0, 1], **0 = close, 1 = open** (mostly binary) |
| `data/prompt` | (N, 128) uint8 | episode task string, UTF-8, truncated / zero-padded to 128 bytes; same for all frames of an episode |
| `meta/episode_ends` | (E,) int64 | see above |
| `data/phase_label` | (N,) int8 | phase index, 0-based — from the operator keypress column (Section 4) |
| `data/phase_prompt` | (N, 128) uint8 | phase text, same encoding as `prompt` — *added by annotator* |
| `data/point_pixel` | (N, 2) float32 | (x, y) normalized to [0, 1] on `agent_view` (x = col / width, y = row / height, origin top-left); `-1, -1` for no-point phases — *added by annotator* |

The first six keys come from conversion (Section 2), the last three from annotation (Sections 4–5).
Storing images at 224×224 means the training loader uses them directly (other sizes trigger an
extra resized copy under `_cache/`). Extra keys are ignored by training.

**Sanity checks after conversion:** all `data/*` lengths equal `episode_ends[-1]`; images look
right (RGB, not BGR); `state[:, 7]` ≈ 1 when the hand is open; `action[t, :7]` tracks
`state[t+1, :7]`; gripper cmd flips 1 → 0 at each grasp.

---

## 2. LeRobot → zarr conversion **[new]**

**Implemented native-data path:** for the corrected JSONL/JPEG sorting data, use
[`scripts/convert_droidplus_to_zarr.py`](scripts/convert_droidplus_to_zarr.py).
It reads `steps_molmo.jsonl`, transforms full-image points into the saved square
crop, writes zarr v2, and also writes the CHW image cache required by the
MolmoSpaces two-phase dataset loader. See
[conversion commands and conventions](docs/sorting-zarr-conversion.md).
The LeRobot steps below describe the alternative Parquet/video input route.

1. Read `meta/info.json`: `fps`, `codebase_version` (v2.1 = one parquet per episode, v3.0 =
   merged files; layouts differ) and `features` (find the front camera, wrist camera,
   joint positions, gripper and action keys).
2. Use `pyarrow` + `PyAV` directly (the `lerobot` package is not needed and needs py ≥ 3.10).
   Per episode, sort rows by `frame_index`, decode the camera mp4s and check that the frame
   count matches the number of rows.
3. Images: crop the front camera to a **fixed square ROI** covering the workspace (record it;
   the same ROI is used at deployment), then resize to 224×224. Do not squash 16:9 to square.
   Store only 224×224 frames; keep the original mp4s as the archive (full-res frames in zarr
   blow up file counts).
4. Fill `state`, `action`, `prompt` per Section 1; copy the `phase_index` column into
   `data/phase_label` (Section 4); write `meta/episode_ends`.
5. Keep the native fps (DROID is typically 15 Hz — verify); do not resample. Training-config
   `horizon` / `n_action_steps` are in steps, not seconds.

---

## 3. What DROID Plus data is missing

| Missing | Fix |
|---|---|
| `phase_index` column (operator keypress during teleop) | implemented in teleop CLI/UI and LeRobot export — Section 4 |/home/atomic/Project/atomic_policy/outputs/sorting_20261021_180802/checkpoints/latest_epoch471.ckpt
| `phase_prompt` | lookup from `phase_label` — Section 6 |
| `point_pixel` | Section 5 |
| Absolute joint-target actions | check what the `action` column holds (velocities won't work); use logged absolute targets if present, else `action[t, :7] = joint_pos[t+1]` |
| Gripper on the sim scale | `state[:, 7] = franka_hand_width / 0.08`; gripper cmd → 0 (close) / 1 (open) |
| Object / container names | prefer logged fields; otherwise parse the task string |

Franka Hand note: the sim uses the same hand, so the `width / 0.08` scale carries over
directly. It is commanded near-binary, but a full close can take up to ~1 s (depends on the
commanded speed), so the gripper reading lags the operator's intent — one more reason to take
phase boundaries from the keypress (Section 4) rather than from the gripper signal.

---

## 4. Phase labels

Phases are labelled **by the operator during teleoperation**, not inferred afterwards.

### Collection side **[implemented in this repository]**

Both `scripts/run_teleop_cli.py` and `scripts/run_teleop_ui.py` now record operator
phase labels. For example, from the repository root with your camera environment
variables exported:

```bash
python scripts/run_teleop_cli.py --gripper franka --task "your_task"
# Or use the browser UI:
python scripts/run_teleop_ui.py --gripper franka --task "your_task"
```

- Each episode starts at **phase 0**. Keys **0–9 set the matching zero-based ID**:
  press `1` to begin phase 1, `2` for phase 2, etc. Repeating the current ID is harmless.
  Choose the task's phase definitions before collection.
- CLI: focus the terminal; SPACE starts recording, ESC ends it.
- Web UI: click phase buttons or use number keys outside input fields. The displayed
  phase confirms the selection. Terminal number keys also work while an episode runs.
  The API `POST /episode/phase` accepts `{"phase_index": 10}` for IDs through 127.
- Pause about 0.5 s at each visually verifiable boundary, select the phase, confirm
  it, then continue. This is an operator pause; selecting a phase does not stop the robot.
- Every saved `steps.jsonl` row contains integer `phase_index`. A phase selection is
  sampled before camera acquisition and applies to that recorded frame and later
  frames until changed. A selection received during acquisition applies on the next step.
- `meta.json` contains `phase_annotation`: segment lengths, issues, and
  `needs_review`. Finalization prints warnings for a nonzero first phase,
  skipped/backward transitions, empty recordings, and segments shorter than 0.5 s
  at the configured recording cadence. Review flagged episodes before training.
  The code does not know your task's final phase, so also check that all expected
  phases were completed.
- The control loop rounds recording intervals to whole control steps:
  100 Hz control / requested 15 Hz records every 7 steps (nominal 14.286 Hz).
  Metadata stores that nominal rate plus `requested_record_rate_hz`; wall timestamps
  remain available for checking actual cadence.

Export normally; the LeRobot v2.1 Parquet files and `meta/info.json` preserve
`phase_index` as int64:

```bash
python scripts/export_lerobot.py output/<run> --out-dir output/lerobot/<run>
```

Old episodes with no labels export as `phase_index = -1`, never as phase 0.
Partially labelled episodes raise an export error. The standalone steps-to-Parquet
export also preserves the column (null for old unlabelled rows). During zarr
conversion copy the column to `data/phase_label` after rejecting unlabelled data.

Collection keeps DROID's native gripper convention: **0=open, 1=closed** for both
the observed fraction and command. The converter must use
`state[:, 7] = 1 - observation.state[:, 7]` and
`action[:, 7] = 1 - action[:, 7]` for the Franka Hand to meet Section 1.
The arm action already contains absolute commanded joint targets.
The native-data annotation and conversion scripts are linked above; the
Parquet/video conversion route remains separate.

The collection protocol is:

- The teleop loop keeps a `current_phase` integer, starting at 0. A keypress (`1`, `2`, `3`, …)
  sets it to that phase id; the operator presses the next number as each phase begins.
- Every recorded frame stores it as an extra LeRobot column, e.g. `phase_index` (int64) —
  one value per frame, no separate event log.
- Operator protocol: pause ~0.5 s at each boundary, press, then continue. This removes keypress
  reaction lag, and matches deployment, where the robot pauses at each phase start for grounding.
- Define boundaries by **visually verifiable** events ("gripper closed on the handle", "object
  lifted off the table"): at deployment a VLM checker must judge the same transition from images.

### Annotator side **[adapt]**

- `data/phase_label` = the `phase_index` column, copied frame by frame. No inference needed.
- **Delete** the gripper-segmentation block — `_detect_gripper_transitions`, `_raw_boundaries`,
  `_extend_to_grasp_complete`, `_remap_boundaries`, `_segment_episode`, and the
  `alpha` / `beta` / `gamma` arguments. It existed only because the sim had no operator input.
- Keep `PHASE_NAMES` / `POINT_PHASES` as the per-task phase definition.
- Validate each episode: ids start at 0, increase by 1, none skipped, each phase longer than a
  plausible minimum. Flag failures for review rather than accepting them silently — operators
  do mis-press.
- Optional: where a boundary coincides with a gripper open/close, snap it to that event, which
  is more precise than a human keypress.

---

## 5. Point data

### Method **[existing]** — `_annotate_points_for_episode`

One point per phase, written to **every frame of that phase** (training only reads the first
frame of each sample window, and deployment grounds once per phase):

1. Sample 5 frames from the phase.
2. Run MolmoPoint with `molmo_prompt`; out-of-range or failed → retry with `fallback_prompt`.
3. If > 2 points: drop outliers (> 2σ from the median), then average.
4. Write to all frames of the phase; no-point phases stay `-1`. Failure stats are logged per phase.

### Adaptation **[adapt]**

- **Sampling window:** the current code spreads samples over the whole phase. Use frames spread
  over the **first ~25%** (skip the first few, motion blur): less arm occlusion, and it matches
  deployment (grounding at phase start). Do not use 5 consecutive frames.
- **Image fed to Molmo:** the code reads `data/agent_view` from the zarr (224 px). Instead decode
  the high-res **ROI** frame from the mp4 (~448 px, as the sim runner's `vlm_render_size`) and
  normalize by ROI size. Use the same ROI and resolution at deployment.
- **Precision:** MolmoPoint loads as fp32 by default (`dtype='auto'`); bf16 halves memory
  (~20 GB vs ~38 GB). Use the **same dtype for annotation and deployment**.
- Keep the failure statistics — they are the per-phase grounding reliability.

---

## 6. `molmo_prompt` and `phase_prompt`

Both are per-phase lookup tables **[existing structure, adapt content]**:
`_get_phase_prompt`, `_get_molmo_prompt`, `_get_molmo_fallback_prompt`.

| | `phase_prompt` | `molmo_prompt` |
|---|---|---|
| Example | `pick up and lift the red cup` | `Point to the red cup` |
| Stored in zarr | yes | no (annotation only) |
| Used by | language-conditioned baselines; reuse as checker description | MolmoPoint |

- Return `None` from `_get_molmo_prompt` for no-point phases.
- Fallback prompts should be **broader** than the primary (e.g. `Point to the handle` vs a
  detailed description).
- Object labels: replace `_parse_object_label` (a tabletop-specific regex) with logged object
  names when available.
- The point policies (`PointUnetImagePolicy`) do not use `phase_prompt`; it is still worth
  storing.

---

## 7. Summary: reuse vs. change

| Part | Status |
|---|---|
| `_encode_prompt` | reuse as is |
| Prompt tables, `_parse_object_label` | **adapt** to the task |
| Gripper segmentation (`_detect_gripper_transitions` … `_segment_episode`, `alpha/beta/gamma`) | **delete** — replaced by the operator keypress label |
| Per-frame `phase_index`, sequence validation and review flags | implemented in collection/export; copy to `phase_label` during conversion |
| Point sampling / voting / outlier rejection | reuse; **change** sampling window and image source |
| Zarr read/write loop (`annotate_zarr`) | reuse; read from the converted zarr |
| LeRobot → zarr converter | **new** |
