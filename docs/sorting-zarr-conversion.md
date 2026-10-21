# Convert annotated real sorting recordings to training zarr

After conversion, use the [validation and Rerun inspection tools](sorting-zarr-inspection.md)
to verify the stored arrays, compare source annotations, and review image/point overlays.

`scripts/convert_droidplus_to_zarr.py` reads the corrected
`steps_molmo.jsonl` and camera JPEGs directly. There is no intermediate LeRobot
export. Original recordings, images and annotation files are not modified.
`meta.json` is used for the instruction and nominal rate only; its stale frame
counts and old phase labels are not used.

## Run with uv

Install the extra dependencies in your existing annotation environment:

```bash
cd /home/atomic/Project/DroidPlus
uv pip install --python .venv-molmo/bin/python \
  'zarr>=2.12,<3' 'numcodecs>=0.10,<0.16'
```

Validate first (no zarr dependencies are needed for the dry run):

```bash
.venv-molmo/bin/python scripts/convert_droidplus_to_zarr.py \
  --dataset-root output_1017_phase_sorting \
  --dry-run
```

Convert all episodes:

```bash
.venv-molmo/bin/python scripts/convert_droidplus_to_zarr.py \
  --dataset-root output_1017_phase_sorting \
  --output data/droidplus_sorting_annotated.zarr
```

To test a subset, add `--max-episodes 1` and use a different output path. To
regenerate an existing output, add `--overwrite`: the previous zarr is renamed
to a timestamped `.backup-*` directory after the replacement finishes. Failed
conversions leave the previous output intact and retain a separate incomplete
staging directory for diagnosis.

## Crops and point coordinates

The output contains 224x224 RGB images. By default, the first episode's saved
square annotation ROI is reused; if its annotations use full camera images,
the converter selects the centered square from the first frame. It then fixes
this ROI for **all** episodes. Wrist images get their own fixed centered crop.

For the current recordings, both raw cameras are 672x376, giving:

```text
agent ROI: x=148, y=0, size=376
wrist ROI: x=148, y=0, size=376
```

Override explicitly with `--roi X Y SIZE` and, if needed,
`--wrist-roi X Y SIZE`. All point coordinates are mapped from the annotation's
full-image/cropped coordinates into the output agent crop:

```text
full-image annotation:
    new_x = (old_x * raw_width  - crop_x) / crop_size
    new_y = (old_y * raw_height - crop_y) / crop_size
```

The source annotation ROI is also accounted for when it is present. Targets
outside the new crop cause an error; they are not silently clipped or removed.
Resizing the square crop to 224x224 does not change its normalized points.
Reuse the output's recorded crop at deployment and convert deployment Molmo
points into that same coordinate system. The wrist image has no point label.

## Output layout

The store uses **zarr format v2**, Blosc/zstd level 5 with bit shuffle, and time
chunks of 100 frames for images (configurable with `--image-chunk`). It writes:

```text
droidplus_sorting_annotated.zarr/
├── data/
│   ├── agent_view         (N,224,224,3) uint8 RGB/HWC
│   ├── wrist_cam          (N,224,224,3) uint8 RGB/HWC
│   ├── state              (N,8) float32
│   ├── action             (N,8) float32
│   ├── prompt             (N,128) uint8 UTF-8
│   ├── phase_label        (N,) int8
│   ├── phase_prompt       (N,128) uint8 UTF-8
│   ├── point_pixel        (N,2) float32 normalized agent-crop xy
│   ├── timestamp          (N,) float64 seconds since each episode's first frame
│   ├── source_seq         (N,) int64 original frame ID
│   └── point_needs_review (N,) bool
├── meta/episode_ends      (E,) int64 cumulative exclusive ends
├── _cache/224x224/
│   ├── agent_view.zarr/images  (N,3,224,224) uint8 RGB/CHW
│   └── wrist_cam.zarr/images   (N,3,224,224) uint8 RGB/CHW
└── conversion_manifest.json
```

The `_cache` layout is required by the reference
`MolmoBotPnPColorPhaseDataset`; that loader accesses these arrays directly.
Source episodes stay separate so its episode-level train/validation partition
keeps all six phase segments from one demonstration together. Every `data/*`
array has the same N, and phase labels remain 0,1,0,1,0,1 within each episode.

Arm actions are the logged **absolute joint targets**. The last state/action
value is converted from native DROID's 0=open, 1=closed to the real-data guide's
**1=open, 0=closed** using `1 - native_value`. The reference simulation dataset
uses different raw gripper units (action 0/255); this converter deliberately
uses the real-data guide's convention. The deployment adapter must invert that
opening command when passing it to DROID's native gripper API.

The converter does not infer phases or resample time. Per-episode observed and
nominal rates are recorded separately. The current data's median observed rate
is about 6.96 Hz despite a nominal recording rate near 14.29 Hz. Set temporal
training/deployment parameters with the actual cadence in mind.

## Quality flags and provenance

Missing, nonfinite, out-of-range or sentinel `[-1,-1]` points are rejected.
Annotation-report segment boundaries and points must match the corrected JSONL.
Images must exist and decode, frame IDs must be consecutive and timestamps
strictly increasing. The source and report hashes are checked again before the
new zarr replaces the staging directory.

By default, valid points still marked `needs_review` are included with a warning
and their flags preserved in `data/point_needs_review` and the manifest. The
training loader **does not automatically exclude** those frames. Add
`--require-reviewed` to reject flagged episodes instead. Numeric validity does
not establish semantic correctness; finish visual review before training.

The manifest records each source episode, annotation hashes, crop, zarr slice,
per-phase review flags, and corrections' human-verification status. It also
notes stale collection metadata/raw-source hashes. Corrected `steps_molmo.jsonl`
is the data source; a later edit to raw `steps.jsonl` does not overwrite those
corrections or change the conversion's row counts.

## Point the reference config at the real dataset

In `/home/atomic/Project/atomic_policy/atomic_policy/config/molmospaces/train_atomic_molmospaces_pnp_color_2phase.yaml`,
set the dataset anchor's `zarr_path` before loading the YAML to:

```yaml
zarr_path: /home/atomic/Project/DroidPlus/data/droidplus_sorting_annotated.zarr
```

When using Hydra command-line overrides, override **both** concrete dataset
paths (YAML anchors are copies, not live references):

```text
task.grasp_dataset.zarr_path=/home/atomic/Project/DroidPlus/data/droidplus_sorting_annotated.zarr
task.place_dataset.zarr_path=/home/atomic/Project/DroidPlus/data/droidplus_sorting_annotated.zarr
```

The reference loader's shapes/cache are compatible, but its configured
`env_runner` performs MolmoSpaces simulation rollouts and expects simulation
semantics. A real-robot training/evaluation configuration must adapt that runner
and the deployment action adapter; changing the dataset path alone does not
make its simulated evaluation represent this real sorting task.
