# Check and visualize the sorting zarr

Use automated validation for array/data integrity, and Rerun for visual review
of targets, camera crops, phase timing and robot state/action traces. A numeric
point in [0,1] can still select the wrong object; passing structural checks is
not proof of correct pointing or good demonstrations.

## Separate visualization environment

The tested viewer uses Rerun SDK 0.40.0, which requires NumPy >=2. Keep it separate
from the training environment's NumPy 1.26/Numba dependencies:

```bash
cd /home/atomic/Project/DroidPlus
uv venv --python 3.11 .venv-viz
uv pip install --python .venv-viz/bin/python \
  rerun-sdk==0.40.0 zarr==2.18.7 numcodecs==0.15.2 opencv-python-headless
```

## Automated validation

```bash
.venv-viz/bin/python scripts/verify_droidplus_zarr.py \
  --zarr data/droidplus_sorting_annotated.zarr \
  --check-sources --full-images \
  --report reports/sorting_zarr_validation.json
```

The validator opens the zarr read-only and checks:

- Format v2, required arrays, dtypes/shapes, and matching frame counts.
- Exclusive cumulative episode ends and exactly six alternating phase segments.
- Finite state/action values, gripper range, valid points, UTF-8 text, ordered
  timestamps, consecutive source frame IDs and constant per-phase prompts/points.
- CHW cache pixels match the RGB/HWC images. `--full-images` reads every image
  and cache chunk; without it the validator samples episode ends plus first/mid
  frames for a faster check.
- With `--check-sources`, current source/report hashes match the conversion
  manifest; all low-dimensional arrays match values reconstructed from corrected
  JSONL (including crop-adjusted points and inverted gripper fractions).
- With `--check-sources`, first/last images from each camera of each episode
  match their source JPEGs after the recorded crop and RGB resize.

Exit code 0 means these checks passed; exit code 1 indicates errors. Existing
point-review flags are reported as warnings, not silently cleared or excluded.
If you correct points after conversion, source checks detect the stale zarr;
regenerate it with the converter before training.

## Rerun visualization

The main viewer entry point opens Rerun and also saves a timestamped recording:

```bash
.venv-viz/bin/python scripts/visualize_zarr.py \
  --zarr data/droidplus_sorting_annotated.zarr --episode 0
```

Use `--all-episodes` for the whole dataset. Use `--no-spawn` to save without
opening the GUI, and `--output reports/my_review.rrd` to choose a new output path.
This viewer replaces the former validation-only contents of `visualize_zarr.py`;
automated checks remain in `verify_droidplus_zarr.py`.

Export one episode:

```bash
.venv-viz/bin/python scripts/visualize_droidplus_zarr_rerun.py \
  --zarr data/droidplus_sorting_annotated.zarr \
  --episode 0 \
  --output reports/sorting_episode000_review.rrd

.venv-viz/bin/rerun reports/sorting_episode000_review.rrd
```

`--episode` is the **global zarr episode index** in the manifest, not the suffix
of an `episode_XXX` source directory. Episode 0 is
`teleop_20261017_174600/episode_000`. Source paths are displayed in the viewer.

The recording includes the exact CHW image caches consumed by training,
transposed to RGB for display:

- Agent view with point overlaid in pixels (`normalized_xy * [width,height]`).
- Synchronized wrist view without a point overlay.
- Phase prompt, task, episode/source frame, segment, normalized point and review flag.
- Joint state versus absolute action targets, and gripper state/command (1=open).

Green points have no review flag; red points need review. Green does not guarantee
semantic correctness. One point is held constant through each phase, matching
the training design; it is not a tracker of a moving object's image position.

Scrub the frame or `recording_time` timeline. Check phase starts especially:
grasp points should identify the object and placement points the intended bin.
Check RGB colors, crop coverage, gripper direction and state/action alignment.
There is no 3D robot reconstruction here: the zarr contains joint angles but no
URDF, camera calibration or object poses for a calibrated 3D scene.

To include all demonstrations:

```bash
.venv-viz/bin/python scripts/visualize_droidplus_zarr_rerun.py \
  --zarr data/droidplus_sorting_annotated.zarr \
  --all-episodes \
  --output reports/sorting_all.rrd

.venv-viz/bin/rerun reports/sorting_all.rrd
```

Add `--stride 3` to reduce logging, or `--max-frames 100` for a quick test.
Images are JPEG-compressed at quality 95 in the visualization only; the zarr
pixels stay unchanged. With all episodes, `recording_time` stitches episode
durations together without idle collection gaps; the global frame timeline
preserves zarr row IDs. Output files must have new names; existing recordings
are not overwritten. `--spawn` also opens the viewer while exporting.

The first-episode RRD was generated and verified using `rerun rrd verify` with
SDK 0.40.0. The GUI itself was not opened in the workspace.
See [Rerun's recording workflow](https://rerun.io/docs/getting-started/data-in)
and [2D point overlays](https://rerun.io/docs/reference/types/archetypes/points2d).
