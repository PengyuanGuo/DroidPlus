# Teleop Data Collection to Zarr

This guide uses four-phase collection as an example. Adjust paths, IP addresses, and camera IDs for your machines.

## 1. Start the Robot Service on the Other Computer

On the robot control computer, enter the DroidPlus directory, activate the configured environment, and run:

```bash
make franky_service
```

The service name is `franky_service`. Unlock the robot and enable FCI.

## 2. Open the Collection UI on Your Computer

First, start the camera service in the camera environment and keep this terminal running:

```bash
make camera_service
```

Open another terminal:

```bash
cd /home/sky/research/Project/DroidPlus
source .venv-teleop/bin/activate

export NUC_IP=192.168.9.167
export FRANKY_SERVICE_URL="http://${NUC_IP}:54321"
export CAMERA_SERVICE_URL=http://127.0.0.1:54322

python scripts/run_teleop_ui.py \
  --gripper franka \
  --left-camera-id 31262010 \
  --wrist-camera-id 14633207 \
  --output-dir output_1023_4phase_sorting \
  --port /dev/ttyACM0
```

Open <http://localhost:54325>, start an episode, use number keys to switch phases, and press Esc to stop. Then save the episode labels.

## 3. Annotate phase_index and phase_prompt

Define the phases for each object before collection:

| phase_index | phase_prompt (banana example) |
|---|---|
| 0 | `pre-grasp banana` |
| 1 | `grasp banana` |
| 2 | `pre-place banana` |
| 3 | `place banana` |

Use `0 → 1 → 2 → 3` for each object, then restart at `0` for the next object.

**The current UI records only phase_index; phase_prompt is initially empty.** After collection and before MolmoPoint annotation, fill in phase_prompt for every frame in `steps.jsonl` according to the actual object order. Keep the text identical within each segment.

For the six-segment format, each of three objects has a grasp segment and a place segment, with labels `0,1,0,1,0,1`. Example prompts are `grasp banana` and `place banana to the blue bin`. Supported objects are banana, orange, apple, and salt bottle. Fruits go into the blue bin; the salt bottle goes into the white bin.

## 4. Annotate Points with MolmoPoint

Use the configured `.venv-molmo` environment:

```bash
cd /home/sky/research/Project/DroidPlus
export DATASET="$PWD/output_1023_4phase_sorting"
export ATOMIC_POLICY_ROOT=/home/sky/research/Project/atomic_policy

# Validate labels and images
.venv-molmo/bin/python scripts/annotate_sorting_points.py \
  --dataset-root "$DATASET" \
  --atomic-policy-root "$ATOMIC_POLICY_ROOT" \
  --full-frame --dry-run

# Annotate points automatically
.venv-molmo/bin/python scripts/annotate_sorting_points.py \
  --dataset-root "$DATASET" \
  --atomic-policy-root "$ATOMIC_POLICY_ROOT" \
  --full-frame --device cuda:0
```

This generates `steps_molmo.jsonl`, `molmo_point_annotations.json`, and `molmo_previews/`. Add `--overwrite` to regenerate existing results, but note that it also overwrites manual corrections.

## 5. Correct Incorrect Points

```bash
export EP="$DATASET/teleop_20261023_192358/episode_004"

.venv-molmo/bin/python scripts/annotate_sorting_points_manual.py \
  --image "$EP/molmo_previews/segment_04.jpg" \
  --segment 4 \
  --port 8765
```

Open <http://127.0.0.1:8765>, click the correct location, then click **Save point** or press Enter. `segment 4` means the fifth consecutive segment, not phase_index.

To have the model predict the point again:

```bash
.venv-molmo/bin/python scripts/annotate_sorting_points_correction.py \
  --image "$EP/molmo_previews/segment_04.jpg" \
  --segment 4 \
  --atomic-policy-root "$ATOMIC_POLICY_ROOT" \
  --device cuda:0
```

To review all segments:

```bash
.venv-molmo/bin/python scripts/annotate_sorting_points_review.py \
  --dataset-root "$DATASET" --all --port 8765
```

Run the web tools separately to avoid port conflicts. Corrections update the selected segment's point annotations, report, and preview, with automatic backups.

## 6. Convert to Zarr

Install the conversion dependencies if needed:

```bash
uv pip install --python .venv-molmo/bin/python \
  'zarr>=2.12,<3' 'numcodecs>=0.10,<0.16'
```

After reviewing the point annotations, convert the dataset:

```bash
export ZARR_OUT="$PWD/data/sorting_4phase_annotated.zarr"

.venv-molmo/bin/python scripts/convert_droidplus_to_zarr.py \
  --dataset-root "$DATASET" \
  --output "$ZARR_OUT" \
  --task-prompt "Sort the fruits into the blue bin and the salt bottle into the white bin." \
  --require-reviewed
```

Add `--dry-run` first to validate the inputs. The converter reads the corrected `steps_molmo.jsonl`, annotation reports, and left/wrist images. By default, it produces 224×224 images and maps points into the cropped image coordinates.

Resolve failed points `[-1,-1]` and segments still requiring review before conversion. If points fall outside the default crop, specify a suitable `--roi X Y SIZE`. If you modify points after conversion, convert again with `--overwrite` to back up and replace the existing Zarr.
