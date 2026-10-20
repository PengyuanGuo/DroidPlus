# MolmoPoint annotation for the real sorting recordings

Run from the **DroidPlus** directory. No LeRobot export or zarr conversion is
needed for this step. `scripts/annotate_sorting_points.py` reads `steps.jsonl`
and the left-camera JPEGs directly, using the existing `phase_index` and
`phase_prompt` fields. It expects six consecutive segments: 0,1,0,1,0,1.

## Environment

Use a working CUDA GPU. The workspace used to prepare this script did not have
PyTorch/Transformers installed or an accessible NVIDIA driver, so actual model
inference has not been verified here. Dataset validation and tests use no GPU.

Create a separate uv annotation environment. The RTX PRO 6000 Blackwell
(`sm_120`) requires a Blackwell-capable PyTorch build; use CUDA 12.8 wheels:

```bash
uv venv --python 3.11 .venv-molmo
uv pip install --python .venv-molmo/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-molmo/bin/python transformers==4.57.1 accelerate pillow einops numpy opencv-python-headless decord2
source .venv-molmo/bin/activate
```

If `.venv-molmo` already exists, skip `uv venv` and install into it directly.
To replace the old CUDA 12.4 packages in that environment, run:

```bash
uv pip install --python .venv-molmo/bin/python --upgrade \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

.venv-molmo/bin/python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print((torch.ones(4, device='cuda:0') + 1).cpu())"
```

The tensor operation tests actual GPU execution; `torch.cuda.is_available()`
alone can return true even when the installed kernels do not support the GPU.
PyTorch 2.7 introduced Blackwell support in its CUDA 12.8 wheels; the older
2.6/CUDA 12.4 setup fails on this GPU with "no kernel image is available".
See the [PyTorch release notes](https://pytorch.org/blog/pytorch-2-7/) and
[official installation commands](https://pytorch.org/get-started/previous-versions/#v271).

These instructions retain the model's Transformers 4.57.1 recommendation.
Choose a PyTorch CUDA build compatible with your machine's NVIDIA driver. See the
[official model card](https://huggingface.co/allenai/MolmoPoint-8B).
The model downloads automatically at first use. `--checkpoint /path/to/model`
can use an existing local checkpoint.

## Validate and test one episode

```bash
cd /home/atomic/Project/DroidPlus
python scripts/annotate_sorting_points.py \
  --dataset-root output_1017_phase_sorting \
  --full-frame --dry-run

python scripts/annotate_sorting_points.py \
  --dataset-root output_1017_phase_sorting \
  --full-frame --max-episodes 1 --device cuda:0
```

The first command validates all episode labels, image references and sampled
images without loading Molmo. The second tests grounding on one episode using
full camera coordinates. BF16 is the default. If needed, add
`--load-in-4bit` after installing `bitsandbytes`; use the same model precision at
deployment.

Each processed episode receives:

- `steps_molmo.jsonl`: original frame fields plus `point_pixel: [x, y]`.
- `molmo_point_annotations.json`: coordinate system, crop, source hash,
  checkpoint/precision, per-segment sampled predictions, failures and review flags.
- `molmo_previews/segment_00.jpg` through `segment_05.jpg`: the final averaged
  point drawn as a red cross on a sampled image from that segment.

Original `steps.jsonl`, images and collection metadata stay unchanged. Paths in
the new steps file resolve relative to the same episode directory.

## Inspect and process all episodes

Inspect all six previews from the first episode. Grasp phases should point to
the object, and place phases to the inside of the appropriate bin.

For training, choose a **fixed square ROI** covering the workspace and reuse it
at deployment and zarr conversion. `--roi X Y SIZE` uses raw-image pixel units:
top-left column X, top-left row Y, and square side length SIZE. It replaces
`--full-frame`. Do not silently reuse full-frame points after cropping.

For example, with your chosen values substituted for X, Y and SIZE:

```bash
python scripts/annotate_sorting_points.py \
  --dataset-root output_1017_phase_sorting \
  --roi X Y SIZE --max-episodes 1 --device cuda:0 --overwrite
```

Then remove `--max-episodes 1` to process all episodes using that same ROI.
`--overwrite` replaces only generated annotations and previews. To deliberately
annotate all full images instead, run:

```bash
python scripts/annotate_sorting_points.py \
  --dataset-root output_1017_phase_sorting \
  --full-frame --device cuda:0 --overwrite
```

Full-image points are normalized relative to the full image; crop-mode points
are normalized relative to the square crop. A normalized point is invariant
under resizing that same image region to 224x224. Keep the coordinate system
consistent in the later zarr converter.

## Grounding behavior and failures

Each **consecutive segment** is handled independently, even when several share
phase index 0 or 1. By default, up to five distinct frames are spread across the
first 25% of the segment, skipping its first two frames where possible. The
script retries failed/out-of-range points with a broader prompt, filters
distance outliers, averages the remaining points, and assigns that one point
to every frame in the segment.

At least three valid inliers are required, capped by the number of available
sample frames for short segments. A failed segment receives `[-1, -1]` and
`needs_review: true`; the command exits with status 2 if any segment fails.
Valid-looking points can still be wrong, so inspect previews as well as reports.
Device failures such as CUDA out-of-memory stop the command rather than being
recorded as ordinary pointing failures.

This script does not create a training zarr. The converter should later read
`steps_molmo.jsonl`, preserve its prompts and points, apply the recorded ROI to
the left images, and convert the native gripper convention to the training scale.

## Correct a phase using a chosen image

Use a clearer left-camera image from the phase you want to correct:

```bash
.venv-molmo/bin/python scripts/annotate_sorting_points_correction.py \
  --image output_1017_phase_sorting/teleop_20261017_174600/episode_005/left/000000200.jpg \
  --segment 3 \
  --device cuda:0
```

The image is matched to its actual row and consecutive phase segment. `--segment`
is an optional check of the **segment number 0–5**, not the repeating phase index
0/1. This example targets segment 03, `place banana to the blue bin`, and its
41 frames (row indices 188 through 228).

The correction uses one image, inherits the existing report's crop and model
settings, predicts a point, and overwrites `point_pixel` for every frame in that
one segment in `steps_molmo.jsonl`. It also updates the segment's annotation
report and `molmo_previews/segment_03.jpg`, using the chosen image for the new
preview. Other segments/previews and the original `steps.jsonl` stay unchanged.

If needed, add `--prompt "Point to the blue bin"` to change the primary pointing
instruction. A custom prompt disables the old fallback to avoid returning to a
different target; use `--fallback-prompt "..."` to provide a matching fallback.
Add `--dry-run`
to validate the image/segment association without loading the model or writing
anything. No `--full-frame`/`--roi` argument is needed: reusing the existing crop
keeps the stored coordinates consistent. A source hash mismatch stops correction
if the raw steps file changed since annotation.

Before overwriting, the script saves the previous steps, report and target
preview in `molmo_correction_backups/segment_03_<timestamp>.tar.gz` inside the
episode. A correction history in the report preserves the previous segment's
predictions. If both prompts fail, it leaves existing files unchanged.

The single-frame correction is flagged `needs_review: true` so you inspect its
new preview; a returned coordinate does not establish that the point is on the
correct object. No other segment's review flag is cleared.

You can also pass an existing segment preview. The script reads its source frame
from the annotation report and sends the **unmarked original JPEG** to Molmo,
using the saved ROI. Example with a custom object description:

```bash
.venv-molmo/bin/python scripts/annotate_sorting_points_correction.py \
  --image output_1017_phase_sorting/teleop_20261017_174600/episode_015/molmo_previews/segment_04.jpg \
  --segment 4 \
  --prompt "Point to the blue bottle" \
  --device cuda:0
```

This changes only the point annotations and recorded Molmo instruction for
segment 04. It does not rename the task's `phase_prompt` (e.g. `grasp salt bottle`).

## Correct a point by clicking

The manual script has the same input selection and output paths as the model
correction script. It needs only NumPy and OpenCV for image reading/writing;
no PyTorch, GPU or GUI build of OpenCV is required.

```bash
cd /home/atomic/Project/DroidPlus
.venv-molmo/bin/python scripts/annotate_sorting_points_manual.py \
  --image output_1017_phase_sorting/teleop_20261017_174600/episode_015/molmo_previews/segment_04.jpg \
  --segment 4
```

Open the `http://127.0.0.1:<port>/` URL printed in the terminal in a browser on
the same computer. The displayed image is the unmarked source JPEG with the
saved annotation crop applied. Click the correct point, click again to adjust,
then choose **Save point** or press Enter. Cancel/Escape leaves files unchanged.
The server exits after saving or cancelling. Closing the tab alone does not
stop the server; use Ctrl+C in the terminal to cancel it before saving.
`--port 8765` selects a fixed local port; the default chooses an available one.

The selected point is normalized as `[x / image_width, y / image_height]` in the
same cropped/full-image coordinate system as the existing annotations. Scaling
the image to fit the browser does not change the stored coordinates. The script
copies that point to every frame in the selected segment in `steps_molmo.jsonl`,
updates `molmo_point_annotations.json`, and replaces only that segment's preview.
The same automatic backups and correction history are preserved. Original
`steps.jsonl` and all other segments remain unchanged.

The report records `method: human_click_correction`, `human_verified: true`, and
`needs_review: false` for the saved segment. Other segments' flags are preserved;
the episode still needs review if any other segment does. Use `--dry-run` to
check the image/segment association without opening a browser or saving.
