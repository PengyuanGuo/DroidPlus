#!/bin/bash
#SBATCH --job-name=dp_teleop_20261009_190030
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a30
#SBATCH --account=shey
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Submit from the DroidPlus dir:  mkdir -p logs && sbatch train_dp.slurm
set -euo pipefail

PROJ=/scratch/gilbreth/guo464/Projects
VENV=$PROJ/.venv-lerobot
DATASET=$PROJ/DroidPlus/output/lerobot/teleop_20261009_190030
OUT=$PROJ/runs/dp_teleop_20261009_190030_${SLURM_JOB_ID}   # lerobot refuses an existing output_dir

# Keep HF / torch caches off the small home quota
export HF_HOME=/scratch/gilbreth/$USER/.cache/huggingface
export TORCH_HOME=/scratch/gilbreth/$USER/.cache/torch
export HF_HUB_OFFLINE=1   # local dataset only; compute nodes may lack internet

source "$VENV/bin/activate"
nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# torchcodec can't find FFmpeg libs on Gilbreth -> decode videos with PyAV
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=local/droidplus \
  --dataset.root="$DATASET" \
  --dataset.video_backend=pyav \
  --policy.resize_shape='[188,336]' \
  --policy.crop_ratio=0.9 \
  --output_dir="$OUT" \
  --job_name=dp_teleop \
  --steps=50000 \
  --batch_size=32 \
  --num_workers=10 \
  --save_freq=10000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=droidplus-dp \
  --wandb.disable_artifact=true
