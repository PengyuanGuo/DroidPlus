#!/bin/bash
#SBATCH --job-name=pi05_jointpos_chicken_lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a100-40gb
#SBATCH --account=shey
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00

# LoRA fine-tune of pi05_droid_jointpos on DroidPlus teleop data (openpi, JAX, 2x A100-40GB data parallel).
# Config: pi05_droid_jointpos_droidplus_lora (batch 32, 10k steps, EMA off, checkpoint every 1k).
#   Submit from the DroidPlus dir:   mkdir -p logs && sbatch train_openpi.slurm
#   Resume an interrupted run:       RESUME_EXP=<exp_name> sbatch train_openpi.slurm
# Checkpoints (every 1k steps, all kept): $PROJ/runs/openpi/$CONFIG/<exp_name>/<step>/
# Dataset: DroidPlus/scripts/convert_teleop_to_openpi.py -> $HF_LEROBOT_HOME/droidplus/chicken_blue_bowl
set -euo pipefail

PROJ=/scratch/gilbreth/guo464/Projects
OPENPI=$PROJ/openpi
CONFIG=pi05_droid_jointpos_droidplus_lora

# Keep caches/data off the small home quota; base checkpoint + tokenizer are pre-downloaded here.
export OPENPI_DATA_HOME=/scratch/gilbreth/$USER/.cache/openpi
export HF_LEROBOT_HOME=$PROJ/DroidPlus/output/openpi
export HF_HOME=/scratch/gilbreth/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1   # flush per-step loss lines to the .out log
# JAX ships its own CUDA libs; don't let a loaded cuda module shadow them.
unset LD_LIBRARY_PATH

if [[ -n "${RESUME_EXP:-}" ]]; then
  EXP_ARGS=(--exp-name="$RESUME_EXP" --resume)
else
  EXP_ARGS=(--exp-name="chicken_lora_${SLURM_JOB_ID}")
fi

cd "$OPENPI"
nvidia-smi
.venv/bin/python -c "import jax; print('jax devices:', jax.devices())"

.venv/bin/python scripts/train.py "$CONFIG" \
  "${EXP_ARGS[@]}" \
  --fsdp-devices=1 \
  --checkpoint-base-dir="$PROJ/runs/openpi"
