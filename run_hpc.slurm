#!/bin/bash
#SBATCH --job-name=num_transfer
#SBATCH --partition=gpu            # adapt to your cluster
#SBATCH --gres=gpu:1               # >= 8 GB VRAM selects Qwen2.5-1.5B-Instruct automatically
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Usage:
#   1) on a node WITH internet (login node):   python main.py --download-only
#   2) submit:                                  sbatch run_hpc.slurm
# Finished training runs are reused if the job is resubmitted after a timeout.

set -euo pipefail
mkdir -p logs
# module load python/3.11 cuda/12.1     # adapt
# source ~/venvs/nit/bin/activate

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1           # compute nodes are often offline; models were pre-fetched in step 1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

nvidia-smi || true
python main.py "$@"
