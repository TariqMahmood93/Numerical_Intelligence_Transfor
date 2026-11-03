#!/bin/bash
#SBATCH --job-name=nit_ext
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# Extension experiments. Each writes ONLY to runs/<name>/; the main study (results/, models/, paper/) is untouched.
#
#   sbatch --dependency=singleton run_extensions.slurm reasoner_3b
#   sbatch --dependency=singleton run_extensions.slurm reasoner_7b
#   sbatch --dependency=singleton run_extensions.slurm encoder_mpnet
#   sbatch --dependency=singleton run_extensions.slurm encoder_labse
#
# --dependency=singleton runs jobs with the same name one at a time, in submission order.
# A job that hits the time limit can simply be resubmitted: finished runs are reused.

set -euo pipefail
EXP=${1:?usage: sbatch run_extensions.slurm <reasoner_3b|reasoner_7b|encoder_mpnet|encoder_labse>}
mkdir -p logs runs

unset PYTHONPATH
source ~/venvs/nit/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

CORE=exp3_math_full,exp3b_math_budget,exp4a_teacher_raw,exp4b_teacher_verified,exp6_learned_projection
REASONER_RUNS=exp3b_math_budget,exp4a_teacher_raw,exp4b_teacher_verified

case "$EXP" in
  reasoner_3b)   ARGS="--teacher Qwen/Qwen2.5-3B-Instruct --teacher-batch 16 --runs $REASONER_RUNS" ;;
  reasoner_7b)   ARGS="--teacher Qwen/Qwen2.5-7B-Instruct --teacher-batch 4 --runs $REASONER_RUNS" ;;
  encoder_mpnet) ARGS="--student sentence-transformers/all-mpnet-base-v2 --runs $CORE --reuse-teacher-from ." ;;
  encoder_labse) ARGS="--student sentence-transformers/LaBSE --runs $CORE --reuse-teacher-from ." ;;
  *) echo "unknown experiment: $EXP"; exit 2 ;;
esac

nvidia-smi || true
echo "=== $EXP: python main.py $ARGS --out runs/$EXP"
python main.py $ARGS --out "runs/$EXP"
echo "=== $EXP: follow-up evaluation"
python src/extra_eval.py --root "runs/$EXP"
echo "=== $EXP: done"
