#!/bin/bash
# Weight-decay ablation for GDRO-FT on Waterbirds-95, second pass.
#
# The first pass swept eight lambdas over 3001 epochs and settled two things:
# the peak sits around epoch 300 and then decays, so the paper's 301-epoch budget
# was not truncated; and nothing below 1e-3 is distinguishable from 0, because
# with lr=1e-5 the L2 term is numerically invisible there. So this pass keeps the
# range that can actually matter and shortens the budget.
#
# It also runs on the backbone retrained by wb95_erm_clean.sh, so backbone, head,
# validation and test all come from the same generation of the dataset.
#
# Submit with:  sbatch --dependency=afterok:<erm_job> --array=0-4%3 scripts/wb95_lambda2.sh
#
#SBATCH --job-name=wb95_lambda2
#SBATCH -t 1-00:00
#SBATCH -o /workspace1/asoto/araymond/svdrop/exp_logs/%x_%A_%a.out
#SBATCH -e /workspace1/asoto/araymond/svdrop/exp_logs/%x_%A_%a.err
#SBATCH --chdir=/workspace1/asoto/araymond/svdrop
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

set -euo pipefail

ROOT=/workspace1/asoto/araymond/svdrop
PYTHON=~/pyenv/versions/mini/bin/python3
cd "$ROOT"

LAMBDAS=(0 0.01 0.1 0.3 1.0)
WD="${LAMBDAS[${SLURM_ARRAY_TASK_ID:-0}]}"
EPOCHS=1001

for SEED in 111 222 333; do
  EXP="ft.erm.gdro__c-95__src-train__bal-rw__frac-1.0__wd-${WD}__lr-1e-5__ep-${EPOCHS}__bn-eval"
  LOGDIR="results/CUB/${EXP}/model_outputs_${SEED}"
  mkdir -p "$LOGDIR"

  "$PYTHON" run_expt.py \
    -s confounder \
    -d CUB \
    -t waterbird_complete95 \
    -c forest2water2 \
    --root_dir ../datasets \
    --metadata_csv_name "metadata.csv" \
    --lr 1e-05 \
    --batch_size 64 \
    --weight_decay "$WD" \
    --model resnet50 \
    --n_epochs "$EPOCHS" \
    --loss_type group_dro \
    --seed "$SEED" \
    --log_dir "$LOGDIR" \
    --save_best \
    --finetune \
    --reweight_groups \
    --cache_features \
    --num_workers 6 \
    --pretrained_path "pretrained_models/CUB/erm_95_clean_${SEED}.pth" &

  sleep 90
done

wait
echo "Finished lambda=${WD} (job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?})"
