#!/bin/bash
# Weight-decay ablation for GDRO-FT on Waterbirds-95.
#
# The paper reports 73.68 WGA at lambda=1.0 and 301 epochs, but the head is still
# improving when the budget runs out, so it is not clear whether 73.68 is a
# truncated optimisation or the effect of the L2 term. This sweeps lambda over
# four orders of magnitude with a 10x longer budget; because every epoch is
# logged, the 301-epoch point can be read off the same CSVs.
#
# The backbone is genuinely frozen here (--freeze via --finetune, BatchNorm in
# eval mode) so --cache_features precomputes the features once and each epoch is
# a couple of matrix products.
#
# Submit with:  sbatch --array=0-7%4 scripts/wb95_lambda.sh
#
#SBATCH --job-name=wb95_lambda
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

LAMBDAS=(0 1e-5 1e-4 1e-3 1e-2 1e-1 1.0 10.0)
WD="${LAMBDAS[${SLURM_ARRAY_TASK_ID:-0}]}"
EPOCHS=3001

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
    --pretrained_path "pretrained_models/CUB/erm_95_${SEED}.pth" &

  # Stagger so the three feature-extraction passes do not peak together on one
  # 8 GB card; once cached, each run needs almost no memory.
  sleep 90
done

wait
echo "Finished lambda=${WD} (job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?})"
