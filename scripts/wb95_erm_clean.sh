#!/bin/bash
# Retrain the ERM backbone for Waterbirds-95 on the dataset copy we actually have.
#
# The checkpoints inherited from the paper (pretrained_models/CUB/erm_95_*.pth)
# were trained on a different generation of waterbird_complete95: same test
# images, but a different background assignment and a different train/val split.
# 963 of the 1199 validation images of the copy on ventress were in that older
# training split, so the old backbone has memorised 80% of the set used to pick
# best_model. Test never overlapped, but backbone, head, val and test should all
# come from one instantiation before we read anything off the numbers.
#
# Same recipe as the paper's erm_95 (lr 1e-3, wd 1e-4, 301 epochs, batch 64) and
# the same selection rule: for an ERM run --save_best keeps the checkpoint with
# the best average validation accuracy, which is the head GDRO-FT then inherits.
#
# The weight decay can be overridden as the second argument. That matters because
# the paper trains the ERM backbone with wd=1e-4 at lr=1e-3, so the shrinkage per
# step is lr*wd = 1e-7, a hundred times weaker than the 1e-5 of Sagawa's GDRO
# recipe (lr 1e-5, wd 1.0). That backbone memorises its 4795 images by epoch 5,
# which is exactly what leaves GDRO-FT with no gradient to work with. Matching
# Sagawa's effective decay at this learning rate means wd=1e-2.
#
# One seed per array task, one GPU per task, assigned by SLURM. Never set
# CUDA_VISIBLE_DEVICES here: it overrides the allocation and can land the job on
# a GPU that belongs to somebody else.
#
# Submit with:
#   sbatch --array=0-2 scripts/wb95_erm_clean.sh          # wd por defecto (1e-4)
#   sbatch --array=0-2 scripts/wb95_erm_clean.sh 1e-2     # otro weight decay
#
#SBATCH --job-name=wb95_erm_clean
#SBATCH -t 1-00:00
#SBATCH -o /workspace1/asoto/araymond/svdrop/exp_logs/%x_%A_%a.out
#SBATCH -e /workspace1/asoto/araymond/svdrop/exp_logs/%x_%A_%a.err
#SBATCH --chdir=/workspace1/asoto/araymond/svdrop
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

ROOT=/workspace1/asoto/araymond/svdrop
PYTHON=~/pyenv/versions/mini/bin/python3
cd "$ROOT"

SEEDS=(111 222 333)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
WD="${1:-1e-04}"
# El checkpoint por defecto conserva el nombre que ya consumen los otros scripts.
TAG=$([ "$WD" = "1e-04" ] && echo "clean" || echo "wd${WD}")

EXP="e2e.erm__c-95__bal-none__frac-1.0__wd-${WD}__lr-1e-3__ep-301"
LOGDIR="results/CUB/${EXP}/model_outputs_${SEED}"
mkdir -p "$LOGDIR"

"$PYTHON" run_expt.py \
  -s confounder \
  -d CUB \
  -t waterbird_complete95 \
  -c forest2water2 \
  --root_dir ../datasets \
  --metadata_csv_name "metadata.csv" \
  --lr 1e-03 \
  --batch_size 64 \
  --weight_decay "$WD" \
  --model resnet50 \
  --n_epochs 301 \
  --loss_type erm \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --save_last \
  --num_workers 4

# Stage the selected checkpoint under the name the finetuning scripts expect.
cp "${LOGDIR}/best_model.pth" "pretrained_models/CUB/erm_95_${TAG}_${SEED}.pth"
echo "Finished seed ${SEED} wd=${WD}; staged pretrained_models/CUB/erm_95_${TAG}_${SEED}.pth"
