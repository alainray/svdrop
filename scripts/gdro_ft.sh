#!/bin/bash
# GDRO-FT on any of the five datasets, with the weight decay as a sweepable knob
# and an optional random head.
#
# Everything else is the paper's recipe (see scripts/dataset_config.sh): same
# optimiser, learning rate, batch size and epoch budget, the budget being the one
# the base model was trained for. The frozen body runs in eval mode, which is the
# fix, and --cache_features precomputes its output once.
#
# Two questions this is meant to answer:
#
# - Does starting the head from random weights instead of inheriting the ERM head
#   change anything? On Waterbirds the inherited head already separates the
#   training set with huge margins, so the GDRO loss starts at 3e-5 and there is
#   no gradient to work with. A random head starts with real loss.
# - Does GDRO-FT survive without L2? Sagawa et al. report that GDRO needs strong
#   regularisation; if the reason is that the model otherwise memorises every
#   group and the robust objective goes flat, then lambda=0 should degenerate the
#   same way here.
#
# One (lambda, seed) pair per array task, one GPU per task, assigned by SLURM.
# Never set CUDA_VISIBLE_DEVICES here, and never run several trainings inside one
# allocation: both put work on GPUs the scheduler did not hand to this job.
#
# Submit with:
#   sbatch --array=0-2      scripts/gdro_ft.sh CelebA 0            # un solo lambda
#   sbatch --array=0-17%4   scripts/gdro_ft.sh CUB "0 0.01 0.1 0.3 1.0 3" reinit
#
# To read the features off a different backbone, export PRETRAIN_PATTERN (with
# the literal word SEED where the seed goes) and FEAT_TAG, which is appended to
# the run name so the two do not collide:
#
#   sbatch --export=ALL,PRETRAIN_PATTERN=pretrained_models/CUB/erm_95_wd1e-02_SEED.pth,\
#          FEAT_TAG=__feat-ermwd1e-2 --array=0-11%4 scripts/gdro_ft.sh CUB "0 0.1 1.0 3"
#
# Exporting DROP_DIRS_LIST sweeps the number of spurious feature directions
# removed instead of the weight decay, which then stays fixed at the single value
# given as the second argument:
#
#   sbatch --export=ALL,DROP_DIRS_LIST="1 2 3 4 5" --array=0-14%4 \
#          scripts/gdro_ft.sh CUB 0 reinit
#
# The array needs 3 tasks per swept value (one per seed).
#
#SBATCH --job-name=gdro_ft
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
#SBATCH --mem=64G

set -euo pipefail

ROOT=/workspace1/asoto/araymond/svdrop
PYTHON=~/pyenv/versions/mini/bin/python3
cd "$ROOT"
source scripts/dataset_config.sh

DATASET="${1:?falta el dataset}"
configure "$DATASET"

read -r -a LAMBDAS <<< "${2:-$WD_FT}"
SEEDS=(111 222 333)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$((TASK % 3))]}"

DROP=0
DROP_SUFFIX=""
if [ -n "${DROP_DIRS_LIST:-}" ]; then
  read -r -a DROPS <<< "$DROP_DIRS_LIST"
  DROP="${DROPS[$((TASK / 3))]}"
  WD="${LAMBDAS[0]}"
  DROP_SUFFIX="__drop-${DROP}"
else
  WD="${LAMBDAS[$((TASK / 3))]}"
fi

EXTRA=()
INIT=""
if [ "${3:-}" = "reinit" ]; then
  EXTRA+=(--reinit_head)
  INIT="__init-random"
fi

export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

PRETRAIN="${PRETRAIN_PATTERN:-$PRETRAIN}"

CORR=$([ "$DATASET" = "CUB" ] && echo 95 || echo std)
# FEAT_TAG marca en el nombre de que backbone salen las features cuando no es el
# por defecto del dataset, p.ej. un ERM entrenado con otro weight decay.
EXP="ft.erm.gdro__c-${CORR}__src-train__bal-rw__frac-1.0__wd-${WD}__lr-${LR}__ep-${EPOCHS_FT}__bn-eval${INIT}${DROP_SUFFIX}${FEAT_TAG:-}"
LOGDIR="results/${RESULTS}/${EXP}/model_outputs_${SEED}"
mkdir -p "$LOGDIR"

"$PYTHON" run_expt.py \
  -s confounder \
  -d "$DATASET_ARG" \
  -t "$TARGET" \
  -c "$CONF" \
  --root_dir ../datasets \
  --metadata_csv_name "$METADATA" \
  --lr "$LR" \
  --batch_size "$BS" \
  --weight_decay "$WD" \
  --model "$MODEL" \
  --use_bert_params 1 \
  --n_epochs "$EPOCHS_FT" \
  --loss_type group_dro \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --finetune \
  --reweight_groups \
  --cache_features \
  --num_workers 6 \
  --drop_spurious_dirs "$DROP" \
  --pretrained_path "${PRETRAIN/SEED/$SEED}" \
  "${EXTRA[@]}"

echo "Finished GDRO-FT ${DATASET} wd=${WD} drop=${DROP} seed=${SEED}${INIT}"
