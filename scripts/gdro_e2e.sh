#!/bin/bash
# End-to-end GDRO on any of the five datasets, with the weight decay as a knob.
#
# The point is to check Sagawa et al.'s claim that group DRO needs strong L2 to
# work at all, and to see *how* it fails: if the failure is that the network
# memorises every group, the training worst-group accuracy should march to 100
# while the test one degrades, which is the same degeneracy we measured for
# GDRO-FT on Waterbirds.
#
# MultiNLI is already a data point without touching anything: its end-to-end GDRO
# baseline in the paper ran with weight decay 0.0, and its test worst-group
# accuracy peaks at epoch 0 (76.97) and falls to 74.56 by epoch 5 while training
# worst-group climbs from 76 to 92.
#
# This trains the whole network, so it cannot use --cache_features and it is
# expensive: hours per run for the vision datasets, more for BERT. The ialab
# partition caps a job at one day, and asking for more leaves it stuck forever
# with Reason=PartitionTimeLimit, so anything that will not fit in 24 h has to be
# split rather than given a longer limit.
#
# One seed per array task, one GPU per task, assigned by SLURM.
# Never set CUDA_VISIBLE_DEVICES here, and never run several trainings inside one
# allocation: both put work on GPUs the scheduler did not hand to this job.
#
# Submit with:  sbatch --array=0-2 scripts/gdro_e2e.sh CUB 0
#
#SBATCH --job-name=gdro_e2e
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

WD="${2:-$WD_E2E}"
SEEDS=(111 222 333)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"

export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

CORR=$([ "$DATASET" = "CUB" ] && echo 95 || echo std)
EXP="e2e.gdro__c-${CORR}__bal-rw__frac-1.0__wd-${WD}__lr-${LR}__ep-${EPOCHS_E2E}"
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
  --n_epochs "$EPOCHS_E2E" \
  --loss_type group_dro \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --reweight_groups \
  --num_workers 6

echo "Finished GDRO end-to-end ${DATASET} wd=${WD} seed=${SEED}"
