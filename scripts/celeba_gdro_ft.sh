#!/bin/bash
# GDRO-FT on CelebA with the frozen backbone actually frozen.
#
# CelebA uses the same ResNet-50 as Waterbirds, so it has the same 53 BatchNorm
# layers that kept adapting to the group-balanced batches while the head was
# supposedly training on frozen features. It is the natural second data point for
# whether fixing that changes the method's numbers, and unlike Waterbirds it
# needs no new backbone: CelebA is not regenerated, it ships with the official
# partition file, and the group counts of the copy on ventress match the paper's
# logs exactly (train 71629 / 66874 / 22880 / 1387). So the inherited
# pretrained_models/CelebA/erm_*.pth are valid as they are.
#
# The paper's recipe is lr 1e-5, wd 0.1, batch 64, 51 epochs. Three weight decays
# around it, since on Waterbirds the paper's value turned out to sit far from the
# range where lambda does anything. The budget goes to 301 epochs because the
# head is trained with plain SGD at a constant learning rate, so the 51-epoch
# point can be read straight off the same CSVs -- unlike the BERT runs, where the
# warmup-linear schedule depends on the total.
#
# One (lambda, seed) pair per array task, one GPU per task, assigned by SLURM.
# Never set CUDA_VISIBLE_DEVICES here, and never run several trainings inside one
# allocation: both put work on GPUs the scheduler did not hand to this job.
#
# Submit with:  sbatch --array=0-8%3 scripts/celeba_gdro_ft.sh
#
#SBATCH --job-name=celeba_gdro_ft
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
#SBATCH --mem=48G

set -euo pipefail

ROOT=/workspace1/asoto/araymond/svdrop
PYTHON=~/pyenv/versions/mini/bin/python3
cd "$ROOT"

LAMBDAS=(0 0.1 1.0)
SEEDS=(111 222 333)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
WD="${LAMBDAS[$((TASK / 3))]}"
SEED="${SEEDS[$((TASK % 3))]}"
EPOCHS=301

EXP="ft.erm.gdro__c-std__src-train__bal-rw__frac-1.0__wd-${WD}__lr-1e-5__ep-${EPOCHS}__bn-eval"
LOGDIR="results/CelebA/${EXP}/model_outputs_${SEED}"
mkdir -p "$LOGDIR"

"$PYTHON" run_expt.py \
  -s confounder \
  -d CelebA \
  -t Blond_Hair \
  -c Male \
  --root_dir ../datasets \
  --metadata_csv_name "list_attr_celeba.csv" \
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
  --pretrained_path "pretrained_models/CelebA/erm_${SEED}.pth"

echo "Finished CelebA lambda=${WD} seed=${SEED}"
