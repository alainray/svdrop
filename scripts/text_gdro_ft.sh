#!/bin/bash
# GDRO-FT on the text datasets with the frozen encoder actually frozen.
#
# The paper's text runs called model.train() on a BERT whose parameters were all
# requires_grad=False, so the 38 dropout layers stayed on and the "frozen"
# representation of an example changed every time it was seen. Nothing was
# corrupted the way BatchNorm was in Waterbirds, but the method was not running
# on fixed features. These runs put the encoder in eval mode.
#
# Once the encoder is fixed, --cache_features precomputes the pooled [CLS]
# vectors once and the head trains on 768-dim tensors, which is what makes a
# budget beyond five epochs affordable at all: every epoch of the old path
# pushed 412k (MultiNLI) or 448k (CivilComments) examples through twelve
# transformer layers.
#
# Two budgets per dataset: the paper's, for comparability, and a long one, since
# the head can now be trained to convergence. They are separate runs rather than
# one long run because BERT uses a warmup-linear schedule over t_total, so the
# five-epoch point cannot be read off a longer run.
#
# Submit with:
#   sbatch --array=0-1 scripts/text_gdro_ft.sh MultiNLI
#   sbatch --array=0-1 scripts/text_gdro_ft.sh civilcomments
#
#SBATCH --job-name=text_gdro_ft
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

DATASET="${1:?uso: sbatch scripts/text_gdro_ft.sh {MultiNLI|civilcomments}}"
BUDGETS=(5 100)
EPOCHS="${BUDGETS[${SLURM_ARRAY_TASK_ID:-0}]}"

case "$DATASET" in
  MultiNLI)
    TARGET=gold_label_random
    CONF=sentence2_has_negation
    METADATA="metadata.csv"
    LR=2e-05
    WD=0.1
    BS=32
    RESULTS=MultiNLI
    ;;
  civilcomments)
    DATASET_ARG=jigsaw
    TARGET=toxicity
    CONF=identity_any
    METADATA="all_data_with_identities.csv"
    LR=1e-05
    WD=0.01
    BS=16
    RESULTS=civilcomments
    ;;
  *)
    echo "dataset desconocido: $DATASET" >&2; exit 1 ;;
esac
DATASET_ARG="${DATASET_ARG:-$DATASET}"

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

for SEED in 111 222 333; do
  EXP="ft.erm.gdro__c-std__src-train__bal-rw__frac-1.0__wd-${WD}__lr-${LR}__ep-${EPOCHS}__bn-eval"
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
    --model bert-base-uncased \
    --use_bert_params 1 \
    --n_epochs "$EPOCHS" \
    --loss_type group_dro \
    --seed "$SEED" \
    --log_dir "$LOGDIR" \
    --save_best \
    --finetune \
    --reweight_groups \
    --cache_features \
    --num_workers 6 \
    --pretrained_path "pretrained_models/${RESULTS}/erm_0.9_${SEED}.pth"
done

echo "Finished ${DATASET} with ${EPOCHS} epochs"
