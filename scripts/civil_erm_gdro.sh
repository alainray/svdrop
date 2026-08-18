#!/bin/bash
#SBATCH --job-name=civil_erm_gdro
#SBATCH -t 1-00:00
#SBATCH -o /workspace1/araymond/svdrop/exp_logs/%x_%j.out
#SBATCH -e /workspace1/araymond/svdrop/exp_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=afraymon@uc.cl
#SBATCH --chdir=/workspace1/araymond/svdrop
#SBATCH --partition=ialab-eph
#SBATCH --nodelist=ventress
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# 1) Seed por argumento (default=111 si no se pasa)
SEED="${1:-111}"

# 2) Logdir dependiente del seed
LOGDIR="results/civilcomments/erm_gdro_0.9/model_outputs_${SEED}"
PRETRAINEDPATH="pretrained_models/civilcomments/erm_0.9_${SEED}.pth"
mkdir -p "$LOGDIR"
export HF_HOME=/workspace1/araymond/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers           # legacy; sigue funcionando
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"
cd /workspace1/araymond/svdrop

python run_expt.py \
  -s confounder \
  -d jigsaw \
  -t toxicity \
  -c identity_any \
  --root_dir ../datasets \
  --metadata_csv_name "all_data_with_identities.csv" \
  --lr 1e-05 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --model bert-base-uncased \
  --use_bert_params 1 \
  --n_epochs 5 \
  --loss_type group_dro \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --save_last \
  --finetune \
  --reweight_groups \
  --pretrained_path "$PRETRAINEDPATH"

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
