#!/bin/bash
#SBATCH --job-name=multinli_erm
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
LOGDIR="results/MultiNLI/gdro_0.9/model_outputs_${SEED}"
mkdir -p "$LOGDIR"

cd /workspace1/araymond/svdrop

python run_expt.py \
  -s confounder \
  -d MultiNLI \
  -t gold_label_random \
  -c sentence2_has_negation \
  --root_dir ../datasets \
  --lr 2e-05 \
  --batch_size 32 \
  --weight_decay 0.1 \
  --model bert-base-uncased \
  --use_bert_params 1 \
  --n_epochs 5 \
  --loss_type group_dro \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --save_last \
  --reweight_groups

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
