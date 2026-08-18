#!/bin/bash
#SBATCH --job-name=wb_gdrfr_rw_new
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
DATASET="${2:-50}"
# 2) Logdir dependiente del seed
LOGDIR="results/CUB/erm_dfr_rw_new_${DATASET}/model_outputs_${SEED}"
PRETRAINEDPATH="pretrained_models/CUB/erm_${DATASET}_${SEED}.pth"
mkdir -p "$LOGDIR"

cd /workspace1/araymond/svdrop

python run_expt.py \
  -s confounder \
  -d CUB \
  -t waterbird_complete50 \
  -c forest2water2 \
  --root_dir ../datasets \
  --metadata_csv_name "metadata_dfr.csv" \
  --fraction 0.10086393088552917 \
  --lr 1e-05 \
  --batch_size 64 \
  --weight_decay 1.0 \
  --model resnet50 \
  --n_epochs 301 \
  --loss_type erm \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --save_last \
  --finetune \
  --pretrained_path "$PRETRAINEDPATH" \
  --reweight_groups

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
