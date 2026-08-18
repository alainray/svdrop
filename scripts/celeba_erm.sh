#!/bin/bash
#SBATCH --job-name=celeba_erm
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
LOGDIR="results/CelebA/erm/model_outputs_${SEED}"
PRETRAINEDPATH="pretrained_models/CelebA/erm_0.9_${SEED}.pth"
mkdir -p "$LOGDIR"

cd /workspace1/araymond/svdrop

python run_expt.py \
  -s confounder \
  -d CelebA \
  -t Blond_Hair \
  -c Male \
  --root_dir ../datasets \
  --lr 0.0001 \
  --batch_size 64 \
  --accum 2 \
  --weight_decay 0.0001 \
  --model resnet50 \
  --n_epochs 51 \
  --loss_type erm \
  --seed "$SEED" \
  --log_dir "$LOGDIR" \
  --save_best \
  --save_last 

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
