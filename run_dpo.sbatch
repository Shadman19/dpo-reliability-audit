#!/bin/bash
#SBATCH --job-name=dpo_audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/dpo_%A_%a.out
# array 0 = base model (eval only); 1..3 = DPO with beta 0.05 / 0.1 / 0.3
module load container_env/0.1 python3/2024.2-py310
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BETAS=(0 0.05 0.1 0.3)
B=${BETAS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}
TAG=${TAG:-}
cd ~/dpo_audit
crun ~/envs/hijack/bin/python dpo_lora.py --beta $B --seed $SEED --out runs${TAG}/beta${B}_s${SEED} $EXTRA
