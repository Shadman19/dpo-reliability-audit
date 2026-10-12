#!/bin/bash
#SBATCH --job-name=safety_calib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/safety_%j.out
module load container_env/0.1 python3/2024.2-py310
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd ~/dpo_audit
crun ~/envs/hijack/bin/python safety_calib.py --out safety_results.json
