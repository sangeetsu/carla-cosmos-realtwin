#!/bin/bash
# Slurm template. Edit the GPU request / partition / account lines for your cluster, then:  sbatch slurm/overnight.sbatch
#SBATCH --job-name=realtwin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=14:00:00
#SBATCH --output=realtwin_%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
cd "$SLURM_SUBMIT_DIR"
bash check.sh && bash run_overnight.sh
