#!/bin/bash
# ASU Sol: one GPU job that runs, or continues, the whole bundle. Submit a chain of these with:  bash slurm/submit_chain.sh 4
# GPU: an 80 GB A100 (multi-control needs ~40 GB). 40 GB A100s, A30s, MIG slices, GH200 (ARM) and MI200 (AMD) will not work.
# If your lab owns H100 nodes, use:  -G h100:1  with  --partition=general --qos=grp_<labname>  (and drop the -C line).
#SBATCH --job-name=realtwin
#SBATCH --partition=public
#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=realtwin_%j.out
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash run_overnight.sh
