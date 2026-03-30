#!/bin/bash
#SBATCH --job-name=clear_minihack
#SBATCH --array=1-2
#SBATCH --output=slurm/logs/clear_%A_%a.out
#SBATCH --error=slurm/logs/clear_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=mingyan-a100
#SBATCH --account=mingyan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# NOTE: Check your account name and adjust --account if needed:
# #SBATCH --account=your_account

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cora_minihack

export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $TMPDIR

cd ~/CORA/continual_rl

mkdir -p slurm/logs

OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
/home/lalkarmi/miniconda3/envs/cora_minihack/bin/python main.py \
    --config-file configs/minihack/clear_minihack_paperdefaults.json \
    --output-dir tmp/clear_run${SLURM_ARRAY_TASK_ID}
