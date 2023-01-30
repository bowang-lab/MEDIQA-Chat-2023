#!/bin/bash
# Requested resources
#SBATCH --mem=36G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=3:00:00
#SBATCH --job-name=baseline
#SBATCH --account=def-wanglab-ab
# Emails me when job starts, ends or fails
#SBATCH --mail-user=johnmgiorgi@gmail.com
#SBATCH --mail-type=FAIL
# Use this command to run the same job interactively
# salloc --mem=36G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-wanglab-ab

# Load the required modules
# Notes: 
# - arrow needed for HF Datasets both during installation and use
module purge
module load python/3.10 StdEnv/2020 gcc/9.3.0 arrow/7.0.0

# Setup the virtual environment under home
PROJECT_NAME="mediqa"
source "$HOME/$PROJECT_NAME/bin/activate"

TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python ./scripts/train_baseline.py

exit