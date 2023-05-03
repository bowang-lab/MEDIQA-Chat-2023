#!/bin/bash
# Requested resources
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=1:00:00
#SBATCH --job-name=baseline
#SBATCH --account=def-gbader
# Emails me when job starts, ends or fails
#SBATCH --mail-user=johnmgiorgi@gmail.com
#SBATCH --mail-type=FAIL
# Use this command to run the same job interactively
# salloc --mem=16G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-wanglab-ab_gpu
# salloc --mem=16G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-gbader

### Example usage ###
# sbatch "./scripts/slurm/run_summarization.sh" "./conf/taskA.yml" "./output/taskA"

### Usage notes ###
# The amount of time needed will depend on the batch size, model and number of GPUs requested.
# Flan-T5 base takes only a few minutes per epoch on a single A100 GPU with a batch size of 8.

### Environment ###
# Add your W&B key here to enable W&B reporting (or login with wandb login)
# export WANDB_API_KEY=""
# Note: If running on a node without internet access, sync your runs to W&B after the job finishes by calling:
# wandb sync -p mediqa-2023 -e wang-lab --sync-all

module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
module load python/3.10 StdEnv/2020 gcc/9.3.0 arrow/7.0.0
PROJECT_NAME="mediqa"
ACCOUNT_NAME="def-wanglab-ab"
source "$HOME/$PROJECT_NAME/bin/activate"
cd "$HOME/projects/$ACCOUNT_NAME/$USER/$PROJECT_NAME-chat-tasks-acl-2023" || exit

### Script arguments ###
# Required arguments
CONFIG_FILEPATH="$1"  # The path on disk to the JSON config file
OUTPUT_DIR="$2"       # The path on disk to save the output to

### Job ###
# This calls a modified version of the example summarization script from HF (with Trainer). For details,
# see: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization#with-trainer
# Note: it is better to cache models, metrics and datasets to $SCRATCH, as it is much faster and larger than $HOME.
WANDB_MODE=offline \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
HF_EVALUATE_OFFLINE=1 \
python ./scripts/run_summarization.py "./conf/base.yml" "$CONFIG_FILEPATH" output_dir="$OUTPUT_DIR" \
    cache_dir="$SCRATCH/.cache/huggingface" overwrite_output_dir=true bleurt_checkpoint=null \
exit
