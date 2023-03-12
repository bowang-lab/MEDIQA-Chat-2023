#!/bin/bash

####################################################################################################
# General setup for use on the Advanced Research Computing (ARC) clusters
# See https://alliancecan.ca/en/services/advanced-research-computing for cluster details.
####################################################################################################

# Load the required modules
# Notes: 
# - arrow needed for HF Datasets both during installation and use
module purge
module load python/3.10 StdEnv/2020 gcc/9.3.0 arrow/7.0.0

# Setup the virtual environment under home
PROJECT_NAME="mediqa-chat-tasks-acl-2023"
virtualenv --no-download "$HOME/$PROJECT_NAME"
source "$HOME/$PROJECT_NAME/bin/activate"
pip install --no-index --upgrade pip

# Setup the project and scratch directories
# NOTE: On some clusters (e.g. Narval), the PROJECT env var does not exist, so you will have to cd manually
# cd "$PROJECT/$USER" || exit
cd "$PROJECT/$USER" || exit
git clone "https://github.com/bowang-lab/mediqa-chat-tasks-acl-2023.git"
cd "mediqa-chat-tasks-acl-2023" || exit
# Outputs from jobs (like model checkpoints) should be stored in the scratch directory
# See: https://docs.alliancecan.ca/wiki/Storage_and_file_management
mkdir -p "$SCRATCH/$PROJECT_NAME"

# Install the package
pip install -r requirements.txt
pip install -r dev-requirements.txt

# Check that all tests are passing
pytest tests