#!/bin/bash
# Submits our LLM (via OpenAI) based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

# Notes:
# - You must provide an OPENAI_API_KEY for this to work
OPENAI_API_KEY="..." \
python3 ./scripts/run_openai.py "$TEST_FP" "./outputs/wanglab/taskB/run3" --task "B" --run "3"