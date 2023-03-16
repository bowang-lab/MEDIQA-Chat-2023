#!/bin/bash
# Submits our LLM based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

# Notes:
# - You must provide an OPENAI_API_KEY for this to work
OPENAI_API_KEY="..." \
python3 ./scripts/llm.py "$TEST_FP" "./outputs/wanglab/taskB/run2" --task "B" --run "2"