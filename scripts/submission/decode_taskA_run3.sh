#!/bin/bash
# Submits our LLM (via LangChain) based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

# Notes:
# - You must provide an OPENAI_API_KEY for this to work
OPENAI_API_KEY="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" \
python3 ./scripts/run_langchain.py "./data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-TrainingSet.csv" \
    "$TEST_FP" \
    "./outputs" \
    --temperature 0.0 \
    --task "A" \
    --run "3"