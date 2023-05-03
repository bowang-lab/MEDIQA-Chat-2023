#!/bin/bash
# Submits our LLM (via LangChain) based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

OUTPUT_DIR="./output"
RUN="1"

# Notes:
# - You must provide an OPENAI_API_KEY for this to work
OPENAI_API_KEY="" \
python3 ./scripts/run_langchain.py "./data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-TrainingSet.csv" \
    "$TEST_FP" \
    "./outputs" \
    --temperature 0.2 \
    --task "B" \
    --run "$RUN"

# Validate submission
python3 ./scripts/submission_checker.py "$OUTPUT_DIR/taskB_wanglab_run$RUN.csv"