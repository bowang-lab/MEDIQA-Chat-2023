#!/bin/bash
# Submits our Flan-T5 large based approach for task A

TEST_FP="$1"  # Provided to the script by the submission system

# Notes:
# - The model will be downloaded from the HuggingFace model hub
# - The script expects a summary column in the test file, but we don't have one, so use the ID column
# - Use the run=1 argument to ensure that the output file is named correctly
# - Set the batch size to one and turn off all mixed precision to avoid errors
python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/task_a.yml" \
    test_file="$TEST_FP" \
    run="1" \
    output_dir="./outputs/wanglab/taskA/run1" \
    per_device_eval_batch_size=1 \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true