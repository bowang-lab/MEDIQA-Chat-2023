#!/bin/bash
# Submits our LED large based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

# Notes:
# - The model will be downloaded from the HuggingFace model hub
# - The script expects a summary column in the test file, but we don't have one, so use the dataset column
# - Use the run=1 argument to ensure that the output file is named correctly
# - Set the batch size to one and turn off all mixed precision to avoid errors
python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskB.yml" \
    test_file="$TEST_FP" \
    model_name_or_path="wanglab/task-b-led-large-16384-pubmed" \
    summary_column="dataset" \
    run="1" \
    output_dir="./outputs" \
    per_device_eval_batch_size=1 \
    fp16=false \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true