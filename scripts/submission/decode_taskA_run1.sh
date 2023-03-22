#!/bin/bash
# Submits our Flan-T5 large based approach for task A

TEST_FP="$1"  # Provided to the script by the submission system

# Notes:
# - The model will be downloaded from the HuggingFace model hub
# - The script expects a summary column in the test file, but we don't have one, so use the dialogue column
# - The script expects a validation file, but we don't have one, so use the test file
# - Set the batch size to one and turn off all mixed precision to avoid errors
# - Set the bertscore_model_type and bleurt_checkpoint to null to avoid running them
# - Use the run=1 argument to ensure that the output file is named correctly
python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskA.yml" output_dir="./outputs" \
    model_name_or_path="wanglab/task-a-flan-t5-large-run-1" \
    summary_column="dialogue" \
    train_file=null \
    validation_file="$TEST_FP" \
    test_file="$TEST_FP" \
    per_device_eval_batch_size=1 \
    fp16=false \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true \
    bertscore_model_type=null \
    bleurt_checkpoint=null \
    run="1"

# Validate submission
python3 ./scripts/submission_checker.py "./outputs/taskA_wanglab_run1.csv"