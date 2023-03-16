#!/bin/bash
# Submits our LED large based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/task_b.yml" \
    test_file="$TEST_FP" \
    run="1" \
    output_dir="./outputs/wanglab/task_b/run1" \
    per_device_eval_batch_size=1 \
    fp16=false \
    do_train=false \
    do_eval=false \
    do_predict=true