#!/bin/bash
# Submits our Flan-T5 large based approach for task A

TEST_FP="$1"  # Provided to the script by the submission system

python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/task_a.yml" \
    test_file="$TEST_FP" \
    output_dir="./output/wanglab/task_a/run_1" \
    per_device_eval_batch_size=1 \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true