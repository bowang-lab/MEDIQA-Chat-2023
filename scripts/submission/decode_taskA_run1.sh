#!/bin/bash

TEST_FP="$1"  # Provided to the script by the submission system

python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/task_a.yml" \
    test_file="$TEST_FP" \
    output_dir="./output/wanglab/task_a/run_1" \
    do_train=false \
    do_eval=false \
    do_predict=true