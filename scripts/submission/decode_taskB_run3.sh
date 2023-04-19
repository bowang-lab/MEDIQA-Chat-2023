#!/bin/bash
# Submits our LED large based approach for task B

TEST_FP="$1"  # Provided to the script by the submission system

OUTPUT_DIR="./output"
RUN="3"

# Notes:
# - The model will be downloaded from the HuggingFace model hub
# - The script expects a summary column in the test file, but we don't have one, so use the dataset column
# - Set the batch size to one to avoid OOM errors
# - Turn off mixed precision to avoid errors on CPUs and some GPUs
# - Set evaluation_strategy="'no'" and load_best_model_at_end=false to avoid evaluation
# - Set bertscore_model_type=null and bleurt_checkpoint=null to avoid loading them
# - Use the run=3 argument to ensure that the output file is named correctly
python3 ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskB.yml" output_dir="$OUTPUT_DIR" \
    model_name_or_path="wanglab/task-b-led-large-16384-pubmed-run-$RUN" \
    summary_column="dataset" \
    train_file=null \
    validation_file=null \
    test_file="$TEST_FP" \
    per_device_eval_batch_size=1 \
    fp16=false \
    bf16=false \
    do_train=false \
    do_eval=false \
    do_predict=true \
    evaluation_strategy="'no'" \
    load_best_model_at_end=false \
    bertscore_model_type=null \
    bleurt_checkpoint=null \
    run="$RUN"

# Postprocess the output file to clean up section headers
python3 ./scripts/postprocess_taskB.py "$OUTPUT_DIR/taskB_wanglab_run$RUN.csv"

# Validate submission
python3 ./scripts/submission_checker.py "$OUTPUT_DIR/taskB_wanglab_run$RUN.csv"