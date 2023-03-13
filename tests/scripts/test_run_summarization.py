import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("task", ["A", "B"])
def test_run_summarization(tmp_path, task: str) -> None:
    """
    A simple tests that fails if the run_summarization.py script returns non-zero exit code.
    """
    cwd = Path(__file__).parent
    script_filepath = cwd / ".." / ".." / "scripts" / "run_summarization.py"
    config_filepath = cwd / ".." / ".." / "conf" / f"task_{task.lower()}.yml"
    train_file = (
        cwd
        / ".."
        / ".."
        / "test_fixtures"
        / f"MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/Task{task}/Task{task}-TrainingSet.csv"
    )
    validation_file = (
        cwd
        / ".."
        / ".."
        / "test_fixtures"
        / f"MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/Task{task}/Task{task}-ValidationSet.csv"
    )
    test_file = validation_file

    print("here")
    _ = subprocess.run(
        [
            "python",
            script_filepath,
            config_filepath,
            # Write all output and cache files to a temporary directory
            f"output_dir={tmp_path}",
            f"cache_dir={tmp_path}" "overwrite_output_dir=True",
            # Use dummy data
            f"train_file={train_file}",
            f"validation_file={validation_file}",
            f"test_file={test_file}",
            # Overide defaults to make test run in a reasonable amount of time
            "model_name_or_path=google/flan-t5-small",
            "max_source_length=4",
            "max_target_length=4",
            "do_train=True",
            "do_eval=True",
            "do_predict=True",
            "num_train_epochs=1",
            "max_train_samples=2",
            "max_eval_samples=2",
            # Disable mixed precision as this is not supported on CPU
            "fp16=False",
            "bf16=False",
            # Use smaller BERTScore/BELURT models to reduce test time
            "bertscore_model_type=microsoft/deberta-base-mnli",
            "bleurt_checkpoint=BLEURT-20-D3",
        ],
        capture_output=True,
        check=True,
    )
