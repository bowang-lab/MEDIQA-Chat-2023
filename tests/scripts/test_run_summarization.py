import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("task_conf", ["task_a.yml", "task_b.yml"])
def test_run_summarization(tmp_path, task_conf: str) -> None:
    """
    A simple tests that fails if the run_summarization.py script returns non-zero exit code.
    """
    cwd = Path(__file__).parent
    script_filepath = cwd / ".." / ".." / "scripts" / "run_summarization.py"
    config_filepath = cwd / ".." / ".." / "conf" / task_conf
    _ = subprocess.run(
        [
            "python",
            script_filepath,
            config_filepath,
            f"output_dir={tmp_path}",
            "overwrite_output_dir=True",
            # Overide defaults to make test run in a reasonable amount of time
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
