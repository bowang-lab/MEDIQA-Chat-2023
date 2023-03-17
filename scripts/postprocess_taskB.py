import typer
from datasets import load_dataset


def main(submission_fp: str = typer.Argument("Filepath (or URL) to the submission file (should be a CSV file).")):
    """Postprocesses the submission file for Task B.

    Example usage:
    python scripts/postprocess_taskB.py "./outputs/taskB_wanglab_run1.csv"
    """
    test = load_dataset(
        "csv",
        data_files={
            "test": submission_fp,
        },
    )["test"]

    # TODO: cleanup the headers
    # TODO: save the resulting file (*IN THE EXACT SAME FORMAT*) to disk (*AT THE EXACT SAME FILEPATH*)
    pass


if __name__ == "__main__":
    typer.run(main)
