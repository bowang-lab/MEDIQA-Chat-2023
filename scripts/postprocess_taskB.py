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
    # Could be easiest by writing a function and update the dataset using `.map()`
    # See: https://huggingface.co/docs/datasets/process#map

    # Save postprocessed submission file to disk
    test.to_csv(submission_fp)


if __name__ == "__main__":
    typer.run(main)
