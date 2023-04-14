import evaluate
import pandas as pd
import typer
from rich import print


def main(
    input_fp: str = typer.Argument(
        "Filepath (or URL) to the original subtask A train, validation or test set (should be a CSV file)."
    ),
):
    """Prints the random and majority baselines for section header prediction on subtask A.

    Example usage:

    python header_prediction_baselines.py "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv"
    """
    exact_match = evaluate.load("exact_match")
    gt_section_headers = pd.read_csv(input_fp).section_header

    predictions = gt_section_headers.sample(frac=1).reset_index(drop=True)
    random_baseline = exact_match.compute(predictions=predictions.tolist(), references=gt_section_headers.tolist())
    score = round(random_baseline["exact_match"] * 100, 4)
    print(f"Random baseline: {score}")

    predictions = [gt_section_headers.value_counts().index[0]] * len(gt_section_headers)
    majority_baseline = exact_match.compute(predictions=predictions, references=gt_section_headers.tolist())
    score = round(majority_baseline["exact_match"] * 100, 4)
    print(f"Majority baseline: {score}")


if __name__ == "__main__":
    typer.run(main)
