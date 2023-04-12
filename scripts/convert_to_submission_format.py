import pandas as pd
import typer
from rich import print


TASK_A = "A"
TASK_B = "B"
TASK_C = "C"
TASKS = [TASK_A, TASK_B, TASK_C]

# Original column names (Task A)
ID_COL = "ID"
SECTION_HEADER_COL = "section_header"
SECTION_TEXT_COL = "section_text"
# Original column names (Task B and C)
ENCOUNTER_ID_COL = "encounter_id"
NOTE_COL = "note"

# Submission file column names
TEST_ID = "TestID"
SYSTEM_OUTPUT_1 = "SystemOutput1"
SYSTEM_OUTPUT_2 = "SystemOutput2"
SYSTEM_OUTPUT = "SystemOutput"


def main(
    input_fp: str = typer.Argument(
        "Filepath (or URL) to the original train, validation or test set (should be a CSV file)."
    ),
    output_fp: str = typer.Argument("Filepath where the converted CSV file should be saved."),
    task: str = typer.Option(TASK_B, help=f"Task name. Should be one of {TASKS}"),
) -> None:
    """Converts the shared task file in the original format it was given (input_fp) to the submission file format.

    Example usage:

    # Task A
    python convert_to_submission_format.py "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv" \
        "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet-SubmissionFormat" \
        --task "A"
    # Task B
    python convert_to_submission_format.py "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-ValidationSet.csv" \
        "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-ValidationSet-SubmissionFormat" \
        --task "B"
    """
    # Error handling
    if task not in TASKS:
        raise ValueError(f"Task should be one of {TASKS}. Got: '{task}'")

    df = pd.read_csv(input_fp)
    print(f"Loaded original dataframe from '{input_fp}'")

    # Rename to match column names of the submission file
    if task == TASK_A:
        df.rename(
            columns={ID_COL: TEST_ID, SECTION_HEADER_COL: SYSTEM_OUTPUT_1, SECTION_TEXT_COL: SYSTEM_OUTPUT_2},
            inplace=True,
        )
        # Add the expected prefix to the IDs
        df[TEST_ID] = df[TEST_ID].apply(lambda x: f"task{task}{x}")
    elif task == TASK_B:
        df.rename(columns={ENCOUNTER_ID_COL: TEST_ID, NOTE_COL: SYSTEM_OUTPUT}, inplace=True)
    else:
        raise NotImplementedError(f"Task {task} is not implemented yet.")

    df.to_csv(output_fp, index=False)
    print(f"Converted to the submission file format and saved to '{output_fp}'")


if __name__ == "__main__":
    typer.run(main)
