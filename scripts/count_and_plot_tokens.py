from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tiktoken
import typer
from datasets import load_dataset


TASK_A = "TaskA"
TASK_B = "TaskB"
TASK_C = "TaskC"
TASKS = [TASK_A, TASK_B, TASK_C]

TRAIN_SET_FN = "{}-TrainingSet.csv"
VALIDATION_SET_FN = "{}-ValidationSet.csv"

TOKEN_COUNTS_FN = "token_counts.csv"


def main(
    shared_task_train_val_dir: str = typer.Argument("Path to the directory containing the shared task data"),
    output_dir: str = typer.Argument("Path to the directory to save the token counts CSV file and resulting plot."),
    model: str = typer.Option("gpt-4", help="We use the tokenizer of this OpenAI model compute token counts"),
):
    """Computes the number of tokens in the dialogues and notes for each example in the shared task subtasks. Saves
    a resulting CSV and plot to the specified output directory.

    Example usage:

    python scripts/count_tokens.py "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023" "./output/token_counts"
    """
    encoding = tiktoken.encoding_for_model(model)

    token_counts = {"task": [], "partition": [], "num_dialogue_tokens": [], "num_note_tokens": []}

    for task in TASKS:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(Path(shared_task_train_val_dir) / task / TRAIN_SET_FN.format(task)),
                "validation": str(Path(shared_task_train_val_dir) / task / VALIDATION_SET_FN.format(task)),
            },
        )

        note_column = "section_text" if task == TASK_A else "note"

        for partition in dataset:
            dataset[partition] = dataset[partition].map(lambda x: {"dialogue_tokens": encoding.encode(x["dialogue"])})
            dataset[partition] = dataset[partition].map(lambda x: {"note_tokens": encoding.encode(x[note_column])})

            token_counts["task"].extend([task] * len(dataset[partition]))
            token_counts["partition"].extend([partition] * len(dataset[partition]))
            token_counts["num_dialogue_tokens"].extend(
                [len(tokens) for tokens in dataset[partition]["dialogue_tokens"]]
            )
            token_counts["num_note_tokens"].extend([len(tokens) for tokens in dataset[partition]["note_tokens"]])

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save token counts to CSV
    df = pd.DataFrame.from_dict(token_counts)
    df.to_csv(output_dir / TOKEN_COUNTS_FN, index=False)

    # Plot token counts for each task
    for task in TASKS:
        task_df = df[df.task == task]
        task_df = task_df.rename(columns={"num_dialogue_tokens": "dialogues", "num_note_tokens": "notes"})
        plt.clf()
        title = f"Subtask {task.split('Task')[-1]} Token Lengths"
        sns.histplot(data=task_df).set(title=title, xlabel="Token Length", ylabel="")
        plt.savefig(output_dir / f"token_lengths_{task}.png", facecolor="white", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    typer.run(main)
