import os
from pathlib import Path

import pandas as pd
import torch
import typer
from datasets import load_dataset
from InstructorEmbedding import INSTRUCTOR
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich import print
from rich.progress import track
from sentence_transformers import util


TASK_A = "A"
TASK_B = "B"
TASK_C = "C"
TASKS = [TASK_A, TASK_B, TASK_C]

RUN_1 = "1"
RUN_2 = "2"
RUN_3 = "3"
RUNS = [RUN_1, RUN_2, RUN_3]

# These are all related to the output files
ENCOUNTER_ID_COL = "encounter_id"
TEST_ID = "TestID"
SYSTEM_OUTPUT = "SystemOutput"
TEAM_NAME = "wanglab"


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def main(
    train_fp: str = typer.Argument("Filepath (or URL) to the train set (should be a CSV file)."),
    test_fp: str = typer.Argument("Filepath (or URL) to the test set (should be a CSV file)."),
    output_dir: str = typer.Argument("Path to the directory where predictions will be written."),
    task: str = typer.Option(TASK_B, help=f"Task name. Should be one of {TASKS}."),
    run: str = typer.Option(RUN_1, help=f"Which challenge run to produce predictions for. Should be one of {RUNS}"),
):
    """Generates predictions using LangChain for the given task and run on the given test set.

    Example usage:
    OPENAI_API_KEY="..." \
        python scripts/run_langchain.py "./data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-TrainingSet.csv" \
        "./data/MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset4participants_inputConversations.csv" \
        "./outputs/wanglab/taskB/run1" \
        --task "B" \
        --run "1"
    """

    if task not in TASKS:
        raise ValueError(f"Task should be one of {TASKS}.")
    if run not in RUNS:
        raise ValueError(f"Run should be one of {RUNS}.")

    # Load the dataset
    train = load_dataset(
        "csv",
        data_files={
            "train": train_fp,
        },
    )["train"]
    test = load_dataset(
        "csv",
        data_files={
            "test": test_fp,
        },
    )["test"]
    ############################################# DO NOT CHANGE ABOVE #############################################

    # Setup the LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, presence_penalty=0.5, max_tokens=2_500)

    if task == TASK_B:
        prompt = PromptTemplate(
            input_variables=[
                "example_dialogue",
                "example_note",
                "dialogue",
            ],
            template="""Summarize the following patient-doctor dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies. Follow the format of the examples below.

Example Dialogue: {example_dialogue}
Example Note: {example_note}

Dialogue: {dialogue}
Note:
            """,
        )
    else:
        raise NotImplementedError(f"Task {task} is not implemented yet.")

    # Setup the chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Retrieve the top-k most similar dialogues as the in-context examples
    print("Retrieving the most similar dialogues as the in-context examples...")
    embedder = INSTRUCTOR("hkunlp/instructor-large")
    queries = embedder.encode(
        [
            ["Represent the Medicine dialogue for clustering:", f"dataset: {dataset} dialogue: {dialogue}"]
            for dataset, dialogue in zip(test["dataset"], test["dialogue"])
        ],
        show_progress_bar=True,
    )
    dialogues = embedder.encode(
        [
            ["Represent the Medicine dialogue for clustering:", f"dataset: {dataset} dialogue: {dialogue}"]
            for dataset, dialogue in zip(train["dataset"], train["dialogue"])
        ],
        show_progress_bar=True,
    )
    scores = util.cos_sim(queries, dialogues)
    _, top_k_indices = torch.topk(scores, k=1, dim=1, sorted=True)

    print("Example prompt:")
    print(
        prompt.format(
            example_dialogue=train["dialogue"][top_k_indices[0][0]],
            example_note=train["note"][top_k_indices[0][0]],
            dialogue=test["dialogue"][0],
        )
    )

    predictions = []
    for dialogue, indices in track(
        zip(test["dialogue"], top_k_indices),
        description="Generating predictions with LangChain",
        total=len(test["dialogue"]),
    ):
        # Grab the in-context examples
        example_dialogue, example_note = train["dialogue"][indices[0]], train["note"][indices[0]]

        # Run the chain
        prediction = chain.run(
            dialogue=dialogue,
            example_dialogue=example_dialogue,
            example_note=example_note,
        )
        predictions.append(prediction)

    ############################################# DO NOT CHANGE BELOW #############################################
    if task == TASK_B:
        ct_output = {TEST_ID: test[ENCOUNTER_ID_COL], SYSTEM_OUTPUT: predictions}
    else:
        raise NotImplementedError(f"Task {task} is not implemented yet.")

    # Save outputs to disk
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write the predictions to a simple text file
    preds_text_fn = f"task{task}_{TEAM_NAME}_run{run}.txt"
    (output_dir / preds_text_fn).write_text("\n".join([sanitize_text(pred) for pred in predictions]))

    # Write the predictions to a CSV file that conforms to the challenge format
    ct_fn = f"task{task}_{TEAM_NAME}_run{run}.csv"
    ct_fp = os.path.join(output_dir, ct_fn)
    pd.DataFrame.from_dict(ct_output).to_csv(ct_fp, index=False)


if __name__ == "__main__":
    typer.run(main)
