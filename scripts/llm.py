import os
from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


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
    test_fp: str = typer.Argument("Filepath to the test set (should be a CSV file)."),
    output_dir: str = typer.Argument("Path to the directory where predictions will be written."),
    task: str = typer.Option(TASK_A, help=f"Task name. Should be one of {TASKS}."),
    run: str = typer.Option(RUN_1, help=f"Which challenge run to produce predictions for. Should be one of {RUNS}"),
):
    """Generates predictions using LangChain for the given task and run on the given test set.

    Example usage:
    OPENAI_API_KEY="..." python scripts/llm.py "./taskB_testset4participants_inputConversations.csv" "./outputs/wanglab/taskB/run1" --task B --run 1
    """

    if task not in TASKS:
        raise ValueError(f"Task should be one of {TASKS}.")
    if run not in RUNS:
        raise ValueError(f"Run should be one of {RUNS}.")

    # Load the dataset
    dataset = load_dataset(
        "csv",
        data_files={
            "test": test_fp,
        },
    )

    # Setup the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    if task == TASK_A:
        raise NotImplementedError("Task A is not implemented yet.")

    else:
        prompt = PromptTemplate(
            input_variables=["dialogue"],
            template="""Summarize the following patient-doctor dialogue. Include all medically relevant information, including family history, diagnosis, past medical (and surgical) history, immunizations, lab results and known allergies:

            Dialogue: {dialogue}
            Note:
            """,
        )

    # embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # queries = dataset["test"]["dialogue"]
    # dialogues = dataset["train"]["dialogue"] + dataset["validation"]["dialogue"]

    # queries = embedder.encode(dataset["test"]["dialogue"], convert_to_tensor=True)
    # dialogues = embedder.encode(
    #     dataset["train"]["dialogue"] + dataset["validation"]["dialogue"], convert_to_tensor=True
    # )
    # scores = util.cos_sim(queries, dialogues)
    # torch.topk(scores, k=in_context_examples)

    # Setup the chain
    chain = LLMChain(llm=llm, prompt=prompt)

    predictions = []
    for dialogue in dataset["test"]["dialogue"]:
        prediction = chain.run(dialogue=dialogue)
        prediction = sanitize_text(prediction)
        predictions.append(prediction)

    if task == TASK_A:
        raise NotImplementedError("Task A is not implemented yet.")
    else:
        ct_output = {TEST_ID: dataset["test"][ENCOUNTER_ID_COL], SYSTEM_OUTPUT: predictions}

    # Save outputs to disk
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write the predictions to a simple text file
    preds_text_fn = f"task{task}_{TEAM_NAME}_run{run}.txt"
    (output_dir / preds_text_fn).write_text("\n".join(predictions))

    # Write the predictions to a CSV file that conforms to the challenge format
    ct_fn = f"task{task}_{TEAM_NAME}_run{run}.csv"
    ct_fp = os.path.join(output_dir, ct_fn)
    pd.DataFrame.from_dict(ct_output).to_csv(ct_fp, index=False)


if __name__ == "__main__":
    typer.run(main)
