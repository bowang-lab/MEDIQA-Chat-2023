import re

import evaluate
import nltk
import numpy as np
import typer
from datasets import load_dataset


GENHX = "GENHX"
TASK_A = "A"
TASK_B = "B"
TASK_C = "C"
TASKS = [TASK_A, TASK_B, TASK_C]

# These are all related to the output files
ENCOUNTER_ID_COL = "encounter_id"
TEST_ID = "TestID"
SYSTEM_OUTPUT = "SystemOutput"


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def postprocess_text(preds, labels):
    preds = [sanitize_text(pred) for pred in preds]
    labels = [sanitize_text(label) for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def extract_header_and_text(texts):
    """Extracts section header and section text predictions from model outputs and targets."""
    section_headers, section_texts = [], []
    for text in texts:
        # Extract from the model predictions and the labels the section headers and the section texts
        section_header = re.findall(r"(?:Section header:)(.*)(?:Section text)", text, re.DOTALL)
        section_text = re.findall(r"(?:Section text:)(.*)", text, re.DOTALL)
        # It is possible the mdoel does not format its outputs as expected. In this case, take the section
        # header to be GENHX (the most likely section header) and the section text to be the whole text.
        section_header = section_header[0].strip() if section_header else GENHX
        section_text = section_text[0].strip() if section_text else text.strip()
        section_headers.append(section_header)
        section_texts.append(section_text)
    return section_texts, section_headers


def main(
    predictions_fp: str = typer.Argument(
        "Filepath (or URL) to the predictions, should be a CSV file that follows this format: https://github.com/abachaa/MEDIQA-Chat-2023#submission-instructions"
    ),
    references_fp: str = typer.Argument(
        "Filepath (or URL) to the references, should be a CSV file in the same format as the challenge data."
    ),
    task: str = typer.Option(TASK_A, help=f"Task name. Should be one of {TASKS}."),
    cache_dir: str = typer.Option("Path to the directory where metrics will be cached."),
):
    """Evaluates the predictions against the references.

    Example usage:
    python ./scripts/evaluate_notes.py "./outputs/taskB_wanglab_run1.csv" \
        "./data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-ValidationSet.csv" \
        --task "B"
    """
    if task not in TASKS:
        raise ValueError(f"Task should be one of {TASKS}.")

    predictions = load_dataset(
        "csv",
        data_files={
            "train": predictions_fp,
        },
    )["train"]

    references = load_dataset(
        "csv",
        data_files={
            "train": references_fp,
        },
    )["train"]

    if predictions[TEST_ID] != references[ENCOUNTER_ID_COL]:
        raise ValueError(f"Prediction IDs do not match reference IDs.")

    rouge = evaluate.load("rouge", cache_dir=cache_dir)
    bertscore = evaluate.load("bertscore", cache_dir=cache_dir)
    bleurt = evaluate.load("bleurt", "BLEURT-20-D12", cache_dir=cache_dir)

    result = {}

    # Lightly postprocess the text
    predictions, references = postprocess_text(
        predictions[SYSTEM_OUTPUT], references["section_text" if task == TASK_A else "note"]
    )

    # If this is task A, we also have to include section header prediction
    if task == TASK_A:
        exact_match = evaluate.load("exact_match")
        predictions, predicted_headers = extract_header_and_text(predictions)
        references, reference_headers = extract_header_and_text(references)
        result.update(exact_match.compute(predictions=predicted_headers, references=reference_headers))

    rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    result.update(rouge_results)

    # Compute the arithmetic mean of ROUGE-1, ROUGE-2 and ROUGE-L following: https://arxiv.org/abs/2110.08499
    result["rouge_avg"] = np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]).item()

    bertscore_result = bertscore.compute(
        predictions=predictions,
        references=references,
        batch_size=4,
        # These are mostly based on the recommendations in https://github.com/Tiiiger/bert_score
        model_type="microsoft/deberta-large-mnli",
        lang="en",
        rescale_with_baseline=True,
        use_fast_tokenizer=True,
    )
    result.update(
        {
            "bertscore_p": np.mean(bertscore_result["precision"]).item(),
            "bertscore_r": np.mean(bertscore_result["recall"]).item(),
            "bertscore_f1": np.mean(bertscore_result["f1"]).item(),
        }
    )

    bleurt_result = bleurt.compute(predictions=predictions, references=references)
    result.update({"bleurt": np.mean(bleurt_result["scores"]).item()})

    result["ensemble_gen_score"] = np.mean([result["rouge1"], result["bertscore_f1"], result["bleurt"]]).item()

    if "exact_match" in result:
        result["ensemble_score"] = np.mean([result["ensemble_gen_score"], result["exact_match"]]).item()

    result = {k: round(v * 100, 4) for k, v in result.items()}

    print(result)


if __name__ == "__main__":
    typer.run(main)
