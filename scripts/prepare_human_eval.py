import random
from pathlib import Path

import pandas as pd
import typer
from rich import print
from rich.progress import track


RANDOM_SEED = 42
SYSTEMS = ["ground truth", "fine-tuned model", "in-context learning model"]
HUMAN_EVAL_FN = "human_eval.tsv"
SYSTEMS_KEY_FN = "systems_key.tsv"


def main(
    validation_fp: str = typer.Argument(..., help="Filepath (or URL) to the validation set (should be a CSV file)."),
    fine_tuned_model_fp: str = typer.Argument(
        ..., help="Filepath to the fine-tuned model output file (should be a CSV file)."
    ),
    icl_model_fp: str = typer.Argument(
        ..., help="Filepath to in-context learning model output file (should be a CSV file)."
    ),
    output_dir: str = typer.Argument(
        ..., help=f"Directory to save the human evaluation ({HUMAN_EVAL_FN}) and systems key ({SYSTEMS_KEY_FN}) files."
    ),
):
    """Builds the human evaluation file and systems key for the shared task. Requires the validation set, the
    fine-tuned model outputs and the in-context learning model outputs. Saves the human evaluation file and the
    systems key to the specified output directory.

    Example usage:

    python scripts/prepare_human_eval.py "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-ValidationSet.csv" \
        "data/paper/TaskB/human_eval/led_large_pubmed_run1_postprocess.csv" \
        "data/paper/TaskB/human_eval/gpt_4_0314_k_3_run3.csv" \
        "data/paper/TaskB/human_eval"
    """

    val_df = pd.read_csv(validation_fp)
    print(f"Loaded validation set from: '{validation_fp}'")
    ft_df = pd.read_csv(fine_tuned_model_fp)
    print(f"Loaded fine-tuned model output from: '{fine_tuned_model_fp}'")
    icl_df = pd.read_csv(icl_model_fp)
    print(f"Loaded in-context learning model output from: '{icl_model_fp}'")

    # Check that all the TestIDs match
    if not val_df.encounter_id.equals(ft_df.TestID) or not val_df.encounter_id.equals(icl_df.TestID):
        raise ValueError("Test IDs in the validation set and the model outputs do not match.")

    human_eval = {
        "example_id": val_df.encounter_id.tolist(),
        "doctor-patient dialogue": val_df.dialogue.tolist(),
        "clinical note A": [],
        "clinical note B": [],
        "clinical note C": [],
    }
    systems_key = {
        "example_id": val_df.encounter_id.tolist(),
        "clinical note A": [],
        "clinical note B": [],
        "clinical note C": [],
    }
    rng = random.Random(RANDOM_SEED)
    for gt_note, ft_note, icl_note in track(
        zip(val_df.note, ft_df.SystemOutput, icl_df.SystemOutput),
        description="Building human evaluation file and systems key",
        total=len(val_df),
    ):
        notes = [gt_note, ft_note, icl_note]
        # Shuffle the order of the clinical notes
        indices = rng.sample(range(3), k=3)
        # Build up the human evaluation file
        human_eval["clinical note A"].append(notes[indices[0]])
        human_eval["clinical note B"].append(notes[indices[1]])
        human_eval["clinical note C"].append(notes[indices[2]])
        # Maintain a key for the systems
        systems_key["clinical note A"].append(SYSTEMS[indices[0]])
        systems_key["clinical note B"].append(SYSTEMS[indices[1]])
        systems_key["clinical note C"].append(SYSTEMS[indices[2]])

    # Save outputs to disk
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # TSV's are easier to copy paste into Google Sheets
    pd.DataFrame(human_eval).to_csv(output_dir / HUMAN_EVAL_FN, index=False, sep="\t")
    print(f"Saved human evaluation file to: '{output_dir / HUMAN_EVAL_FN}'")
    pd.DataFrame(systems_key).to_csv(output_dir / SYSTEMS_KEY_FN, index=False, sep="\t")
    print(f"Saved systems key file to: '{output_dir / SYSTEMS_KEY_FN}'")


if __name__ == "__main__":
    typer.run(main)
