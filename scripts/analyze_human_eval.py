from pathlib import Path

import pandas as pd
import typer


GT = "GT"
FT = "FT"
ICL = "ICL"
MODEL_MODEL_TIE = "model-model tie"
HUMAN_MODEL_TIE = "human-model tie"

OUTPUT_FN = "human_eval_results.tsv"


def _resolve_ties(system_a: str, system_b: str) -> str:
    if sorted([system_a, system_b]) == [FT, ICL]:
        return MODEL_MODEL_TIE
    else:
        return HUMAN_MODEL_TIE


def main(
    annotations_dir: str = typer.Argument(
        ..., help="Location to the directory containing the human annotation results. Expect .tsv files."
    ),
    systems_key_fp: str = typer.Argument(
        ..., help="Location to the directory containing the systems key. Expect a .tsv file."
    ),
    output_dir: str = typer.Argument(..., help=f"Directory to save results of the human evaluation ({OUTPUT_FN})"),
) -> None:
    """Analyzes the human evaluation results.

    Example usage:
    python scripts/analyze_human_eval.py "data/paper/TaskB/human_eval/annotations" \
        "data/paper/TaskB/human_eval/systems_key.tsv" \
        "data/paper/TaskB/human_eval"
    """
    results = {}

    # Load the model key
    systems_key_df = pd.read_csv(systems_key_fp, sep="\t")

    # Load the human annotations
    annotator_df = {"example_id": systems_key_df.example_id.tolist()}
    for annotator_file in Path(annotations_dir).glob("*.tsv"):
        # The first two lines are instructions, and the last column is annotator comments, skip
        ann_df = pd.read_csv(annotator_file, sep="\t", skiprows=2, usecols=list(range(6)))
        # Check that the example IDs match
        if not systems_key_df.example_id.equals(ann_df.example_id):
            raise ValueError("Example IDs in the systems key and the annotations do not match.")
        annotator_df[annotator_file.stem] = ann_df["preferred"]
    annotator_df = pd.DataFrame(annotator_df)

    # Merge the two dataframes
    annotations_df = systems_key_df.merge(annotator_df, on="example_id")

    # Unblind the human annotations
    annotators = list(annotator_df.columns)[1:]
    for annotator in annotators:
        for i, preference in enumerate(annotations_df[annotator]):
            if "/" in preference:
                system_a = annotations_df.loc[i, f"clinical note {preference.split('/')[0]}"]
                system_b = annotations_df.loc[i, f"clinical note {preference.split('/')[1]}"]
                annotations_df.loc[i, annotator] = _resolve_ties(system_a, system_b)
            else:
                annotations_df.loc[i, annotator] = annotations_df.loc[i, f"clinical note {preference}"]

        # Count the number of times each system was preferred
        system_counts = annotations_df[annotator].value_counts()

        # Collect all counts for each system, ties, and win rates
        gt_preferred, ft_preferred, icl_preferred = (
            system_counts.get(GT, 0),
            system_counts.get(FT, 0),
            system_counts.get(ICL, 0),
        )
        model_model_ties = system_counts.get(MODEL_MODEL_TIE, 0)
        human_model_ties = system_counts.get(HUMAN_MODEL_TIE, 0)
        ties = model_model_ties + human_model_ties
        counts_excluding_ties = len(annotations_df) - ties

        results[annotator] = {
            f"{GT}_preferred": gt_preferred,
            f"{FT}_preferred": ft_preferred,
            f"{ICL}_preferred": icl_preferred,
            MODEL_MODEL_TIE: model_model_ties,
            HUMAN_MODEL_TIE: human_model_ties,
            f"{GT}_win_rate": gt_preferred / counts_excluding_ties,
            f"{FT}_win_rate": ft_preferred / counts_excluding_ties,
            f"{ICL}_win_rate": icl_preferred / counts_excluding_ties,
        }

    results_df = pd.DataFrame(results)
    # Save outputs to disk
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write the counts and win rates to disk
    results_df.to_csv(output_dir / OUTPUT_FN)


if __name__ == "__main__":
    typer.run(main)
