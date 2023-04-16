# Data Artifacts

This directory contains all model predictions and evaluations. Results for all experiments and runs are also available in a Google Sheet [here](https://docs.google.com/spreadsheets/d/1u6vwZduDYfLOKsE1IJQheEGa37jvL-hMIhpU7SNvl5M/edit?usp=sharing).

At the top level, this directory is divided into tasks `TaskA` and `TaskB`.

## TaskA

- `TaskA-ValidationSet-SubmissionFormat.csv` contains the shared task validation set in the submission format for easier evaluation with `model outputs`. This file can be re-generated with `scripts/convert_to_submission_format.py`.
- `predictions` contains the model outputs for three runs
- `results` contains the metrics after evaluating the outputs of each run

## TaskB

- `TaskB-ValidationSet-SubmissionFormat.csv` contains the shared task validation set in the submission format for easier evaluation with model outputs. This file can be re-generated with `scripts/convert_to_submission_format.py`.
- `predictions` contains the model outputs for three runs
- `results` contains the metrics after evaluating the outputs of each run
- `predictions` and `results` are further divided by approach, into `fine-tuning` and `in-context-learning`
- `in-context-learning` is further divided according to the ablation into `filtered` and `unfiltered` and then `random` and `similar`, and finally `note_only` and `dialogue_note`
- `human_eval` contains all the resources used in the human evaluation (see `human_eval/README.md` for more details)
