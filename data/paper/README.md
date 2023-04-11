# Data Artifacts

This directory contains all model predictions and evaluations. 

At the top level, the directory structure is divided into tasks `TaskA` and `TaskB`.

## TaskA

TODO

## TaskB

- `predictions` contains the model outputs for three runs
- `results` contains the metrics after evaluating the outputs of each run
- `predictions` and `results` are further divided by approach, into `fine-tuning` and `in-context-learning`
- `in-context-learning` is further divided according to the ablation into `filtered` and `unfiltered` and then `random` and `similar`, and finally `note_only` and `dialogue_note`
- `human_eval` contains all the resources used in the human evaluation (see `human_eval/README.md` for more details)
