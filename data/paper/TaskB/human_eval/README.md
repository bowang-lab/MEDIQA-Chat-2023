# Human Evaluation

This directory contains all the resources used in the human evaluation. Results for the human eval are also available in a Google sheet [here](https://docs.google.com/spreadsheets/d/1hDb5rvEGZnkHgYoXJFxtGqTswwCoxxH9kVZ3WcQadR8/edit?usp=sharing).

Two files under `best_runs` contain the model predictions from the best-performing model on the validation set, one for the fine-tuning-based approach and one for the in-context learning-based approach:

- `led_large_pubmed_run1_postprocess.csv`: contains the predictions from `led-large-16384-pubmed`, run 1 (should be identical to `data/paper/TaskB/predictions/fine_tuning/led_large_pubmed_run1_postprocess.csv`).
- `gpt_4_0314_k_3_run3.csv`: contains the predictions from `gpt_4_0314`, run 3, with the following strategy (should be identical to `data/paper/TaskB/predictions/in_context_learning/filtered/similar/note_only/gpt_4_0314_k_3_run3.csv`):
    - `k`: `3`
    - `retrieve_similar`: `True`
    - `include_dialogue`: `False`
    - `filter_by_dataset`: `True`

The `human_eval.tsv` file contains the dialogues and randomly shuffled clinical notes that were presented to the human annotators. The `systems_key.tsv` can be used to determine which system generated each note. Both files can be re-created with `scripts/prepare_human_eval.py`. The directory `annotations` contains the raw annotations from the human annotators. Finally, `human_eval_results.tsv` contains the aggregated results from the human evaluation. This file can be re-created with `scripts/analyze_human_eval.py`.