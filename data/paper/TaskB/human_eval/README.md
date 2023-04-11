# Human Evaluation

This directory contains all the resources used in the human evaluation.

Two files contain the model predictions from the best-performing model on the validation set, one for the fine-tuning-based approach and one for the in-context learning-based approach:

- `led_large_pubmed_run1_postprocess.csv`: contains the predictions from `led-large-16384-pubmed`, run 1.
- `gpt_4_0314_k_3_run3.csv`: contains the predictions from `gpt_4_0314`, run 3, with the following strategy:
    - `k`: `3`
    - `retrieve_similar`: `True`
    - `include_dialogue`: `True`
    - `filter_by_dataset`: `True`

The `human_eval.tsv` file contains the dialogues and randomly shuffled clinical notes that were presented to the human annotators. The `systems_key.tsv` can be used to determine which system generated each note. Both files can be re-created with `scripts/prepare_human_eval.py`.