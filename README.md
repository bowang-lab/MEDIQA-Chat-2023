[![ci](https://github.com/bowang-lab/mediqa-chat-tasks-acl-2023/actions/workflows/ci.yml/badge.svg)](https://github.com/bowang-lab/mediqa-chat-tasks-acl-2023/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# MEDIQA-Chat-2023-WangLab

This repository contains our submission (and the resulting short paper) to the MEDIQA-Chat Shared Task @ ACL-ClinicalNLP 2023

## Table of contents

- [MEDIQA-Chat-2023-WangLab](#mediqa-chat-2023-wanglab)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Fine-tuning a model on the shared task data](#fine-tuning-a-model-on-the-shared-task-data)
      - [Training](#training)
      - [Validation](#validation)
      - [Testing](#testing)
    - [Generate notes with LangChain](#generate-notes-with-langchain)
  - [Pre-trained models, outputs and results](#pre-trained-models-outputs-and-results)
  - [Submitting to the shared task](#submitting-to-the-shared-task)

## Installation

Requires python>=3.8. First, create and activate a virtual environment, then install the requirements:

```bash
pip install -r requirements.txt
```

> __Note__: For setup on a cluster managed by the [Alliance](https://alliancecan.ca/en/services/advanced-research-computing), please see [`./scripts/slurm/setup_on_arc.sh`](./scripts/slurm/setup_on_arc.sh).

## Usage

### Fine-tuning a model on the shared task data

Models can be fine-tuned on the shared task data using the [`run_summarization.py`](./scripts/run_summarization.py) script, which is adapted from the HuggingFace [`run_summarization.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py) script. To see all available options, run:

```bash
python ./scripts/run_summarization.py --help
```

Arguments can be modified in the [config files](./conf/) or passed as command-line arguments. Valid arguments are anything from the HuggingFace [`TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments), [`Seq2SeqTrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments) or arguments specified in the script itself. At a minimum, you must provide a path to the dataset partitions with `train_file`, `validation_file` and, optionally, `test_file`.

#### Training

To train the model, run one of the following:

```bash
# Task A (train)
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskA.yml" \
    output_dir="./output/taskA"
```

```
# Task B (train)
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskB.yml" \
    output_dir="./output/taskB"
```

> __Note__: `base.yml` contains good default arguments that should be used for all experiments. `taskA.yml`/`taskB.yml` contain arguments specific to Task A/B. Arguments passed via the command line arguments will override those in the config files.

#### Validation

To evaluate a trained model on the validation set, run one of the following:

```bash
# Task A
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskA.yml" \
    output_dir="./output/taskA/fine_tune" \
    model_name_or_path="./path/to/model/checkpoint" \
    do_train=False \
    do_eval=True
```

```
# Task B
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/taskB.yml" \
    output_dir="./output/taskB/fine_tune" \
    model_name_or_path="./path/to/model/checkpoint" \
    do_train=False \
    do_eval=True
```

#### Testing

To make predictions with a trained model on the test set, see the [Submission](#submission).

---

By default, the model will be evaluated by ROUGE, BERTScore and BLEURT. You can change the underlying models for BERTScore and BLEURT by modifying the `bertscore_model_type` and `bleurt_checkpoint` arguments. We choose reasonable defaults here, which balance model size and evaluation time with automatic metric performance. For more information on possible models and metric performance, see [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?usp=sharing) for BERTScore and [here](https://github.com/google-research/bleurt/blob/master/checkpoints.md) for BLEURT.

Results will be automatically logged to any integrations that are _installed_ and _supported_ by the [HuggingFace trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.report_to). If `do_predict=True`, a file which contains the model's predictions formatted for submission to the challenge task will be saved to `output_dir / "taskX_wanglab_runY.csv"`. `X` corresponds to the script argument `task` and `Y` to the script argument `run`.

> We also provide a SLURM submission script for ARC clusters, which can be found at [`./scripts/slurm/run_summarization.sh`](./scripts/slurm/run_summarization.sh).

### Generate notes with LangChain

To generate notes with a large language model (LLM, via LangChain), use the [`run_langchain.py`](./scripts/run_langchain.py) script. To see all available options, run:

```bash
python ./scripts/run_langchain.py --help
```

To reproduce our best results for Task B, run the following:

```bash
# Task B
OPENAI_API_KEY="..." python scripts/run_langchain.py \
    "./MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset4participants_inputConversations.csv" \
    "./output/taskB/in_context_learning" \
    --train-fp "./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-TrainingSet.csv" \
    --task "B" \
    --run "1"
```

You will need to provide your own `OPENAI_API_KEY`.

> __Note__: Due to the non-deterministic nature of OpenAI's models and API, results may vary slightly from our reported results.

## Pre-trained models, outputs and results

All model outputs and results (as well as data from the human evaluation) reported in our paper are available in the [data/paper](./data/paper) directory.

## Submitting to the shared task

To submit a run to the shared task we used the following commands:

```bash
./scripts/submission/install.sh
./scripts/submission/activate.sh
# Then, choose one of the decode scripts, e.g.
./scripts/submission/decode_taskA_run1.sh
```

The submission scripts also demonstrate how to make predictions on the test set using a trained model.
