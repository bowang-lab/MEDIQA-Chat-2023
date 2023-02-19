# mediqa-chat-tasks-acl-2023

A repository organizing our submission to the MEDIQA-Chat Tasks @ ACL-ClinicalNLP 2023

## Installation

Requires python>=3.7. First, create and activate a virtual environment, then install the requirements:

```bash
pip install -r requirements.txt
```

For setup on a cluster managed by the [Alliance](https://alliancecan.ca/en/services/advanced-research-computing), please see [`./scripts/slurm/setup_on_arc.sh`](./scripts/slurm/setup_on_arc.sh).

## Usage

### Task A

The model for Task A can be trained with the [`train_task_a.py`](./scripts/train_task_a.py) script. To see all available options, run:

```bash
python ./scripts/train_task_a.py --help
```

Arguments can be modified in the [config files](./conf/), or passed as command line arguments. To train the model, run:

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/task_a.yml" output_dir="./output/task_a"
```

We also provide a SLURM submission script for ARC clusters, which can be found at [`./scripts/slurm/train_task_a.sh`](./scripts/slurm/train_task_a.sh).

### Task B

TODO

### Task C

TODO

