from datasets import load_dataset

taskA = load_dataset("csv", data_files={"train": "/home/johnmg/projects/def-wanglab-ab/johnmg/mediqa-chat-tasks-acl-2023/data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv", "validation": "/home/johnmg/projects/def-wanglab-ab/johnmg/mediqa-chat-tasks-acl-2023/data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv"})

def combine_section_header_and_text(example):
    example["target"] = f'Section header: {example["section_header"]} Section text: {example["section_text"]}'
    return example

taskA["train"] = taskA["train"].map(combine_section_header_and_text)
taskA["validation"] = taskA["validation"].map(combine_section_header_and_text)

# Dataset(taskA).to_csv("taskA.csv")
taskA["train"].to_csv("taskA_train.csv")
taskA["validation"].to_csv("taskA_validation.csv")