from flash import Trainer
from flash.text import SummarizationData, SummarizationTask
from datasets import load_dataset, DatasetDict, Dataset
from pytorch_lightning import seed_everything
from pprint import pprint
from transformers import AutoTokenizer

# Choose the dataset and model
DATASET = "bigbio/meqsum"
MODEL = "google/flan-t5-large"

# Set all seeds
SEED = 42
seed_everything(SEED, workers=True)

# Simple helper function to clean up the input text
def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


# Setup the dataset
dataset = load_dataset(DATASET, name="meqsum_bigbio_t2t", split="train")
# Format examples as natural language instructions to do instruction tuning
instruction_template = "You are a helpful medical knowledge assistant. Accurately and succinctly summarize the given user's medical question. Question: {} Summarized:"
dataset = dataset.map(
    lambda x: {
        "text_1": f"{sanitize_text(instruction_template.format(x['text_1']))}",
        "text_2": x["text_2"],
    }
)
# meqsum does not have validation/test splits, so create them here
# Following example from: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/2
train_test_valid = dataset.train_test_split(test_size=0.1, seed=SEED, shuffle=True)
test_valid = train_test_valid["test"].train_test_split(test_size=0.5, seed=SEED)
dataset = DatasetDict(
    {
        "train": train_test_valid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"],
    }
)

# Create the DataModule
datamodule = SummarizationData.from_hf_datasets(
    input_field="text_1",
    target_field="text_2",
    # Lightning expects a Dataset object, not a DatasetDict
    train_hf_dataset=Dataset(dataset["train"].data),
    val_hf_dataset=Dataset(dataset["valid"].data),
    test_hf_dataset=Dataset(dataset["test"].data),
    batch_size=6,
)

# Setup the model
# first, find the max lengths of the source and target sequences to set the max lengths appropriately
tokenizer = AutoTokenizer.from_pretrained(MODEL)
max_source_length = max(
    len(tokenizer(example["text_1"])["input_ids"]) for example in dataset["train"]
)
max_target_length = max(
    len(tokenizer(example["text_2"])["input_ids"]) for example in dataset["train"]
)
model = SummarizationTask(
    backbone=MODEL,
    learning_rate=5e-4,
    max_source_length=max_source_length,
    max_target_length=max_target_length,
)

# Finally, create the trainer
trainer = Trainer(
    accelerator="gpu",
    devices="auto",
    # Some models (e.g. Flan-T5) require a higher precision
    precision=32 if "flan-t5" in MODEL else 16,
    max_epochs=3,
)

if __name__ == "__main__":
    # Fine-tune the model
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # Sanity check the model by looking at some predictions
    datamodule = SummarizationData.from_lists(
        predict_data=dataset["test"]["text_1"][:5],
        batch_size=5,
    )
    predictions = trainer.predict(model, datamodule=datamodule)

    for input_, target, pred in zip(dataset['test']['text_1'][:5], dataset['test']['text_2'][:5], predictions[0]):
        print(f"Input: {input_}")
        print(f"Target: {target}")
        print(f"Prediction: {pred}")

    # Lastly, saved the tuned model
    checkpoint_name = f"{DATASET.replace('/', '_')}_{MODEL.replace('/', '_').replace('-', '_')}.pt"
    trainer.save_checkpoint(checkpoint_name)
