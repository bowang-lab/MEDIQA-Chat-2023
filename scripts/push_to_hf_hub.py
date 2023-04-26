import typer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rich import print
from rich.status import Status


def main(
    model_name_or_path: str = typer.Argument(..., help="Path to the model directory."),
    hf_hub_name: str = typer.Argument(..., help="Name of the model once pushed to the Hugging Face Hub."),
) -> None:
    """Pushes the model at `model_name_or_path` to the Hugging Face Hub as `hf_hub_name`. Must be logged in to the Hub."""
    with Status(f"Loading model and tokenizer from {model_name_or_path}"):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    print(f"[green]Model and tokenizer loaded from {model_name_or_path}[/green]")

    with Status(f"Pushing model and tokenizer to HuggingFace Hub under {hf_hub_name} (this can take a while)"):
        tokenizer.push_to_hub(hf_hub_name)
        model.push_to_hub(hf_hub_name)
    print(f"[green]Model and tokenizer pushed to {hf_hub_name}[/green]")


if __name__ == "__main__":
    typer.run(main)
