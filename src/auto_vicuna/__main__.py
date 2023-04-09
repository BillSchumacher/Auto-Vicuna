"""
    Main entry point for the auto_vicuna package.
"""
import os
import sys
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from fastchat.serve.inference import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()


@click.command()
@click.option(
    "--vicuna_weights",
    type=click.Path(exists=True),
    default=lambda: os.environ.get("VICUNA_WEIGHTS", ""),
)
@click.option("--num_gpus", type=int, default=1)
@click.option("--device", type=str, default=DEVICE)
@click.option("--debug", is_flag=True)
@click.option("--load_8bit", is_flag=True)
def main(
    vicuna_weights: Path,
    num_gpus: int = 1,
    device: torch.device = torch.device(DEVICE),
    debug: bool = False,
    load_8bit: bool = False,
) -> None:
    """Auto-Vicuna: A Python package for automatically generating Vicuna randomness."""
    click.echo(f"Auto-Vicuna\n===========\nVersion: 0.0.1\nWeights: {vicuna_weights}")
    click.echo(f"Device: {device}\nTorch version: {torch.__version__}")
    if "cpu" in torch.__version__:
        click.echo("\nError: CPU not supported. Install a GPU version of PyTorch.")
        click.echo("See https://pytorch.org/get-started/locally/ for more info.\n")
        sys.exit(1)
    if not isinstance(device, torch.device):
        device = torch.device(device)
    try:
        model = load_model(
            vicuna_weights,
            device=device,
            num_gpus=num_gpus,
            debug=debug,
            load_8bit=load_8bit,
        )
    except ValueError as _:  # noqa: F841
        click.echo(
            "Error: CPU not supported. Select a GPU device instead, e.g."
            " --device=cuda"
        )
        sys.exit(1)
    click.echo(f"Model: {model}")


if __name__ == "__main__":
    main(sys.argv[1:])
