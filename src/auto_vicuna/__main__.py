"""
    Main entry point for the auto_vicuna package.
"""
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option(
    "--vicuna_weights",
    type=click.Path(exists=True),
    default=lambda: os.environ.get("VICUNA_WEIGHTS", ""),
)
def main(vicuna_weights: Path):
    """Auto-Vicuna: A Python package for automatically generating Vicuna randomness."""
    click.echo(f"Auto-Vicuna\n===========\nVersion: 0.0.1\nWeights: {vicuna_weights}")


if __name__ == "__main__":
    main(sys.argv[1:])
