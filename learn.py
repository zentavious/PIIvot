import argparse
import os
from typing import Dict
from pathlib import Path
import torch
from rich.panel import Panel

from eedi_piivot.experiments import Experiment
from eedi_piivot.utils.immutable import global_immutable
from eedi_piivot.utils.console import console


repo_path = Path(__file__).resolve().parents[0]

RESULTS_DIRNAME = "results"

def main(args: argparse.Namespace) -> None:
    """Main function to run the training or inference.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    global_immutable.DEBUG = args.debug

    global_immutable.YES = args.yes

    global_immutable.DEVICE = (
        "cuda" if not args.use_cpu and torch.cuda.is_available() else "cpu"
    )

    if global_immutable.DEBUG:
        console.print(Panel("Running in DEBUG mode.", style="bold white on red"))
    
    if (args.use_parallelism):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    
    config_filepaths = Path(args.exp_folder).glob("*.json")

    for config_filepath in config_filepaths:
        experiment = Experiment(
            repo_path / RESULTS_DIRNAME,
            config_filepath,
            resume_checkpoint_filename=None,
        )

        experiment.run_train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_folder", type=str, required=True, help="Path to experiment folder."
    )

    
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Run in debug mode.",
    )
    parser.add_argument(
        "--use_cpu",
        default=False,
        action="store_true",
        help="Use the CPU (even if a GPU is available).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        default=False,
        action="store_true",
        help="Use 'Y' for all confirm actions.",
    )
    parser.add_argument(
        "--use_parallelism",
        default=False,
        action="store_true",
        help="Allow parallelism during training.",
    )


    args = parser.parse_args()

    main(args)