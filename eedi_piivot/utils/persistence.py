"""A module for managing persistence of experiment results."""

import datetime
import shutil
from glob import glob
from pathlib import Path

import torch
from rich.prompt import Confirm
from rich.table import Table
from pandas import DataFrame

from eedi_piivot.utils.console import console
from eedi_piivot.utils.immutable import global_immutable

CHECKPOINT_DIRNAME = "checkpoints"
DATASET_DIRNAME = "datasets"
ERRORS_DIRNAME = "model_errors"

class Persistence:
    """A class to manage persistence of experiment results."""

    def __init__(
        self,
        results_dirpath: Path,
        candidate_exp_dirname: str,
        resume_checkpoint_filename: str | None = None,
    ):
        """Initializes the Persistence class.

        Sets the results directory and checks for the candidate experiment directory. If
        the candidate experiment directory does not exist, create it. If it does exist,
        check whether there are any checkpoints. If there are checkpoints, and the
        resume checkpoint file is not None, resume training. Otherwise, prompt the user
        to confirm whether they want to override the contents of the existing directory.

        Args:
            results_dirpath (Path): The path to the results directory which will contain
            the individual experiment directories.
            candidate_exp_dirname (str): A candidate name for the experiment directory.
            resume_checkpoint_filename (str, optional): The name of the checkpoint file
            to resume training from. Defaults to None.

            {results_dirpath}
              - {exp_dirname_1}
              - {exp_dirname_2}
        """
        self.results_dirpath = results_dirpath
        self.resume_checkpoint_filename = resume_checkpoint_filename
        self.exp_dirname = self.get_or_set_experiment_directory(candidate_exp_dirname)

    def get_or_set_experiment_directory(self, candidate_exp_dirname: str) -> str:
        """Get or set the results directory.

        Args:
            candidate_exp_dirname (str): A candidate name for the experiment directory.

        Returns:
            str: The actual name of the experiment directory.
        """
        candidate_exp_dirpath = self.results_dirpath / candidate_exp_dirname

        # Case 1: Experiment directory does not exist so create it.
        if not candidate_exp_dirpath.exists():
            console.print(
                f"Creating results directory for experiment [bold green]{candidate_exp_dirname}[/bold green]."
            )
            candidate_exp_dirpath.mkdir(parents=True, exist_ok=True)

        # Case 2: Resume checkpoint exists so resume training.
        elif self.resume_checkpoint_filename:
            resume_checkpoint_filepath = (
                candidate_exp_dirpath
                / CHECKPOINT_DIRNAME
                / self.resume_checkpoint_filename
            )
            if resume_checkpoint_filepath.exists():
                console.print(
                    f"Resuming experiment [bold green]{candidate_exp_dirname}[/bold green] from checkpoint {self.resume_checkpoint_filename}."
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint file {self.resume_checkpoint_filename} not found in {candidate_exp_dirpath}."
                )

        else:
            checkpoint_filepaths = glob(
                str(candidate_exp_dirpath / CHECKPOINT_DIRNAME / "*.pt")
            )

            if checkpoint_filepaths:
                checkpoint_filenames = [
                    Path(checkpoint_filepath).name
                    for checkpoint_filepath in checkpoint_filepaths
                ]
                console.print(
                    f"[bold yellow]Found existing checkpoints for experiment {candidate_exp_dirname}[/bold yellow]:"
                )
                table = Table(title="Existing checkpoints")

                table.add_column("Index", justify="center")
                table.add_column("Filename", justify="left")

                for idx, filename in enumerate(checkpoint_filenames, start=1):
                    table.add_row(str(idx), filename)

                console.print(table)

            if global_immutable.YES or Confirm.ask(
                "Do you want to override the contents of the existing directory?",
                default=False,
            ):
                # Case 3: User wants to override the existing directory.
                console.print(
                    f"Emptying results directory for experiment [bold green]{candidate_exp_dirname}[/bold green]."
                )
                shutil.rmtree(candidate_exp_dirpath)
                candidate_exp_dirpath.mkdir(parents=True, exist_ok=True)
            
            else:
                # Case 4: User does not want to override the existing directory so create a new one.
                candidate_exp_dirname = (
                    f"{candidate_exp_dirname}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
                )
                console.print(
                    f"Creating new results directory for experiment [bold green]{candidate_exp_dirname}[/bold green]."
                )
                candidate_exp_dirpath = self.results_dirpath / candidate_exp_dirname
                candidate_exp_dirpath.mkdir(parents=True, exist_ok=True)

        # Add within results dirs if it doesn't exist
        if not (self.results_dirpath / candidate_exp_dirpath / CHECKPOINT_DIRNAME).exists():
            (self.results_dirpath / candidate_exp_dirpath / CHECKPOINT_DIRNAME).mkdir(parents=True, exist_ok=True)
        if not (self.results_dirpath / candidate_exp_dirpath / DATASET_DIRNAME).exists():
            (self.results_dirpath / candidate_exp_dirpath / DATASET_DIRNAME).mkdir(parents=True, exist_ok=True)
        if not (self.results_dirpath / candidate_exp_dirpath / ERRORS_DIRNAME).exists():
            (self.results_dirpath / candidate_exp_dirpath / ERRORS_DIRNAME).mkdir(parents=True, exist_ok=True)

        return candidate_exp_dirpath
    
    def save_errors(
        self,
        split_name,
        errors: DataFrame,
    ):
        errors_filepath = (
            self.results_dirpath
            / self.exp_dirname
            / ERRORS_DIRNAME
            / f"{split_name}_errors.csv"
        )
        errors.to_csv(errors_filepath)
    
    def save_data(
        self,
        data: DataFrame,
    ):
        dataset_filepath = (
            self.results_dirpath
            / self.exp_dirname
            / DATASET_DIRNAME
            / "dataset.csv"
        )
        data.to_csv(dataset_filepath)
        
    def save_checkpoint(
        self,
        model_state_dict: dict[str, any],
        optimizer_state_dict: dict[str, any],
        epoch: int,
        verbose: bool = False,
    ):
        """Save a model checkpoint.

        Args:
            epoch (int): The epoch number.
            model_state_dict (dict[str, any]): The model state dictionary.
            optimizer_state_dict (dict[str, any]): The optimizer state dictionary.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        checkpoint_filename = f"checkpoint_ep_{epoch}.pt"
            
        checkpoint_filepath = (
            self.results_dirpath
            / self.exp_dirname
            / CHECKPOINT_DIRNAME
            / checkpoint_filename
        )

        if verbose:
            console.print(
                f"Saving model at epoch {epoch} to [bold]{checkpoint_filename}[/bold]."
            )

        checkpoint_dict = {
            "epoch": epoch,
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "id_to_label": global_immutable.IDS_TO_LABELS, #TODO find a better way to store this mapping
        }

        torch.save(checkpoint_dict, checkpoint_filepath)

    def load_checkpoint(
        self,
        checkpoint_filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
        """Load model weights from a checkpoint.

        Args:
            checkpoint_filename (str): The name of the checkpoint file to load.
            model (torch.nn.Module): The model to load the checkpoint into.
            optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint into.

        Returns:
            tuple[torch.nn.Module, torch.optim.Optimizer, int]: The model, optimizer, and epoch number.
        """
        console.print(f"Loading checkpoint from [bold]{checkpoint_filename}[/bold].")

        checkpoint_filepath = (
            self.results_dirpath
            / self.exp_dirname
            / CHECKPOINT_DIRNAME
            / checkpoint_filename
        )

        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        return model, optimizer, checkpoint["epoch"]