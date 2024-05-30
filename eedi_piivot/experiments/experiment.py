"""A module for managing experiments."""

import json
from pathlib import Path
from torch.utils.data import random_split
from pydantic import ValidationError
from torch.utils.data import DataLoader

from .config import Config
from .persistence import Persistence
from eedi_piivot.modeling.BERTDialogueDataset import BERTDialogueDataset
from eedi_piivot.modeling.Tracker import WandbTracker
from eedi_piivot.modeling.DialogueTrainer import DialogueTrainer
from eedi_piivot.modeling.DialogueEvaluator import DialogueEvaluator
from eedi_piivot.utils.immutable import global_immutable
from eedi_piivot.utils.console import console
from eedi_piivot.utils.random import set_seed
from eedi_piivot.modeling import (
    create_model,
    create_optimizer,
)

class Experiment:
    """A class to manage experiments."""

    def __init__(
        self,
        results_dirpath: Path,
        config_filepath: Path,
        resume_checkpoint_filename: str | None = None,
    ):
        """Initializes the Experiment class.

        Given the experiment details it sets up persistence, loads the config, and sets up neccessary precursors to running an experiment.

        Args:
            results_dirpath (Path): The path to the results directory which will contain
            the individual experiment directories.
            config_filepath (Path): The path to the config file.
            resume_checkpoint_filename (str, optional): The name of the checkpoint file
            to resume training from. Defaults to None.

        """
        self.__setup__(results_dirpath, config_filepath, resume_checkpoint_filename)
        self.__load_data__()
        self.__initialize_model__()

    def __setup__(self, results_dirpath, config_filepath, resume_checkpoint_filename):
        console.rule(
            f"Step 1: Loading config and setting up persistence for file {config_filepath.name}"
        )

        candidate_exp_dirname = config_filepath.stem

        self.persistence = Persistence(
            results_dirpath=results_dirpath,
            candidate_exp_dirname=candidate_exp_dirname,
            resume_checkpoint_filename=resume_checkpoint_filename,
        )

        with open(config_filepath, "r") as f:
            raw_data = json.load(f)

        try:
            self.config = Config(**raw_data)
        except ValidationError as e:
            print(e)

        global_immutable.SEED = self.config.experiment.seed
        set_seed(self.config.experiment.seed)

        print(f"DEVICE: {global_immutable.DEVICE}")

    def __load_data__(self):
        console.rule("Step 2: Loading the data.")

        # TODO add this to config + factory
        full_dataset = BERTDialogueDataset(self.config.input_data.max_len)
        
        (self.config.input_data.train_split, self.config.input_data.valid_split)

        train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [self.config.input_data.train_split, 
                                                                                 self.config.input_data.valid_split,
                                                                                 1 - (self.config.input_data.train_split + self.config.input_data.valid_split)])
        
        self.train_dataloader = DataLoader(train_dataset, **self.config.input_data.train_params.dict())
        self.valid_dataloader = DataLoader(valid_dataset, **self.config.input_data.valid_params.dict())
        self.test_dataloader = DataLoader(test_dataset, **self.config.input_data.valid_params.dict())

    def __initialize_model__(self):
        console.rule(
            f"Step 3: Initializing the {self.config.experiment.model.model_params.model_name} model."
        )
        
        self.model = create_model(self.config.experiment.model.model_params.model_name, 
                                  self.config.experiment.model.model_params.from_pretrained, 
                                  **self.config.experiment.model.pretrained_params.dict()) # not sure if this line works
        
        self.model.to(global_immutable.DEVICE)

        self.optimizer = create_optimizer(self.config.experiment.trainer.optimizer.name,
                                          self.model.parameters(),
                                          **self.config.experiment.trainer.optimizer.params.dict())

    def run_train(self):
        exp_name = self.persistence.exp_dirname
        resume = False
        config={
            "learning_rate": self.config.experiment.trainer.optimizer.params.lr,
            "epochs": self.config.experiment.trainer.epochs,
        }
        tracker = WandbTracker(exp_name, config, resume)

        trainer = DialogueTrainer(
                self.model,
                self.optimizer,
                tracker=tracker,
                epoch=0,
                exp_name=exp_name,
                grad_clipping_max_norm=self.config.experiment.trainer.grad_clipping_max_norm, # TODO is this a value we want to hyper parameterize?
            )

        evaluator = DialogueEvaluator(
            self.model,
        )

        try:
            trainer.train(
                self.train_dataloader,
                self.valid_dataloader,
                evaluator=evaluator,
                num_epochs=self.config.experiment.trainer.epochs,
            )

            print("most recent epoch", trainer._epoch)

            self.persistence.save_checkpoint(
                self.model.state_dict(), self.optimizer.state_dict(), trainer._epoch
            )
            print("model saved..")

        except KeyboardInterrupt:
            print("most recent epoch", trainer._epoch)

            self.persistence.save_checkpoint(
                self.model.state_dict(), self.optimizer.state_dict(), trainer._epoch
            )

        tracker.end_run()
