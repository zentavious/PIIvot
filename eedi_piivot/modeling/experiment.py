"""A module for managing experiments."""

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pydantic import ValidationError
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from eedi_piivot.utils import Config
from eedi_piivot.utils import Persistence
from .BERTDialogueDataset import BERTDialogueDataset, MultiSentenceBERTDialogueDataset
from .Tracker import WandbTracker
from .DialogueTrainer import DialogueTrainer
from .DialogueEvaluator import DialogueEvaluator
from eedi_piivot.utils.immutable import global_immutable
from eedi_piivot.utils.console import console
from eedi_piivot.utils import set_seed
from .tokenizer_factory import create_tokenizer
from .dataset_factory import create_dataset

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
        self.train_dataloaders = []
        self.val_dataloaders = []

        self.__setup__(results_dirpath, config_filepath, resume_checkpoint_filename)
        self.__load_data__()

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
        set_seed(global_immutable.SEED)

        print(f"DEVICE: {global_immutable.DEVICE}")

    def __load_data__(self):
        console.rule("Step 2: Loading the data.")

        self.tokenizer = create_tokenizer(self.config.experiment.model.params.name, 
                                          self.config.experiment.model.params.from_pretrained, 
                                          self.config.experiment.model.pretrained_params.pretrained_model_name_or_path)
        
        full_dataset = create_dataset(self.config.input_data.dataset.params.name,
                                      self.config.experiment.model.params.max_len, 
                                      self.tokenizer,
                                      self.config.input_data.dataset.params.augmented_non_pii,
                                      self.config.input_data.dataset.params.augmented_pii)
        
        # map non-augmented indidices back to full_dataset 
        full_indices = np.array(full_dataset.get_non_augmented_indices())
        full_minority_labels = full_dataset.get_non_augmented_df()['minority_label']

        train_valid_idx, test_idx = train_test_split(np.arange(len(full_indices)),
                                                    train_size=self.config.input_data.train_split,
                                                    random_state=global_immutable.SEED,
                                                    shuffle=True,
                                                    stratify=full_minority_labels)
        
        test_idx = full_indices[test_idx]
        # TODO somethign wrong with this, it needs to be prepped for the next rain_test_split
        train_valid_minority_labels = full_minority_labels.iloc[train_valid_idx]
        train_valid_indices = full_indices[train_valid_idx]

        # train_valid_dataset = Subset(full_dataset, train_valid_idx)
        test_dataset = Subset(full_dataset, test_idx)

        # train_valid_df = full_dataset.get_df_from_indicies(train_valid_idx)
        # train_valid_labels = [train_valid_dataset.dataset.data.iloc[i].minority_label for i in train_valid_dataset.indices]

        # TODO remove k folds logic
        # if (self.config.input_data.k_folds):
        #     kfold = StratifiedKFold(n_splits=self.config.input_data.k_folds, 
        #                             shuffle=True, 
        #                             random_state=global_immutable.SEED)
             
        #     for train_idx, val_idx in kfold.split(train_valid_dataset, train_valid_labels):
        #         train_dataset = Subset(train_valid_dataset,train_idx)
        #         valid_dataset = Subset(train_valid_dataset,val_idx)
                
        #         self.train_dataloaders.append(DataLoader(train_dataset, **self.config.input_data.train_params.dict()))
        #         self.valid_dataloaders.append(DataLoader(valid_dataset, **self.config.input_data.valid_params.dict()))
        # else:
        
        train_idx, val_idx = train_test_split(np.arange(len(train_valid_indices)),
                                                    test_size=self.config.input_data.valid_split,
                                                    random_state=global_immutable.SEED,
                                                    shuffle=True,
                                                    stratify=train_valid_minority_labels)
        
        aug_train_idx = full_dataset.expand_group_indices(train_valid_indices[train_idx])
        val_idx = train_valid_indices[val_idx]
        
        self.train_dataloader = DataLoader(Subset(full_dataset, aug_train_idx), **self.config.input_data.train_params.model_dump())
        self.valid_dataloaders = [DataLoader(Subset(full_dataset, val_idx), **self.config.input_data.valid_params.model_dump())]
        if self.config.input_data.dataset.params.augmented_non_pii or self.config.input_data.dataset.params.augmented_pii:
            aug_val_idx = full_dataset.expand_group_indices(val_idx)
            self.valid_dataloaders.append(DataLoader(Subset(full_dataset, aug_val_idx), **self.config.input_data.valid_params.model_dump()))
        
        self.test_dataloader = DataLoader(test_dataset, **self.config.input_data.valid_params.model_dump())

    def run_train(self):
        exp_name = self.persistence.exp_dirname
        resume = False
        config={
            "learning_rate": self.config.experiment.trainer.optimizer.params.lr,
            "epochs": self.config.experiment.trainer.epochs,
        }
        tracker = WandbTracker(exp_name, config, resume)

        trainer = DialogueTrainer(
                tracker=tracker,
                exp_name=exp_name,
                experiment_config=self.config.experiment,
                grad_clipping_max_norm=self.config.experiment.trainer.grad_clipping_max_norm, # TODO is this a value we want to hyper parameterize?
            )
        
        evaluator = DialogueEvaluator(self.tokenizer)

        try:
            errors = trainer.train(
                    self.train_dataloader,
                    self.valid_dataloaders,
                    evaluator=evaluator,
                    num_epochs=self.config.experiment.trainer.epochs,
                    device=global_immutable.DEVICE
                )

            print("most recent epoch", trainer._epoch)

            self.persistence.save_checkpoint(
                trainer.model.state_dict(), trainer.optimizer.state_dict(), trainer._epoch
            )

            print("model saved...")

            if errors is None:
                print("no validation used during model training. try increasing val_frequency.")
            else:
                self.persistence.save_errors(
                    "val",
                    errors)

            print("validation errors saved...")

        except KeyboardInterrupt:
            print("most recent epoch", trainer._epoch)

            self.persistence.save_checkpoint(
                trainer.model.state_dict(), trainer.optimizer.state_dict(), trainer._epoch, True
            )

        tracker.end_run()
