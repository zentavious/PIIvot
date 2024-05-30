import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from pathlib import Path
from typing import List
import wandb
from tqdm.autonotebook import tqdm, trange

from eedi_piivot.modeling import WandbTracker
from eedi_piivot.modeling import DialogueEvaluator
from eedi_piivot.utils.immutable import global_immutable

repo_path = Path(__file__).resolve().parents[0]
DEFAULT_DEVICE = "cpu"
SAVE_MODEL_EVERY = 30

class DialogueTrainer:

    def __init__(
        self,
        model,
        optimizer: Optimizer,
        tracker: WandbTracker,
        exp_name: str,
        lr_scheduler: lr_scheduler = None,
        epoch: int = 0,
        grad_clipping_max_norm: int = 10,
    ) -> None:
        """
        Class to train a BERT model
        :param model: The model to train, drived from interface BaseModel
        :param optimizer: The optimizer to use, should be already instantiated
        :param tracker: The tracker to use for logging
        :param lr_scheduler: The learning rate scheduler to use, default None
        :param epoch: The epoch to start training from, default 0 unless resume logic is initiated.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._epoch = epoch
        self.grad_clipping_max_norm = grad_clipping_max_norm
        self.losses_per_n_steps: List[int] = []
        self.losses_per_epochs: List[int] = []
        self.acc_per_n_steps: List[int] = []
        self.acc_per_epochs: List[int] = []
        self.tracker = tracker
        self.exp_name = exp_name

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        evaluator: DialogueEvaluator,
        num_epochs: int = 1,
        split: bool = True,
        use_tqdm: bool = True,
        val_frequency: int = 1,
        device=DEFAULT_DEVICE,
    ) -> None:
        """
        Train the model for a specified number of epochs

        :param dataset: The dataset to train on
        :param negative_sampler: The negative sampler to use
        :param evaluator: The evaluator to use
        :param num_epochs: The number of epochs to train for
        :param use_tqdm: Whether to use tqdm for progress bar
        :param val_frequency: The frequency of evaluation
        :param device: The device to use for computation
        """

        # loss_fn = nn.CrossEntropyLoss(reduction="mean") # Not defined in prior code
        # Initialising loss function
        # NOTED: CLE with (reduction="mean") param is equivalent to the Negative-Log Likelihood loss but with softmax being applied before log transformation

        # progress bar implementation
        if use_tqdm:
            epochs = trange(
                self._epoch + 1,
                1 + num_epochs,
                desc="Epochs",
                initial=self._epoch,
                total=num_epochs,
            )
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        save_every = list(range(SAVE_MODEL_EVERY, num_epochs, SAVE_MODEL_EVERY))

        for epoch in epochs:
            # loss for current epoch
            if epoch in save_every:
                checkpoint_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_path = str(
                    repo_path.parents[0]
                    / "results"
                    / self.exp_name
                    / "checkpoints"
                    / f'checkpoint_ep_{checkpoint_dict["epoch"]}.pt'
                )
                torch.save(checkpoint_dict, save_path)
            
            if use_tqdm:
                # progress bar for batches
                batches = tqdm(train_dataloader, desc="Batches")
            else:
                batches = train_dataloader  # no progress bar
                
            current_epoch_loss, tr_balanced_accuracy = 0, 0
            current_batch_loss = []
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_preds, tr_labels = [], []

            self.model.train()
            for idx, batch in enumerate(batches):
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)

                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output.loss
                tr_logits = output.logits
                # loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)

                current_epoch_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)
                
                if idx % 100==0:
                    # add the loss to losses_per_n_steps list
                    loss_step = current_epoch_loss/nb_tr_steps

                    # TODO check loss_step type (is tensor?)

                    self.losses_per_n_steps.append(loss_step)
                    current_batch_loss.append(loss_step)

                    if use_tqdm:
                        # sets the value of "prev_loss" to the second-to-last loss value if there are at least two losses recorded, otherwise set to None.
                        batches.set_postfix(
                            {
                                "Train loss": self.losses_per_n_steps[-1],
                                "prev_loss": (
                                    self.losses_per_n_steps[-2]
                                    if len(self.losses_per_n_steps) > 1
                                    else None
                                ),
                            }
                        )
                
                # compute training accuracy
                flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
                active_logits = tr_logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                
                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
                #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
                
                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                tr_labels.extend(labels)
                tr_preds.extend(predictions)

                # tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                # tr_accuracy += tmp_tr_accuracy

                # tmp_tr_balanced_accruacy = balanced_accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                # tr_balanced_accuracy += tmp_tr_balanced_accruacy

                # if idx % 100==0:
                #     # add the loss to acc_per_n_steps list
                #     acc_step = tr_balanced_accuracy/nb_tr_steps
                #     self.acc_per_n_steps.append(loss_step)
                #     # loss logging
                #     self.tracker.log_metrics({"Step Accuracy/train_acc": acc_step}, step=nb_tr_steps)

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.grad_clipping_max_norm
                )

                # current_epoch_loss += loss.detach()  # loss aggregation
                
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # clear batch data
            del batches

            # compute loss of current epoc
            epoch_loss = current_epoch_loss / len(train_dataloader)
            
            # loss logging
            print(f"Logging step and epoch-level training loss to tracker.")
            # self.tracker.log_metrics({f"Step Loss/{epoch}_train_loss": self.losses_per_n_steps}, step=epoch)
            temp_table = [[loss, step] for (loss, step) in zip(current_batch_loss, list(range(0, len(current_batch_loss) * 100, 100)))]
            table = wandb.Table(data=temp_table, columns=["loss", "step"])
            self.tracker.log_metrics(
                {
                    f"Step Loss/{epoch}_train_loss": wandb.plot.line(
                        table, "loss", "step", title=f"Epoch {epoch} loss by step"
                    )
                }
            )

            # add the loss to losses_per_epochs list
            self.losses_per_epochs.append(epoch_loss)
            # loss logging
            self.tracker.log_metrics({"Loss/train_loss": epoch_loss}, step=epoch)
            # TODO Print datatype of epoch_loss

            if use_tqdm:
                # sets the value of "prev_loss" to the second-to-last loss value if there are at least two losses recorded, otherwise set to None.
                epochs.set_postfix(
                    {
                        "Train loss": self.losses_per_epochs[-1],
                        "prev_loss": (
                            self.losses_per_epochs[-2]
                            if len(self.losses_per_epochs) > 1
                            else None
                        ),
                    }
                )

            # the last successful finished epoch
            self._epoch = epoch

            # --- validataion
            if split and (epoch + 1) % val_frequency == 0:
                # # get metrics for predicting tail
                # print("\n\n---- EVALUATION STARTING (train) ----\n\n") # TODO Remove training evaluation
                # evaluator.evaluate(
                #     "train",
                #     train_dataloader,
                #     tracker=self.tracker,
                #     cur_epoch=epoch,
                # )
                # print("---- EVALUATION FINISED (train)----\n")
                # print("mrr_opt is:", mrr_opt)

                print("\n\n---- EVALUATION STARTING (val) ----\n\n")
                evaluator.evaluate(
                    "val",
                    val_dataloader,
                    tracker=self.tracker,
                    cur_epoch=epoch,
                )
                print("---- EVALUATION FINISED (val)----\n")