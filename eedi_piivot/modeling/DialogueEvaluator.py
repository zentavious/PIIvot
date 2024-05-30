
from collections import defaultdict
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    balanced_accuracy_score,
)

from eedi_piivot.modeling import Tracker
from eedi_piivot.utils.immutable import global_immutable

def classification_report_custom(y_true, y_pred, labels=None):
    """
    Build a text report showing the main classification metrics.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (array-like, optional): List of labels to include in the report.

    Returns:
        str: Text report showing the main classification metrics.
    """

    # Calculate true positives, false positives, false negatives, and support for each label
    tp = defaultdict(int)  # True positives
    fp = defaultdict(int)  # False positives
    fn = defaultdict(int)  # False negatives
    support = defaultdict(int)  # Support

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
        support[true] += 1

    # Compute precision, recall, and F1-score for each label
    precision = {}
    recall = {}
    f1_score = {}

    for label in set(y_true + y_pred):
        precision[label] = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) != 0 else 0
        recall[label] = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) != 0 else 0
        f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) != 0 else 0

    # Compute macro-average precision, recall, and F1-score
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1_score = np.mean(list(f1_score.values()))

    # Build the classification report
    report = "              precision    recall  f1-score   support\n\n"

    for label in sorted(set(y_true + y_pred)):

        report += f"    {global_immutable.IDS_TO_LABELS[label]:<10}   {precision.get(label, 0):<9.2f}   {recall.get(label, 0):<6.2f}   {f1_score.get(label, 0):<8.2f}   {support.get(label, 0):<7}\n"

    report += "\n"

    report += f"    macro avg   {macro_precision:<9.2f}   {macro_recall:<6.2f}   {macro_f1_score:<8.2f}   {len(y_true):<7}\n"
    report += f" weighted avg   {macro_precision:<9.2f}   {macro_recall:<6.2f}   {macro_f1_score:<8.2f}   {len(y_true):<7}\n"

    return report

class DialogueEvaluator:
    def __init__(
        self,
        model
    ):
        """Class for evaluating PII predictions
        :param model: Model to evaluate
        """
        self.model = model

    def evaluate(
        self,
        split_name,
        eval_dataloader: DataLoader,
        tracker: Tracker = None,
        cur_epoch: int = None,
    ) -> Dict[str, float]:
        """Method to evaluate the model on the validation set

        :param eval_dataloader: Dataloader to get batches of Shape:(batch_size, 3) -> Misleading name -> evaluation_dataloader, dataloader.
:param tracker: Tracker: a tracker class to log the loss and metrics
        :param cur_epoch: int: current epoch number before training

        returns: dictionary of metrics
        """
        # Panagiota's variables
        y_pred_all = []
        y_true_all = []
        loss_all = 0

        
        # my variables       
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        # eval_preds, eval_labels, eval_tokens = [], [], []

        # loss_fn = nn.CrossEntropyLoss(reduction="mean")
        eval_dataloader_len = len(eval_dataloader)

        with torch.no_grad():  # Disable gradient tracking
            self.model.eval()

            for batch in eval_dataloader:  # Iterate over all evaluation batches.
                ids = batch['input_ids'].to(global_immutable.DEVICE, dtype = torch.long)
                mask = batch['attention_mask'].to(global_immutable.DEVICE, dtype = torch.long)
                labels = batch['labels'].to(global_immutable.DEVICE, dtype = torch.long)
                
                output = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output.loss
                eval_logits = output.logits
                
                loss_all += loss.item()

                # compute evaluation accuracy
                flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                                
                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
            
                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                y_true_all.extend(labels.squeeze().tolist())
                y_pred_all.extend(predictions.squeeze().tolist())
                
                # tmp_tr_balanced_accruacy = balanced_accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                # balanced_accurary_all += tmp_tr_balanced_accruacy
            
        metrics = self.metrics_for_all_batches(
            eval_dataloader_len,
            split_name,
            loss_all,
            y_pred_all,
            y_true_all,
        )
        self.log_to_tracker(
            tracker,
            metrics,
            cur_epoch,
            verbose=True,
        )  # save metrics to tracker

        return metrics

    def metrics_for_all_batches(
        self,
        eval_dataloader_len,
        split_name,
        loss_all,
        y_pred_all,
        y_true_all,
    ):

        print(classification_report_custom(y_true_all, y_pred_all))

        epoch_loss = loss_all / eval_dataloader_len
        epoch_accuracy = balanced_accuracy_score(y_true_all, y_pred_all)
        
        metrics = {
                f"Loss/{split_name}_loss": epoch_loss,
                f"Balanced Accuracy/{split_name}_accuracy": epoch_accuracy,
            }
        return metrics

    def log_to_tracker(
        self,
        tracker: Tracker,
        metrics: Dict[str, float],
        cur_epoch: int = None,
        verbose=False,
    ) -> None:
        """Method to save metrics to the given tracker. Could extend this to have multiple trackers

        :param tracker: tracker class
        :param metrics: Dict[str, float]: dictionary of metrics to log
        :param cur_epoch: int: report the epoch (step) for the given metrics
        :param verbose: Bool: if True emits the metrics to cmd
        Returns:

        """
        print(f"Logging metrics to tracker:\n{metrics}")
        
        for k, v in metrics.items():
            if verbose:
                print(f"{k}: {v:.2f}")
        tracker.log_metrics(metrics, step=cur_epoch)