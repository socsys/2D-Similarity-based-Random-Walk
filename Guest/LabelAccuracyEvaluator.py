import torch
from torch.utils.data import DataLoader
import logging
import os
import csv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve, auc
from matplotlib import pyplot
import numpy as np

logger = logging.getLogger(__name__)

def batch_to_device(batch, target_device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class LabelAccuracyEvaluator():
    """
    Evaluate a model based on its accuracy on a labeled dataset
    This requires a model with LossFunction.SOFTMAX
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset
        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "random_accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = [
                            "epoch",
                            "steps",
                            "accuracy",
                            "macro_f1",

                            "precision_class_0_non_misogynistic",
                            "precision_class_1_misogynistic",

                            "recall_class_0_non_misogynistic",
                            "recall_class_1_misogynistic",

                            "f1_class_0_non_misogynistic",
                            "f1_class_1_misogynistic",

                            "pr_auc"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        true_labels = []
        pred_labels = []
        pos_probs = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
            true_labels.extend(list(label_ids.cpu()))
            pred_labels.extend(list(torch.argmax(prediction, dim=1).cpu()))
            pos_probs.extend(list(prediction[:, 1].cpu()))
        accuracy = correct/total
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)

        # Class-wise metrics
        class_precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
        class_recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

        P, R, _ = precision_recall_curve(true_labels, pos_probs)
        auc_score = auc(R, P)

        print('Accuracy:',accuracy)
        print('Macro F1:', macro_f1)
        print('Class Precision:', class_precision)
        print('Class Recall:', class_recall)
        print('Class F1:', class_f1)
        print('PR AUC:',auc_score)
        

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            
            # Check if file exists
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)  # Write header
                    writer.writerow([epoch, steps, accuracy, macro_f1, class_precision[0], class_precision[1], class_recall[0], class_recall[1], class_f1[0], class_f1[1], auc_score])
            else:
                # Read existing data to count rows (excluding header)
                with open(csv_path, newline='', encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    header = reader[0]  # First row is header
                    data_rows = reader[1:]  # Exclude header

                # Append new row
                new_row = [epoch, steps, accuracy, macro_f1, class_precision[0], class_precision[1], class_recall[0], class_recall[1], class_f1[0], class_f1[1], auc_score]
                data_rows.append(new_row)

                # Write the new row
                with open(csv_path, mode="a", newline='', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(new_row)

                # If exactly 3 data rows exist (excluding header), compute the mean and append it
                NUM_FOLDS = 3

                if len(data_rows) == NUM_FOLDS:
                    values = np.array(data_rows, dtype=float)

                    mean_values = np.mean(values, axis=0)
                    var_values  = np.var(values, axis=0)

                    mean_row = ["Mean"] + list(mean_values[1:])
                    var_row  = ["Variance"] + list(var_values[1:])

                    with open(csv_path, mode="a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(mean_row)
                        writer.writerow(var_row)


        return accuracy


