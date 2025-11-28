from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)


class Evaluator:
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = (
            class_names if class_names else [str(i) for i in range(num_classes)]
        )

        # Ensure artifacts directories exist
        Path("artifacts/img").mkdir(exist_ok=True)
        Path("artifacts/csv").mkdir(exist_ok=True)

    def _get_filename(
        self, base_name: str, tags: Optional[Dict[str, str]], ext: str
    ) -> str:
        """Helper to construct filename with tags."""
        folder = "img" if ext == "png" else ext

        if not tags:
            return f"artifacts/{folder}/{base_name}.{ext}"

        # Construct suffix from tags (e.g. _bert-mini_f66bce0c)
        # We prioritize regex_hash if present, then text_model_name
        parts = [base_name]

        if "text_model_name" in tags:
            # Clean model name (prajjwal1/bert-mini -> prajjwal1_bert-mini)
            clean_name = tags["text_model_name"].replace("/", "_")
            parts.append(clean_name)

        if "regex_hash" in tags:
            parts.append(tags["regex_hash"])

        return f"artifacts/{folder}/{'_'.join(parts)}.{ext}"

    def compute_metrics(
        self, targets: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes F1, Precision, Recall (micro and macro).
        """
        metrics = {}
        for avg in ["micro", "macro"]:
            metrics[f"f1_{avg}"] = f1_score(
                targets, preds, average=avg, zero_division=0
            )
            metrics[f"precision_{avg}"] = precision_score(
                targets, preds, average=avg, zero_division=0
            )
            metrics[f"recall_{avg}"] = recall_score(
                targets, preds, average=avg, zero_division=0
            )

        return metrics

    def save_per_class_metrics(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Calculates precision/recall per class and saves to CSV.
        """
        logger.info("Calculating per-class metrics...")

        precision = precision_score(targets, preds, average=None, zero_division=0)
        recall = recall_score(targets, preds, average=None, zero_division=0)
        f1 = f1_score(targets, preds, average=None, zero_division=0)

        # Support (number of true instances)
        support = targets.sum(axis=0)

        data = []
        # use .5f precision for float columns
        for i, name in enumerate(self.class_names):
            data.append(
                {
                    "class_name": name,
                    "precision": f"{precision[i]:.5f}",
                    "recall": f"{recall[i]:.5f}",
                    "f1": f"{f1[i]:.5f}",
                    "support": support[i],
                }
            )

        df = pd.DataFrame(data)
        # Sort by support (descending) to see most frequent classes first
        df = df.sort_values("support", ascending=False)

        output_path = self._get_filename("per_class_metrics", tags, "csv")
        df.to_csv(output_path, index=False)
        logger.success(f"Per-class metrics saved to {output_path}")

    def plot_roc_curve(
        self,
        targets: np.ndarray,
        probs: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Plots Micro-Averaged ROC Curve.
        """
        logger.info("Generating ROC curve...")

        # Flatten for micro-average
        fpr, tpr, _ = roc_curve(targets.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr,
            tpr,
            label=f"Micro-average ROC curve (area = {roc_auc:0.2f})",
            color="darkorange",
            lw=2,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (Micro-Average)")
        plt.legend(loc="lower right")

        output_path = self._get_filename("roc_curve", tags, "png")
        plt.savefig(output_path)
        plt.close()
        logger.success(f"ROC curve saved to {output_path}")

    def plot_pr_curve(
        self,
        targets: np.ndarray,
        probs: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Plots Micro-Averaged Precision-Recall Curve.
        """
        logger.info("Generating Precision-Recall curve...")

        precision, recall, _ = precision_recall_curve(targets.ravel(), probs.ravel())
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 8))
        plt.plot(
            recall,
            precision,
            label=f"Micro-average PR curve (area = {pr_auc:0.2f})",
            color="blue",
            lw=2,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Micro-Average)")
        plt.legend(loc="lower left")

        output_path = self._get_filename("pr_curve", tags, "png")
        plt.savefig(output_path)
        plt.close()
        logger.success(f"PR curve saved to {output_path}")

    def plot_global_confusion_matrix(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Plots a global 2x2 confusion matrix (TP/FP/FN/TN) across all classes.
        """
        logger.info("Generating global confusion matrix...")

        # Flatten to treat as binary classification problem
        cm = confusion_matrix(targets.ravel(), preds.ravel())

        # cm is [[TN, FP], [FN, TP]]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["True Negative", "True Positive"],
        )
        plt.title("Global Confusion Matrix (All Classes Flattened)")

        output_path = self._get_filename("global_confusion_matrix", tags, "png")
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Global confusion matrix saved to {output_path}")

    def evaluate_regex_performance(
        self,
        regex_features: np.ndarray,
        true_labels: np.ndarray,
        regex_map: Dict[int, str],
        code2idx: Dict[int, int],
        output_path: str,
    ):
        """
        Evaluates regex performance per code and saves to CSV.
        """
        logger.info("Evaluating regex performance...")

        # True Positives: both are 1
        tp = (regex_features * true_labels).sum(axis=0)
        # False Positives: regex is 1, true is 0
        fp = (regex_features * (1 - true_labels)).sum(axis=0)
        # False Negatives: regex is 0, true is 1
        fn = ((1 - regex_features) * true_labels).sum(axis=0)

        # Support (number of true instances)
        support = true_labels.sum(axis=0)

        metrics_data = []

        # Iterate over classes in the encoder
        for code, idx in code2idx.items():
            t = tp[idx]
            f = fp[idx]
            n = fn[idx]
            s = support[idx]

            precision = t / (t + f) if (t + f) > 0 else 0.0
            recall = t / (t + n) if (t + n) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics_data.append(
                {
                    "code": code,
                    "count": int(s),
                    "precision": f"{precision:.5f}",
                    "recall": f"{recall:.5f}",
                    "f1": f"{f1:.5f}",
                    "tp": int(t),
                    "fp": int(f),
                    "fn": int(n),
                    "regex": regex_map.get(code, "N/A"),
                }
            )

        df = pd.DataFrame(metrics_data)
        df.sort_values("f1", ascending=False, inplace=True)
        df.to_csv(output_path, index=False)
        logger.success(f"Saved results to {output_path}")
        logger.info("\nTop 10 Regexes by F1 Score:")
        logger.info(
            "\n"
            + df.head(10)[
                ["code", "count", "f1", "precision", "recall", "regex"]
            ].to_string()
        )
