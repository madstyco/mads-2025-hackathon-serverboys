import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from akte_classifier.datasets.dataset import DatasetFactory
from akte_classifier.models.llm import LLMClassifier
from akte_classifier.models.neural import (HybridClassifier, NeuralClassifier,
                                           TextVectorizer)
from akte_classifier.models.prompts import ClassificationPromptTemplate
from akte_classifier.models.regex import RegexGenerator, RegexVectorizer
from akte_classifier.utils.data import get_long_tail_labels, load_descriptions
from akte_classifier.utils.evaluation import Evaluator
from akte_classifier.utils.logging import (CompositeLogger, ConsoleLogger,
                                           MLFlowLogger)


def get_default_device() -> str:
    device = os.environ.get("DEVICE")
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainingConfig:
    data_path: str = "assets/aktes.jsonl"
    csv_path: str = "assets/rechtsfeiten.csv"
    batch_size: int = 32
    split_ratio: float = 0.8
    model_name: str = "prajjwal1/bert-tiny"
    learning_rate: float = 1e-3
    num_epochs: int = 50
    device: str = get_default_device()
    model_class: str = "HybridClassifier"  # Default to Hybrid
    use_regex: bool = True  # Whether to use regex features
    hidden_dim: int = 256  # Hidden layer dimension
    max_length: Optional[int] = None  # Max token length (None = auto)
    pooling: Optional[str] = None  # Pooling strategy: "mean", "cls", or None (auto)
    long_tail_threshold: Optional[int] = None  # Threshold for long-tail labels
    experiment_name: str = "kadaster_experiment"  # MLFlow experiment name


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.loaders: Dict[str, DataLoader] = {}
        self.num_classes = 0
        self.class_names: List[str] = []

        # Models
        self.vectorizer = None
        self.regex_vectorizer = None
        self.classifier = None

        # training
        self.loss_fn = None
        self.optimizer = None

        # evaluation
        self.evaluator = None
        self.logger = CompositeLogger([ConsoleLogger(), MLFlowLogger()])

    def get_data(self):
        logger.info("Initializing DatasetFactory...")
        factory = DatasetFactory(
            self.config.data_path,
            batch_size=self.config.batch_size,
            split_ratio=self.config.split_ratio,
        )

        # Initialize Text Vectorizer
        if self.config.model_class in ["NeuralClassifier", "HybridClassifier"]:
            self.vectorizer = TextVectorizer(
                self.config.model_name,
                max_length=self.config.max_length,
                pooling=self.config.pooling,
            )
            self.vectorizer.model.to(self.device)
            for param in self.vectorizer.model.parameters():
                param.requires_grad = False
            logger.info("Intialized TextVectorizer & freezing weights...")
        else:
            logger.info("Skipping TextVectorizer initialization...")
            self.vectorizer = None

        # Initialize Regex Vectorizer if using Hybrid or RegexOnly
        if self.config.use_regex:
            logger.info("Initializing RegexVectorizer...")
            regex_gen = RegexGenerator(self.config.csv_path)
            # Pass the encoder to ensure alignment
            self.regex_vectorizer = RegexVectorizer(
                regex_gen, label_encoder=factory.encoder
            )

        # Instead of vectorizing the dataset every time
        # which is a huge bottleneck, we can cache the
        # vectorized dataset
        self.loaders = factory.get_vectorized_loader(
            self.vectorizer, self.regex_vectorizer
        )

        self.num_classes = len(factory.encoder)
        self.class_names = [
            str(factory.encoder.idx2code.get(i + 1, i + 1))
            for i in range(self.num_classes)
        ]
        logger.info(f"Number of classes: {self.num_classes}")

        # Initialize Evaluator
        self.evaluator = Evaluator(self.num_classes, class_names=self.class_names)

    def setup_models(self):
        logger.info(f"Initializing {self.config.model_class}...")

        if self.config.model_class == "NeuralClassifier":
            self.classifier = NeuralClassifier(
                input_dim=self.vectorizer.hidden_size,
                num_classes=self.num_classes,
                hidden_dim=self.config.hidden_dim,
            )
        elif self.config.model_class == "HybridClassifier":
            self.classifier = HybridClassifier(
                input_dim=self.vectorizer.hidden_size,
                regex_dim=self.regex_vectorizer.output_dim,
                num_classes=self.num_classes,
                hidden_dim=self.config.hidden_dim,
            )
        elif self.config.model_class == "RegexOnlyClassifier":
            # Ensure regex vectorizer is available
            if not self.regex_vectorizer:
                raise ValueError(
                    "RegexOnlyClassifier requires regex features. Ensure regex_vectorizer is initialized."
                )

            self.classifier = NeuralClassifier(
                input_dim=self.regex_vectorizer.output_dim,
                num_classes=self.num_classes,
                hidden_dim=self.config.hidden_dim,
            )
        else:
            raise ValueError(f"Unknown model class: {self.config.model_class}")

        self.classifier.to(self.device)

    def setup_optimization(self):
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.classifier.parameters(), lr=self.config.learning_rate
        )

    def train_epoch(self, epoch: int):
        assert self.classifier is not None
        assert self.optimizer is not None
        assert self.loss_fn is not None

        self.classifier.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.loaders["train"], desc=f"Epoch {epoch}/{self.config.num_epochs}"
        )

        for batch in progress_bar:
            # Handle variable number of items from loader
            if len(batch) == 3:
                embeddings, labels, regex_feats = batch
                embeddings = embeddings.to(self.device)
                regex_feats = regex_feats.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if self.config.model_class == "HybridClassifier":
                    logits = self.classifier(embeddings, regex_feats)
                elif self.config.model_class == "RegexOnlyClassifier":
                    logits = self.classifier(regex_feats)
                else:
                    # NeuralClassifier ignores regex features if present (shouldn't happen with correct loader but safe fallback)
                    logits = self.classifier(embeddings)
            else:
                embeddings, labels = batch
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.classifier(embeddings)

            # Compute loss
            loss = self.loss_fn(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.loaders["train"])
        self.logger.log_metrics({"train_loss": avg_loss}, step=epoch)

    def validate(self, epoch: int) -> Dict[str, Any]:
        assert self.classifier is not None
        assert self.loss_fn is not None
        assert self.evaluator is not None

        self.classifier.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in self.loaders["test"]:
                if len(batch) == 3:
                    embeddings, labels, regex_feats = batch
                    embeddings = embeddings.to(self.device)
                    regex_feats = regex_feats.to(self.device)
                    labels = labels.to(self.device)

                    if self.config.model_class == "HybridClassifier":
                        logits = self.classifier(embeddings, regex_feats)
                    elif self.config.model_class == "RegexOnlyClassifier":
                        logits = self.classifier(regex_feats)
                    else:
                        logits = self.classifier(embeddings)
                else:
                    embeddings, labels = batch
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.classifier(embeddings)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                # Probabilities
                probs = torch.sigmoid(logits)
                # Predictions (threshold 0.5)
                preds = (probs > 0.5).float()

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.loaders["test"])

        # Concatenate
        all_preds_arr = np.vstack(all_preds)
        all_probs_arr = np.vstack(all_probs)
        all_targets_arr = np.vstack(all_targets)

        # Compute Metrics using Evaluator
        metrics = self.evaluator.compute_metrics(all_targets_arr, all_preds_arr)
        metrics["val_loss"] = avg_loss

        self.logger.log_metrics(metrics, step=epoch)
        return {
            "preds": all_preds_arr,
            "probs": all_probs_arr,
            "targets": all_targets_arr,
            "val_loss": avg_loss,
            **metrics,
        }

    def run(self):
        self.get_data()
        self.setup_models()
        self.setup_optimization()

        self.logger.log_params(asdict(self.config))

        # Log model tags (hashes)
        tags = {}

        # Only log text model name if we are using it (Neural or Hybrid)
        if self.config.model_class != "RegexOnlyClassifier":
            tags["text_model_name"] = self.vectorizer.model_name

        if self.regex_vectorizer:
            tags["regex_hash"] = self.regex_vectorizer.hash

        self.logger.log_tags(tags)

        best_val_loss = float("inf")

        for epoch in range(1, self.config.num_epochs + 1):
            self.train_epoch(epoch)
            val_results = self.validate(epoch)

            if val_results["val_loss"] < best_val_loss:
                best_val_loss = val_results["val_loss"]
                # Save best model logic here if needed

            # Plot artifacts at the end
            if epoch == self.config.num_epochs:
                self.evaluator.plot_global_confusion_matrix(
                    val_results["targets"], val_results["preds"], tags=tags
                )
                self.evaluator.plot_roc_curve(
                    val_results["targets"], val_results["probs"], tags=tags
                )
                self.evaluator.plot_pr_curve(
                    val_results["targets"], val_results["probs"], tags=tags
                )
                self.evaluator.save_per_class_metrics(
                    val_results["targets"], val_results["preds"], tags=tags
                )


class LLMRunner:
    def __init__(
        self,
        threshold: int,
        limit: Optional[int],
        model_name: str,
        experiment_name: str,
        max_length: Optional[int] = None,
    ):
        self.threshold = threshold
        self.limit = limit
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.max_length = max_length

        # Init MLFlow Logger
        self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name)
        self.mlflow_logger.enable_genai_autolog()

    def _load_resources(self):
        """
        Loads long-tail labels, descriptions, dataset, and initializes the classifier.
        """
        # 1. Get long-tail labels
        dist_path = "artifacts/csv/label_distribution.csv"
        long_tail_codes = get_long_tail_labels(dist_path, self.threshold)
        if not long_tail_codes:
            logger.error("No long-tail labels found. Check threshold or file.")
            return None, None, None, None

        logger.info(f"Found {len(long_tail_codes)} long-tail labels.")

        # 2. Get descriptions for these labels
        csv_path = "assets/rechtsfeiten.csv"
        descriptions = load_descriptions(csv_path, long_tail_codes)

        if not descriptions:
            logger.error("No descriptions found for the long-tail labels.")
            return None, None, None, None

        # 3. Init DatasetFactory with filtering
        factory = DatasetFactory(
            file_path="assets/aktes.jsonl",
            long_tail_threshold=self.threshold,
            batch_size=1,
        )

        if not factory.train_dataset or len(factory.train_dataset) == 0:
            logger.error("No samples found in dataset after filtering.")
            return None, None, None, None

        # 4. Init LLM Classifier
        prompt_template = ClassificationPromptTemplate()
        classifier = LLMClassifier(
            model_name=self.model_name,
            descriptions=descriptions,
            prompt_template=prompt_template,
            max_length=self.max_length,
        )

        return long_tail_codes, descriptions, factory, classifier

    def _run_inference(self, factory, classifier, long_tail_codes):
        """
        Runs the classification loop.
        """
        logger.info("Classifying samples...")

        # Access the underlying HuggingFace dataset to get raw text
        hf_dataset = factory.train_dataset.dataset

        all_true_labels = []
        all_pred_labels = []
        predictions_data = []

        count = 0

        # Determine total for progress bar
        total_samples = len(hf_dataset)
        if self.limit is not None:
            total_samples = min(self.limit, total_samples)

        for i in tqdm(range(len(hf_dataset)), total=total_samples, desc="Classifying"):
            if self.limit is not None and count >= self.limit:
                break

            sample = hf_dataset[i]
            text = sample["text"]
            akte_id = sample.get("akteId", "unknown")
            true_labels = [int(c) for c in sample["rechtsfeitcodes"]]

            # Only process if it actually has long-tail labels (double check, though factory filtered it)
            relevant_true_labels = [c for c in true_labels if c in long_tail_codes]

            predicted_labels = classifier.classify(text)

            # Save prediction data
            predictions_data.append(
                {
                    "akte_id": akte_id,
                    "text_snippet": text[:30] + "...",
                    "true_labels": str(relevant_true_labels),
                    "predicted_labels": str(predicted_labels),
                    "all_true_labels": str(true_labels),
                }
            )

            # For evaluation, we need to map these to a consistent binary format or similar
            # We can create binary vectors for the long_tail_codes.

            true_binary = [
                1 if c in relevant_true_labels else 0 for c in long_tail_codes
            ]
            pred_binary = [1 if c in predicted_labels else 0 for c in long_tail_codes]

            all_true_labels.append(true_binary)
            all_pred_labels.append(pred_binary)

            count += 1

        return predictions_data, all_true_labels, all_pred_labels

    def _save_predictions(self, predictions_data):
        """
        Saves predictions to CSV.
        """
        safe_model_name = self.model_name.replace("/", "_")
        if predictions_data:
            df_preds = pd.DataFrame(predictions_data)
            pred_csv_path = f"artifacts/csv/llm_predictions_{safe_model_name}.csv"
            df_preds.to_csv(pred_csv_path, index=False)
            logger.success(f"Saved predictions to {pred_csv_path}")
            self.mlflow_logger.log_artifact(pred_csv_path)

    def _evaluate(self, true_labels, pred_labels, long_tail_codes):
        """
        Calculates metrics and generates plots.
        """
        if not true_labels:
            return

        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)

        safe_model_name = self.model_name.replace("/", "_")

        # Set run name to model name and log tags
        self.mlflow_logger.log_tags(
            {"mlflow.runName": self.model_name, "text_model_name": self.model_name}
        )

        # Calculate scalar metrics using Evaluator
        class_names = [str(c) for c in long_tail_codes]
        evaluator = Evaluator(num_classes=len(long_tail_codes), class_names=class_names)

        metrics = evaluator.compute_metrics(y_true, y_pred)

        logger.info(f"Evaluation Metrics: {metrics}")
        self.mlflow_logger.log_metrics(metrics, step=0)

        # Generate plots using Evaluator
        tags = {"text_model_name": self.model_name}

        # Per-class metrics
        evaluator.save_per_class_metrics(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(
            f"artifacts/csv/per_class_metrics_{safe_model_name}.csv"
        )

        # Confusion Matrix
        evaluator.plot_global_confusion_matrix(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(
            f"artifacts/img/global_confusion_matrix_{safe_model_name}.png"
        )

        # ROC and PR curves
        evaluator.plot_roc_curve(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(
            f"artifacts/img/roc_curve_{safe_model_name}.png"
        )

        evaluator.plot_pr_curve(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(f"artifacts/img/pr_curve_{safe_model_name}.png")

    def run(self):
        logger.info(
            f"Starting LLM classification (threshold={self.threshold}, limit={self.limit})"
        )

        long_tail_codes, descriptions, factory, classifier = self._load_resources()

        if not classifier:
            return

        predictions_data, all_true_labels, all_pred_labels = self._run_inference(
            factory, classifier, long_tail_codes
        )

        self._save_predictions(predictions_data)
        self._evaluate(all_true_labels, all_pred_labels, long_tail_codes)
