import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from akte_classifier.datasets.dataset import DatasetFactory
from akte_classifier.models.neural import (HybridClassifier, NeuralClassifier,
                                           TextVectorizer)
from akte_classifier.models.regex import RegexGenerator, RegexVectorizer
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
    model_name: str = "prajjwal1/bert-mini"
    learning_rate: float = 1e-3
    num_epochs: int = 50
    device: str = get_default_device()
    model_class: str = "HybridClassifier"  # Default to Hybrid
    use_regex: bool = True  # Whether to use regex features


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
        logger.info("Initializing TextVectorizer & freezing weights...")
        self.vectorizer = TextVectorizer(self.config.model_name)
        self.vectorizer.model.to(self.device)
        for param in self.vectorizer.model.parameters():
            param.requires_grad = False

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
                input_dim=self.vectorizer.hidden_size, num_classes=self.num_classes
            )
        elif self.config.model_class == "HybridClassifier":
            self.classifier = HybridClassifier(
                input_dim=self.vectorizer.hidden_size,
                regex_dim=self.regex_vectorizer.output_dim,
                num_classes=self.num_classes,
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
