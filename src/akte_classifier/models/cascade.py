"""
Cascade classifier combining hybrid model with LLM fallback.

Architecture:
    1. Hybrid Model (regex + neural) → confidence check
    2. If confidence < threshold → LLM fallback
    3. Otherwise → use hybrid prediction
"""

import csv
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

from akte_classifier.models.llm import LLMClassifier


class CascadeClassifier:
    """
    Cascade classifier with confidence-based LLM fallback.

    Architecture:
        1. Hybrid Model (regex + neural) → confidence check
        2. If confidence < threshold → LLM fallback
        3. Otherwise → use hybrid prediction
    """

    def __init__(
        self,
        hybrid_model: torch.nn.Module,
        llm_classifier: LLMClassifier,
        label_encoder: Any,
        confidence_threshold: float = 0.99,
        device: str = "cpu",
    ):
        """
        Args:
            hybrid_model: Trained HybridClassifier model
            llm_classifier: LLMClassifier instance
            label_encoder: Label encoder (code <-> index mapping)
            confidence_threshold: Confidence threshold for LLM fallback (default: 0.99)
            device: Device to run hybrid model on
        """
        self.hybrid_model = hybrid_model
        self.llm_classifier = llm_classifier
        self.label_encoder = label_encoder
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.hybrid_model.to(device)
        self.hybrid_model.eval()

        # Statistics tracking
        self.stats = {
            "total_predictions": 0,
            "hybrid_predictions": 0,
            "llm_fallback_predictions": 0,
            "hybrid_confidences": [],
        }

        logger.info(f"CascadeClassifier initialized with threshold={confidence_threshold}")

    def predict(
        self,
        text: str,
        text_embedding: torch.Tensor,
        regex_features: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Predict with cascade: hybrid first, LLM fallback if low confidence.

        Args:
            text: Raw document text
            text_embedding: Text embedding tensor [1, embedding_dim]
            regex_features: Regex features tensor [1, regex_dim]

        Returns:
            Dictionary with:
                - prediction: Predicted code (str)
                - confidence: Confidence score (float)
                - method: "hybrid" or "llm"
                - hybrid_top5: Top 5 hybrid predictions (if hybrid was used)
                - hybrid_confidence: Hybrid confidence (if LLM was used)
                - llm_codes: All LLM predicted codes (if LLM was used)
        """
        self.stats["total_predictions"] += 1

        # Get hybrid prediction
        with torch.no_grad():
            text_emb = text_embedding.to(self.device)
            regex_feat = regex_features.to(self.device)

            # Forward pass through hybrid model
            logits = self.hybrid_model(text_emb, regex_feat)
            probs = F.softmax(logits, dim=1)

            # Get top prediction and confidence
            max_prob, pred_idx = torch.max(probs, dim=1)
            confidence = max_prob.item()
            predicted_code = self.label_encoder.idx2code.get(pred_idx.item(), 0)  # 0 for unknown

            # Get top 5 for reference
            top5_probs, top5_indices = torch.topk(probs, min(5, probs.shape[1]), dim=1)
            hybrid_top5 = [
                {
                    "code": self.label_encoder.idx2code.get(idx.item(), 0),
                    "confidence": prob.item(),
                }
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]

        # Track hybrid confidence
        self.stats["hybrid_confidences"].append(confidence)

        # Decision: Use hybrid or fallback to LLM?
        if confidence >= self.confidence_threshold:
            # High confidence → use hybrid
            self.stats["hybrid_predictions"] += 1
            return {
                "prediction": predicted_code,
                "confidence": confidence,
                "method": "hybrid",
                "hybrid_top5": hybrid_top5,
            }

        # Low confidence → fallback to LLM
        self.stats["llm_fallback_predictions"] += 1
        logger.debug(f"Hybrid confidence {confidence:.4f} < {self.confidence_threshold}, " "using LLM fallback")

        # Call LLM
        llm_codes = self.llm_classifier.classify(text)

        # LLM returns list of codes (can be multiple or empty)
        if not llm_codes:
            # LLM failed or returned empty - fall back to hybrid prediction
            logger.warning("LLM returned empty, falling back to hybrid prediction")
            return {
                "prediction": predicted_code,
                "confidence": confidence,
                "method": "llm_failed_hybrid_fallback",
                "hybrid_confidence": confidence,
                "hybrid_top5": hybrid_top5,
            }

        # Use first LLM code (most confident)
        llm_primary_code = llm_codes[0]

        # Convert to string if needed
        if isinstance(llm_primary_code, int):
            llm_primary_code = str(llm_primary_code)

        return {
            "prediction": llm_primary_code,
            "confidence": 0.99,  # Assume high confidence for LLM
            "method": "llm",
            "hybrid_confidence": confidence,
            "hybrid_top5": hybrid_top5,
            "llm_codes": llm_codes,
        }

    def get_statistics(self) -> dict:
        """Get cascade statistics."""
        total = self.stats["total_predictions"]
        hybrid_count = self.stats["hybrid_predictions"]
        llm_count = self.stats["llm_fallback_predictions"]

        avg_hybrid_conf = (
            sum(self.stats["hybrid_confidences"]) / len(self.stats["hybrid_confidences"])
            if self.stats["hybrid_confidences"]
            else 0.0
        )

        return {
            "total_predictions": total,
            "hybrid_predictions": hybrid_count,
            "llm_fallback_predictions": llm_count,
            "hybrid_percentage": (hybrid_count / total * 100) if total > 0 else 0.0,
            "llm_percentage": (llm_count / total * 100) if total > 0 else 0.0,
            "avg_hybrid_confidence": avg_hybrid_conf,
            "confidence_threshold": self.confidence_threshold,
        }

    def print_statistics(self):
        """Print cascade statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("📊 CASCADE CLASSIFIER STATISTICS")
        print("=" * 60)
        print(f"Total predictions:           {stats['total_predictions']}")
        print(f"Hybrid predictions:          {stats['hybrid_predictions']} " f"({stats['hybrid_percentage']:.1f}%)")
        print(f"LLM fallback predictions:    {stats['llm_fallback_predictions']} " f"({stats['llm_percentage']:.1f}%)")
        print(f"Average hybrid confidence:   {stats['avg_hybrid_confidence']:.4f}")
        print(f"Confidence threshold:        {self.confidence_threshold:.2f}")
        print("=" * 60 + "\n")


def load_cascade_classifier(
    model_timestamp: str,
    confidence_threshold: float = 0.99,
    llm_model: str = "openai/gpt-oss-120b",
    llm_provider: str = "nebius",
) -> CascadeClassifier:
    """
    Load a trained hybrid model and create cascade classifier.

    Args:
        model_timestamp: Timestamp of trained model (e.g., "20251201_154538")
        confidence_threshold: Confidence threshold for LLM fallback
        llm_model: LLM model name (default: OpenAI GPT 120B via Nebius)
        llm_provider: Provider (nebius, openai, anthropic) - currently uses Nebius API

    Returns:
        CascadeClassifier instance
    """
    import glob
    import json

    from akte_classifier.datasets.dataset import LabelEncoder

    logger.info(f"Loading hybrid model with timestamp {model_timestamp}...")

    # Find model files
    model_dir = "artifacts/models"
    pt_files = glob.glob(f"{model_dir}/*_{model_timestamp}.pt")
    codes_files = glob.glob(f"{model_dir}/*_{model_timestamp}_codes.json")
    config_files = glob.glob(f"{model_dir}/*_{model_timestamp}_config.json")

    if len(pt_files) != 1 or len(codes_files) != 1:
        raise ValueError(
            f"Expected 1 model file and 1 codes file for timestamp {model_timestamp}, "
            f"found {len(pt_files)} and {len(codes_files)}"
        )

    model_path = pt_files[0]
    codes_path = codes_files[0]
    config_path = config_files[0] if config_files else None

    # Load encoder codes
    logger.info(f"Loading encoder codes from {codes_path}...")
    with open(codes_path) as f:
        encoder_codes = json.load(f)

    # Load config
    if config_path:
        with open(config_path) as f:
            config_data = json.load(f)
        hidden_dim = config_data.get("hidden_dim", 256)
    else:
        logger.warning("Config file not found, using default hidden_dim=256")
        hidden_dim = 256

    # Create label encoder
    label_encoder = LabelEncoder(train_codes=encoder_codes)

    # Load weights first to get dimensions
    logger.info(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")

    # Extract dimensions from state_dict
    text_dim = state_dict["text_pathway.0.weight"].shape[1]  # Input dim of first layer
    regex_dim = state_dict["regex_pathway.0.weight"].shape[1]  # Regex features
    num_classes = state_dict["output_layer.weight"].shape[0]  # Number of classes

    logger.info(
        f"Model dimensions from checkpoint: text_dim={text_dim}, " f"regex_dim={regex_dim}, num_classes={num_classes}"
    )

    # Create model instance with correct dimensions
    from akte_classifier.models.neural import HybridClassifier

    hybrid_model = HybridClassifier(
        input_dim=text_dim,
        regex_dim=regex_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )

    # Load the state dict
    hybrid_model.load_state_dict(state_dict)
    hybrid_model.eval()

    # Load label descriptions for LLM
    logger.info("Loading label descriptions from rechtsfeiten.csv...")
    descriptions = load_label_descriptions("assets/rechtsfeiten.csv")

    # Create LLM classifier
    logger.info(f"Initializing LLM classifier: {llm_model}")
    llm_classifier = LLMClassifier(
        model_name=llm_model,
        descriptions=descriptions,
    )

    # Create cascade
    cascade = CascadeClassifier(
        hybrid_model=hybrid_model,
        llm_classifier=llm_classifier,
        label_encoder=label_encoder,
        confidence_threshold=confidence_threshold,
        device="cpu",
    )

    logger.success("Cascade classifier loaded successfully")
    return cascade


def load_label_descriptions(csv_path: str) -> dict[int, str]:
    """
    Load label descriptions from rechtsfeiten.csv.

    Args:
        csv_path: Path to rechtsfeiten.csv

    Returns:
        Dictionary mapping code (int) to description (str)
    """
    descriptions = {}

    # Use utf-8-sig to handle BOM (Byte Order Mark) if present
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            code = int(row["Code"])
            value = row["Waarde"]
            descriptions[code] = value

    logger.info(f"Loaded {len(descriptions)} label descriptions")
    return descriptions
