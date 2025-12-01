import json
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loguru import logger
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer


class NebiusTextVectorizer:
    """
    Vectorizer using Nebius API embeddings for fast cloud-based inference.
    Much faster than local HuggingFace models.
    """

    # Model dimensions mapping
    MODEL_DIMS = {
        "BAAI/bge-multilingual-gemma2": 3584,
        "Qwen/Qwen3-Embedding-8B": 4096,
        "BAAI/bge-en-icl": 4096,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-multilingual-gemma2",
        max_length: int | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length or 8192  # Nebius models support long context

        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set")

        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key,
        )

        self._hidden_size = self.MODEL_DIMS.get(model_name, 3584)
        logger.success(f"NebiusTextVectorizer initialized with model={model_name}, " f"hidden_size={self._hidden_size}")

    def _embed_batch(self, batch: list[str], batch_idx: int) -> tuple[int, list]:
        """Embed a single batch with retry logic. Returns (batch_idx, embeddings)."""
        import time

        for attempt in range(3):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                return (batch_idx, [e.embedding for e in response.data])
            except Exception as e:
                if attempt < 2:
                    wait_time = 2**attempt
                    logger.warning(f"Nebius API error (batch {batch_idx}), retry in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Nebius API failed after 3 attempts: {e}")
                    raise
        return (batch_idx, [])

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes texts using Nebius API with concurrent requests.
        Uses ThreadPoolExecutor for parallel API calls.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Truncate texts (Nebius has input limits)
        max_chars = 8000  # ~2000 tokens
        truncated_texts = [t[:max_chars] for t in texts]

        # Split into batches
        api_batch_size = 16  # Smaller batches, but run concurrently
        batches = []
        for i in range(0, len(truncated_texts), api_batch_size):
            batches.append(truncated_texts[i : i + api_batch_size])

        # Process batches concurrently (4 parallel requests)
        results = [None] * len(batches)
        max_workers = 4

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._embed_batch, batch, idx): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                batch_idx, embeddings = future.result()
                results[batch_idx] = embeddings

        # Flatten results in order
        all_embeddings = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)

        return torch.tensor(all_embeddings, dtype=torch.float32)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size


class TextVectorizer:
    """
    Wraps a HuggingFace model to vectorize text.
    """

    def __init__(self, model_name: str, max_length: int | None = None, pooling: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to eval mode by default

        if pooling is None:
            self.pooling = self._detect_pooling_strategy(model_name)
        else:
            self.pooling = pooling

        # Determine max_length
        if max_length is not None:
            self.max_length = max_length
        else:
            # Try to get from tokenizer
            model_max_len = self.tokenizer.model_max_length
            # HuggingFace tokenizers often return a very large int if not set
            # We treat anything > 10000 as "infinite"/unset and fallback to 512
            if model_max_len > 10000:
                logger.warning(
                    f"Tokenizer model_max_length is {model_max_len}, falling back to 512. "
                    "Specify max_length explicitly if needed."
                )
                self.max_length = 512
            else:
                self.max_length = model_max_len

        logger.info(f"TextVectorizer initialized with max_length={self.max_length}, pooling={self.pooling}")

    def _detect_pooling_strategy(self, model_name: str) -> str:
        """
        Attempts to detect the pooling strategy from the model's configuration.
        """
        try:
            # Download modules.json
            path = hf_hub_download(model_name, "modules.json")
            with open(path) as f:
                modules = json.load(f)

            # Look for the pooling module
            for module in modules:
                if module["type"] == "sentence_transformers.models.Pooling":
                    # Download the pooling config
                    config_path = hf_hub_download(model_name, f"{module['path']}/config.json")
                    with open(config_path) as f:
                        config = json.load(f)

                    if config.get("pooling_mode_cls_token"):
                        logger.success(f"Auto-detected pooling strategy: cls (from {model_name})")
                        return "cls"
                    if config.get("pooling_mode_mean_tokens"):
                        logger.success(f"Auto-detected pooling strategy: mean (from {model_name})")
                        return "mean"
        except Exception:
            # Fail silently on network errors or missing files
            pass

        logger.warning(
            f"Could not detect pooling strategy for {model_name}, defaulting to 'mean'. "
            "Please check the model card to see if 'cls' pooling is required."
        )
        return "mean"

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes a list of texts into vectors.
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.pooling == "cls":
            # CLS token is usually the first token
            return outputs.last_hidden_state[:, 0, :]

        # Default to mean pooling
        # batches of text are padded such that they have the same length,
        # however, the padding tokens are zero and we dont want to include them in the mean
        attention_mask = inputs["attention_mask"]
        # attention_mask shape: (batch_size, seq_len)
        token_embeddings = outputs.last_hidden_state
        # last_hidden_state shape: (batch_size, seq_len, hidden_dim)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Zero out padding tokens so they don't skew the sum, then
        # divide by the count of real tokens (not total length) for a true average.
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size


class NeuralClassifier(nn.Module):
    """
    A simple neural network for classification.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns logits.
        """
        x = self.sequential(x)
        return x


class HybridClassifier(nn.Module):
    """
    Improved classifier combining text embeddings and regex features.

    Enhancements:
    - Separate processing pathways for each modality
    - Skip connections for better gradient flow
    - Batch normalization for stable training
    - Gating mechanism to dynamically weight modalities
    """

    def __init__(self, input_dim: int, regex_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.regex_dim = regex_dim
        self.hidden_dim = hidden_dim

        # Separate pathways for each modality
        # Text embedding pathway
        self.text_pathway = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Regex features pathway
        self.regex_pathway = nn.Sequential(
            nn.Linear(regex_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Gating mechanism: learn how much to trust each modality
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())  # Output between 0 and 1

        # Combined processing with skip connection
        combined_dim = hidden_dim * 2
        self.fusion_layer1 = nn.Linear(combined_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fusion_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.2)

        logger.info(
            f"ImprovedHybridClassifier initialized: "
            f"text_dim={input_dim}, regex_dim={regex_dim}, "
            f"hidden_dim={hidden_dim}, num_classes={num_classes}"
        )

    def forward(self, text_emb: torch.Tensor, regex_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with separate modality processing and gating.

        Args:
            text_emb: [batch_size, input_dim]
            regex_feats: [batch_size, regex_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        # Process each modality separately
        text_features = self.text_pathway(text_emb)  # [batch, hidden_dim]
        regex_features = self.regex_pathway(regex_feats)  # [batch, hidden_dim]

        # Concatenate for fusion
        combined = torch.cat([text_features, regex_features], dim=1)  # [batch, hidden_dim*2]

        # Gating: learn to weight text vs regex dynamically
        gate_weights = self.gate(combined)  # [batch, hidden_dim]

        # First fusion layer with skip connection
        x = self.fusion_layer1(combined)
        x = self.bn1(x)
        x = torch.relu(x)

        # Apply gate to modulate features
        x = x * gate_weights

        # Save for skip connection
        skip = x

        # Second fusion layer
        x = self.fusion_layer2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Skip connection: add original features
        x = x + skip

        # Output layer
        logits = self.output_layer(x)

        return logits


class HybridClassifierSimple(nn.Module):
    """
    Original simple hybrid classifier (kept for comparison).
    A classifier that combines text embeddings and regex features.
    """

    def __init__(self, input_dim: int, regex_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()

        # Concatenated input dimension
        combined_dim = input_dim + regex_dim
        logger.info(f"Combined dimension: {combined_dim}")

        self.sequential = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, regex_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Concatenates inputs and returns logits.
        """
        # Concatenate along the feature dimension (dim=1)
        combined = torch.cat((text_emb, regex_feats), dim=1)
        x = self.sequential(combined)
        return x
