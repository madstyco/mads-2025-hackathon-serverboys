import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class TextVectorizer:
    """
    Wraps a HuggingFace model to vectorize text.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to eval mode by default

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes a list of texts into vectors.
        Uses mean pooling of the last hidden state.
        """
        # Tokenize
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        # batches of text are padded such that they have the same length,
        # however, the padding tokens are zero and we dont want to include them in the mean
        attention_mask = inputs["attention_mask"]
        # attention_mask shape: (batch_size, seq_len)
        token_embeddings = outputs.last_hidden_state
        # last_hidden_state shape: (batch_size, seq_len, hidden_dim)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

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

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
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
    A classifier that combines text embeddings and regex features.
    """

    def __init__(
        self, input_dim: int, regex_dim: int, num_classes: int, hidden_dim: int = 256
    ):
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

    def forward(
        self, text_emb: torch.Tensor, regex_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass. Concatenates inputs and returns logits.
        """
        # Concatenate along the feature dimension (dim=1)
        combined = torch.cat((text_emb, regex_feats), dim=1)
        x = self.sequential(combined)
        return x
