import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dataset.log", rotation="10 MB", level="DEBUG")


class LabelEncoder:
    """
    Helper to map raw sparse integer codes (504, 508, etc.) to dense indices (0, 1, ..., N) and back.
    Reserves index 0 for 'Unknown' codes.
    """

    def __init__(self, train_codes: List[int]):
        # Get unique codes and sort them for deterministic behavior
        self.unique_codes: List[int] = sorted(list(set(train_codes)))

        # Reserve 0 for unknown
        self.unknown_idx: int = 0

        # Create mappings
        # We start real codes from index 1
        self.code2idx: Dict[int, int] = {
            code: i + 1 for i, code in enumerate(self.unique_codes)
        }
        self.idx2code: Dict[int, int] = {
            i + 1: code for i, code in enumerate(self.unique_codes)
        }

        # Add unknown mapping
        self.idx2code[self.unknown_idx] = 0

    def __len__(self) -> int:
        # +1 for the unknown category
        return len(self.unique_codes) + 1

    def encode(self, codes: List[int]) -> torch.Tensor:
        """Converts a list of raw codes into a Multi-Hot Tensor."""
        # Create a vector of zeros with length equal to total classes
        vector = torch.zeros(len(self), dtype=torch.float32)

        # Set indices to 1 for present codes
        for code in codes:
            if code in self.code2idx:
                vector[self.code2idx[code]] = 1.0
            else:
                # Map to unknown if not found
                vector[self.unknown_idx] = 1.0
        return vector

    def decode(self, vector: torch.Tensor) -> List[int]:
        """Converts a Multi-Hot Tensor/Probabilities back to raw codes."""
        # Get indices where value is high (assuming threshold 0.5 for logits)
        indices = (vector > 0.5).nonzero(as_tuple=True)[0]
        decoded = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val in self.idx2code:
                decoded.append(self.idx2code[idx_val])
            else:
                # Should not happen if logic is correct, but safe fallback
                decoded.append(0)
        return decoded


class RechtsfeitDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset, encoder: LabelEncoder):
        self.dataset = hf_dataset
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        item = self.dataset[idx]
        text: str = item["text"]
        raw_codes: List[int] = item["rechtsfeitcodes"]

        # Transform raw codes to Multi-Hot Tensor
        label_tensor: torch.Tensor = self.encoder.encode(raw_codes)

        return text, label_tensor


class VectorizedRechtsfeitDataset(Dataset):
    def __init__(
        self,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        regex_features: Optional[torch.Tensor] = None,
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.regex_features = regex_features

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[None, torch.Tensor, torch.Tensor],
    ]:
        # Return format: (embeddings, labels, regex_features)
        # If embeddings is None, we return None for it

        if self.regex_features is not None:
            if self.embeddings is not None:
                return self.embeddings[idx], self.labels[idx], self.regex_features[idx]
            else:
                # If we only have regex features (e.g. RegexOnlyClassifier),
                # we return them as the "input" features to match the (inputs, labels) signature
                # expected by the trainer's default loop.
                return self.regex_features[idx], self.labels[idx]

        if self.embeddings is None:
            raise ValueError("Both embeddings and regex_features cannot be None")

        return self.embeddings[idx], self.labels[idx]


class DatasetFactory:
    def __init__(
        self,
        file_path: str,
        batch_size: int = 32,
        split_ratio: float = 0.8,
        long_tail_threshold: Optional[int] = None,
        encoder_codes: Optional[List[int]] = None,
    ):
        self.file_path: str = file_path
        self.batch_size: int = batch_size
        self.split_ratio: float = split_ratio
        self.long_tail_threshold: Optional[int] = long_tail_threshold
        self.encoder_codes: Optional[List[int]] = encoder_codes

        self.train_dataset: Optional[RechtsfeitDataset] = None
        self.test_dataset: Optional[RechtsfeitDataset] = None
        self.encoder: Optional[LabelEncoder] = None

        self._prepare_data()

    def _prepare_data(self) -> None:
        logger.info(f"Loading {self.file_path}...")
        full_dataset = load_dataset("json", data_files=self.file_path, split="train")

        # Filter for long-tail if requested
        if self.long_tail_threshold is not None:
            from akte_classifier.utils.data import get_long_tail_labels

            # Assuming label_distribution.csv is in artifacts/csv/
            # Ideally this path should be configurable, but for now we hardcode relative to project root
            dist_path = "artifacts/csv/label_distribution.csv"
            long_tail_codes = set(
                get_long_tail_labels(dist_path, self.long_tail_threshold)
            )
            logger.info(
                f"Filtering for {len(long_tail_codes)} long-tail labels (threshold < {self.long_tail_threshold})"
            )

            def filter_long_tail(example):
                # Keep example if it has at least one long-tail code
                example_codes = [int(c) for c in example["rechtsfeitcodes"]]
                return any(c in long_tail_codes for c in example_codes)

            original_len = len(full_dataset)
            full_dataset = full_dataset.filter(filter_long_tail)
            logger.info(
                f"Filtered dataset from {original_len} to {len(full_dataset)} samples."
            )

        if self.split_ratio == 1.0:
            train_data = full_dataset
            # Create empty dataset for test
            test_data = full_dataset.select([])
        else:
            split_ds = full_dataset.train_test_split(
                train_size=self.split_ratio, seed=42
            )
            train_data = split_ds["train"]
            test_data = split_ds["test"]

        # Build encoder
        if self.encoder_codes:
            logger.info("Using provided encoder codes...")
            self.encoder = LabelEncoder(self.encoder_codes)
        else:
            # Build encoder based ONLY on training data
            logger.debug("Building label index from training data...")
            train_codes: List[int] = []
            for codes in train_data["rechtsfeitcodes"]:
                # Ensure all codes are integers to avoid duplicates (str vs int)
                train_codes.extend([int(c) for c in codes])

            self.encoder = LabelEncoder(train_codes)

        logger.debug(
            f"Found {len(self.encoder)} unique categories (including Unknown)."
        )

        self.train_dataset = RechtsfeitDataset(train_data, self.encoder)
        self.test_dataset = RechtsfeitDataset(test_data, self.encoder)
        logger.success("Data prepared successfully.")

    def get_dataset(self) -> Dict[str, RechtsfeitDataset]:
        if self.train_dataset is None or self.test_dataset is None:
            self._prepare_data()

        assert self.train_dataset is not None
        assert self.test_dataset is not None

        return {"train": self.train_dataset, "test": self.test_dataset}

    def get_loader(self) -> Dict[str, DataLoader]:
        datasets = self.get_dataset()

        train_loader = DataLoader(
            datasets["train"], batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        test_loader = DataLoader(
            datasets["test"], batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        return {"train": train_loader, "test": test_loader}

    def get_vectorized_dataset(
        self,
        vectorizer=None,
        regex_vectorizer=None,
        cache_dir: str = "artifacts/vectorcache",
        splits: Optional[List[str]] = None,
    ) -> Dict[str, VectorizedRechtsfeitDataset]:
        """
        Pre-computes embeddings for the entire dataset using the provided vectorizer.
        If regex_vectorizer is provided, also computes regex features.
        Results are cached to disk.
        """
        from akte_classifier.utils.tensor import load_or_compute_tensor

        vectorized_datasets = {}
        if splits is None:
            splits = ["train", "test"]

        # Create a slug for the model name to use in filenames
        if vectorizer:
            model_name_slug = vectorizer.model_name.replace("/", "_")
            # Handle both local (TextVectorizer) and API (NebiusTextVectorizer)
            if hasattr(vectorizer, 'model'):
                device = next(vectorizer.model.parameters()).device
                logger.info(
                    f"Vectorizing datasets on {device} for model {vectorizer.model_name}..."
                )
            else:
                # API-based vectorizer (Nebius) - no local model
                device = "cpu"  # Embeddings come from API
                logger.info(
                    f"Vectorizing datasets via API for model {vectorizer.model_name}..."
                )
        else:
            model_name_slug = "no_text_model"
            logger.info("No text vectorizer provided. Skipping text embeddings.")

        # Compute hash of the file path to ensure cache uniqueness per dataset file
        import hashlib

        file_hash = hashlib.md5(
            str(Path(self.file_path).absolute()).encode()
        ).hexdigest()[:8]

        # Include split_ratio in the hash/filename to avoid collisions between
        # full dataset (eval) and split dataset (train)
        split_slug = f"split{int(self.split_ratio * 100)}"

        logger.debug(f"File hash for {self.file_path}: {file_hash} ({split_slug})")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Ensure base datasets are loaded
        self.get_dataset()

        for split in splits:
            logger.info(f"Processing {split} set...")
            dataset = self.train_dataset if split == "train" else self.test_dataset

            # Determine cache name for the split
            # If we are using the full dataset (split_ratio=1.0) and this is the 'train' slot,
            # we call it 'full' in the cache filename to be less confusing.
            if self.split_ratio == 1.0 and split == "train":
                cache_split_name = "full"
            else:
                cache_split_name = split

            # 1. Embeddings (Only if vectorizer is provided)
            embeddings = None
            if vectorizer:
                emb_path = (
                    cache_path
                    / f"{model_name_slug}_{file_hash}_{split_slug}_{cache_split_name}_embeddings.pt"
                )

                def compute_embeddings():
                    if len(dataset) == 0:
                        return torch.empty(0)

                    all_embeddings = []
                    # Use a larger batch size for inference if possible
                    batch_size = 64
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=False
                    )

                    with torch.no_grad():
                        for batch_texts, _ in tqdm(
                            dataloader, desc=f"Vectorizing {split}"
                        ):
                            emb = vectorizer.forward(list(batch_texts))
                            all_embeddings.append(emb.cpu())

                    if not all_embeddings:
                        return torch.empty(0)

                    return torch.cat(all_embeddings, dim=0)

                embeddings = load_or_compute_tensor(emb_path, compute_embeddings)

            # 2. Labels
            # We use the model slug in the label filename too, to keep them paired,
            # though labels are technically model-independent.
            # But if we change the split ratio, everything changes.
            # Ideally labels should be cached by split/dataset hash.
            # For now, let's keep using model_name_slug or a default if None.
            lbl_path = (
                cache_path
                / f"{model_name_slug}_{file_hash}_{split_slug}_{cache_split_name}_labels.pt"
            )

            def compute_labels():
                if len(dataset) == 0:
                    return torch.empty(0)

                all_labels = []
                dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
                for _, batch_labels in dataloader:
                    all_labels.append(batch_labels)

                if not all_labels:
                    return torch.empty(0)

                return torch.cat(all_labels, dim=0)

            labels = load_or_compute_tensor(lbl_path, compute_labels)

            # 3. Regex Features (Optional)
            regex_features = None
            if regex_vectorizer:
                # Use the regex hash for the filename to ensure cache invalidation on regex changes
                regex_hash = getattr(regex_vectorizer, "hash", "default")
                regex_path = (
                    cache_path
                    / f"regex_{regex_hash}_{file_hash}_{split_slug}_{cache_split_name}.pt"
                )

                def compute_regex():
                    if len(dataset) == 0:
                        return torch.empty(0)

                    all_regex = []
                    # Regex vectorizer might be CPU bound, batch size matters less for GPU memory but good for progress bar
                    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
                    for batch_texts, _ in tqdm(
                        dataloader, desc=f"Regex Vectorizing {split}"
                    ):
                        feats = regex_vectorizer.forward(list(batch_texts))
                        all_regex.append(feats)

                    if not all_regex:
                        return torch.empty(0)

                    return torch.cat(all_regex, dim=0)

                regex_features = load_or_compute_tensor(regex_path, compute_regex)

            vectorized_datasets[split] = VectorizedRechtsfeitDataset(
                labels=labels, embeddings=embeddings, regex_features=regex_features
            )

        return vectorized_datasets

    def get_vectorized_loader(
        self, vectorizer, regex_vectorizer=None, splits: Optional[List[str]] = None
    ) -> Dict[str, DataLoader]:
        """
        Returns DataLoaders for pre-vectorized data.
        """
        datasets = self.get_vectorized_dataset(
            vectorizer, regex_vectorizer, splits=splits
        )

        loaders = {}
        if "train" in datasets:
            loaders["train"] = DataLoader(
                datasets["train"], batch_size=self.batch_size, shuffle=True
            )

        if "test" in datasets:
            loaders["test"] = DataLoader(
                datasets["test"], batch_size=self.batch_size, shuffle=False
            )

        return loaders
