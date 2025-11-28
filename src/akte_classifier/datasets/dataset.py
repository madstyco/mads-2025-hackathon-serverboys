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
                return None, self.labels[idx], self.regex_features[idx]

        if self.embeddings is None:
            raise ValueError("Both embeddings and regex_features cannot be None")

        return self.embeddings[idx], self.labels[idx]


class DatasetFactory:
    def __init__(self, file_path: str, batch_size: int = 32, split_ratio: float = 0.8):
        self.file_path: str = file_path
        self.batch_size: int = batch_size
        self.split_ratio: float = split_ratio

        self.train_dataset: Optional[RechtsfeitDataset] = None
        self.test_dataset: Optional[RechtsfeitDataset] = None
        self.encoder: Optional[LabelEncoder] = None

        self._prepare_data()

    def _prepare_data(self) -> None:
        logger.info(f"Loading {self.file_path}...")
        full_dataset = load_dataset("json", data_files=self.file_path, split="train")

        split_ds = full_dataset.train_test_split(train_size=self.split_ratio, seed=42)
        train_data = split_ds["train"]
        test_data = split_ds["test"]

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
    ) -> Dict[str, VectorizedRechtsfeitDataset]:
        """
        Pre-computes embeddings for the entire dataset using the provided vectorizer.
        If regex_vectorizer is provided, also computes regex features.
        Results are cached to disk.
        """
        from akte_classifier.utils.tensor import load_or_compute_tensor

        vectorized_datasets = {}
        splits = ["train", "test"]

        # Create a slug for the model name to use in filenames
        if vectorizer:
            model_name_slug = vectorizer.model_name.replace("/", "_")
            device = next(vectorizer.model.parameters()).device
            logger.info(
                f"Vectorizing datasets on {device} for model {vectorizer.model_name}..."
            )
        else:
            model_name_slug = "no_text_model"
            logger.info("No text vectorizer provided. Skipping text embeddings.")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Ensure base datasets are loaded
        self.get_dataset()

        for split in splits:
            logger.info(f"Processing {split} set...")
            dataset = self.train_dataset if split == "train" else self.test_dataset

            # 1. Embeddings (Only if vectorizer is provided)
            embeddings = None
            if vectorizer:
                emb_path = cache_path / f"{model_name_slug}_{split}_embeddings.pt"

                def compute_embeddings():
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
                    return torch.cat(all_embeddings, dim=0)

                embeddings = load_or_compute_tensor(emb_path, compute_embeddings)

            # 2. Labels
            # We use the model slug in the label filename too, to keep them paired,
            # though labels are technically model-independent.
            # But if we change the split ratio, everything changes.
            # Ideally labels should be cached by split/dataset hash.
            # For now, let's keep using model_name_slug or a default if None.
            lbl_path = cache_path / f"{model_name_slug}_{split}_labels.pt"

            def compute_labels():
                all_labels = []
                dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
                for _, batch_labels in dataloader:
                    all_labels.append(batch_labels)
                return torch.cat(all_labels, dim=0)

            labels = load_or_compute_tensor(lbl_path, compute_labels)

            # 3. Regex Features (Optional)
            regex_features = None
            if regex_vectorizer:
                # Use the regex hash for the filename to ensure cache invalidation on regex changes
                regex_hash = getattr(regex_vectorizer, "hash", "default")
                regex_path = cache_path / f"regex_{regex_hash}_{split}.pt"

                def compute_regex():
                    all_regex = []
                    # Regex vectorizer might be CPU bound, batch size matters less for GPU memory but good for progress bar
                    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
                    for batch_texts, _ in tqdm(
                        dataloader, desc=f"Regex Vectorizing {split}"
                    ):
                        feats = regex_vectorizer.forward(list(batch_texts))
                        all_regex.append(feats)
                    return torch.cat(all_regex, dim=0)

                regex_features = load_or_compute_tensor(regex_path, compute_regex)

            vectorized_datasets[split] = VectorizedRechtsfeitDataset(
                labels=labels, embeddings=embeddings, regex_features=regex_features
            )

        return vectorized_datasets

    def get_vectorized_loader(
        self, vectorizer, regex_vectorizer=None
    ) -> Dict[str, DataLoader]:
        """
        Returns DataLoaders for pre-vectorized data.
        """
        datasets = self.get_vectorized_dataset(vectorizer, regex_vectorizer)

        train_loader = DataLoader(
            datasets["train"], batch_size=self.batch_size, shuffle=True
        )

        test_loader = DataLoader(
            datasets["test"], batch_size=self.batch_size, shuffle=False
        )

        return {"train": train_loader, "test": test_loader}


if __name__ == "__main__":
    logger.info("Starting dataset processing...")
    from pathlib import Path

    datadir = Path("data/raw")
    file_path = "aktes.jsonl"
    path = datadir / file_path

    factory = DatasetFactory(str(path), batch_size=2, split_ratio=0.6)
    dataset = factory.get_dataset()
    train = dataset["train"]
    X, y = train[0]
    logger.info(f"First x {X[:100]}, type {type(X)}")
    logger.info(f"First y {y}, type {type(y)}")

    loaders = factory.get_loader()

    # Assert encoder is initialized for mypy
    assert factory.encoder is not None

    logger.info("\n--- Testing Train Batch ---")
    for batch_texts, batch_labels in loaders["train"]:
        logger.info(f"Batch text count: {len(batch_texts)}")
        logger.info(f"Batch labels shape: {batch_labels.shape}")
        # Decode the first label in the batch
        logger.info(f"Sample restored codes: {factory.encoder.decode(batch_labels[0])}")
        break

    logger.info("\n--- Testing Test Batch ---")
    for batch_texts, batch_labels in loaders["test"]:
        logger.info(f"Batch text count: {len(batch_texts)}")
        logger.info(f"Batch labels shape: {batch_labels.shape}")
        logger.info(f"Sample restored codes: {factory.encoder.decode(batch_labels[0])}")

        # Check if we have any unknowns (index 0)
        if batch_labels[0][0] == 1.0:
            logger.info("-> Found an UNKNOWN code in this sample!")
        break
