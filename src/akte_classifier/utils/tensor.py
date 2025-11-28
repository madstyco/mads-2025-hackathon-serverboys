from pathlib import Path
from typing import Callable

import torch
from loguru import logger


def load_or_compute_tensor(
    path: Path, compute_fn: Callable[[], torch.Tensor], force_recompute: bool = False
) -> torch.Tensor:
    """
    Loads a tensor from disk if it exists, otherwise computes it using the provided function
    and saves it to disk.

    Args:
        path: Path to the cache file (.pt).
        compute_fn: Function that returns the tensor if cache is missing.
        force_recompute: If True, ignores existing cache and recomputes.

    Returns:
        The loaded or computed tensor.
    """
    if path.exists() and not force_recompute:
        logger.info(f"Loading cached tensor from {path}...")
        try:
            tensor = torch.load(path)
            logger.success(f"Loaded cached tensor from {path}")
            return tensor
        except Exception as e:
            logger.warning(f"Failed to load cache from {path}: {e}. Recomputing...")

    logger.info("Computing tensor (cache miss or forced)...")
    tensor = compute_fn()

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.success(f"Computed tensor and saved to {path}")
    torch.save(tensor, path)

    return tensor
