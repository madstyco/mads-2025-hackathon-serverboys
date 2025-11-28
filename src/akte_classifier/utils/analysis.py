from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from akte_classifier.datasets.dataset import RechtsfeitDataset


def analyze_label_distribution(dataset: RechtsfeitDataset) -> Dict[str, int]:
    """
    Analyzes and prints the distribution of labels in the dataset.
    Returns a dictionary of code -> count.
    """
    logger.debug("Analyzing label distribution...")

    # Access underlying HF dataset for speed
    hf_dataset = dataset.dataset
    all_codes = []

    # Iterate and collect all codes
    # The dataset has 'rechtsfeitcodes' column which is List[int]
    # We can iterate directly over the column if it's loaded in memory or stream it
    for codes in hf_dataset["rechtsfeitcodes"]:
        all_codes.extend(codes)

    # Count
    counts = Counter(all_codes)

    # Sort by count desc
    sorted_counts = counts.most_common()

    logger.info(f"Total samples: {len(hf_dataset)}")
    logger.info(f"Total label occurrences: {len(all_codes)}")
    logger.info(f"Unique labels found: {len(counts)}")

    # Save to CSV
    import csv

    total_occurrences = len(all_codes)
    labeled_counts = {}
    csv_output_path = "artifacts/csv/label_distribution.csv"
    with open(csv_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Code", "Count", "Percentage"])
        for code, count in sorted_counts:
            percentage = (count / total_occurrences) * 100
            writer.writerow([code, count, f"{percentage:.2f}%"])
            labeled_counts[str(code)] = count

    logger.success(f"Label distribution saved to {csv_output_path}")

    # Visualization
    logger.debug("Generating label distribution plot...")
    plt.figure(figsize=(15, 8))

    codes = [str(c) for c, _ in sorted_counts]
    values = [count for _, count in sorted_counts]

    sns.barplot(x=codes, y=values, hue=codes, palette="viridis", legend=False)
    plt.yscale("log")  # Log scale
    plt.xticks(rotation=90)
    plt.xlabel("Label Code")
    plt.ylabel("Count (Log Scale)")
    plt.title("Label Distribution")
    plt.tight_layout()

    output_path = "artifacts/img/label_distribution.png"
    plt.savefig(output_path)
    logger.success(f"Label distribution plot saved to {output_path}")

    return labeled_counts
