import csv
from typing import List


def get_long_tail_labels(csv_path: str, threshold: int) -> List[int]:
    """
    Reads the label distribution CSV and returns a list of label codes
    that have fewer occurrences than the specified threshold.
    """
    long_tail_codes = []

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    count = int(row["Count"])
                    code = int(row["Code"])
                    if count < threshold:
                        long_tail_codes.append(code)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Returning empty list.")
        return []

    return long_tail_codes


def load_descriptions(csv_path: str, filter_codes: List[int]) -> dict[int, str]:
    """
    Loads descriptions for the specified codes from the CSV file.
    """
    descriptions = {}
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    code = int(row["Code"])
                    if code in filter_codes:
                        descriptions[code] = row["Waarde"]
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found.")
        return {}

    return descriptions
