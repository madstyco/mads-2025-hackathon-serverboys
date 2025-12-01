"""
Utility functions for analyzing code patterns and combinations in the training data.
"""
import json
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple


def analyze_code_combinations(
    train_file: str, long_tail_codes: List[int], min_cooccurrence: int = 2
) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Analyzes which long-tail codes frequently occur together in the training data.
    
    Args:
        train_file: Path to train.jsonl
        long_tail_codes: List of long-tail code IDs
        min_cooccurrence: Minimum number of times a pair must occur to be included
        
    Returns:
        Dictionary with:
        - 'pairs': List of (code1, code2, count) tuples for frequent pairs
        - 'single_distribution': Counter of individual code frequencies
    """
    long_tail_set = set(long_tail_codes)
    code_pairs = Counter()
    code_counts = Counter()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            codes = sorted([int(c) for c in data['rechtsfeitcodes']])
            
            # Filter to only long-tail codes
            lt_codes = [c for c in codes if c in long_tail_set]
            
            if not lt_codes:
                continue
            
            # Count individual codes
            for code in lt_codes:
                code_counts[code] += 1
            
            # Count pairs
            for pair in combinations(lt_codes, 2):
                code_pairs[pair] += 1
    
    # Filter pairs by minimum co-occurrence
    frequent_pairs = [
        (c1, c2, count)
        for (c1, c2), count in code_pairs.items()
        if count >= min_cooccurrence
    ]
    # Sort by frequency (descending)
    frequent_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'pairs': frequent_pairs[:10],  # Top 10 most frequent pairs
        'single_distribution': code_counts
    }


def format_code_combinations_info(
    patterns: Dict[str, List[Tuple[int, int, int]]],
    descriptions: Dict[int, str]
) -> str:
    """
    Formats code combination patterns into a readable string for the prompt.
    
    Args:
        patterns: Output from analyze_code_combinations
        descriptions: Dictionary mapping code IDs to descriptions
        
    Returns:
        Formatted string describing common code combinations
    """
    if not patterns['pairs']:
        return "No frequent code combinations found in the training data."
    
    lines = ["Common code combinations observed in training data:"]
    for code1, code2, count in patterns['pairs']:
        desc1 = descriptions.get(code1, f"Code {code1}")
        desc2 = descriptions.get(code2, f"Code {code2}")
        lines.append(f"  - Codes {code1} ({desc1}) and {code2} ({desc2}) occur together {count} times")
    
    return "\n".join(lines)

