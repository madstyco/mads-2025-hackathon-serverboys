# Analyseer structuur van aktes
import json
import re

from collections import Counter
akte_types = Counter()
title_patterns = Counter()

with open('assets/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        text = data['text']
        
        # Extract title
        title_match = re.search(r'De bewaarder\.\s*([A-Z][A-Z\s\/\*\-]+?)(?:\s+(?:Kenmerk|Heden|Op|Vandaag))', text)
        if title_match:
            title = title_match.group(1).strip()
            title_patterns[title[:50]] += 1
        
        # Analyseer lengte
        codes = data['rechtsfeitcodes']
        akte_types[len(codes)] += 1