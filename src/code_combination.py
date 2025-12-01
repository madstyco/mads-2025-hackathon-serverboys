# Script om te analyseren welke long-tail codes vaak samen voorkomen
import json
from collections import Counter, defaultdict
from itertools import combinations

# Definieer drempel voor long-tail
LONG_TAIL_THRESHOLD = 10

# Stap 1: Tel alle codes om te bepalen wat de long-tail is
total_code_counts = Counter()
with open('assets/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        codes = [int(c) for c in data['rechtsfeitcodes']]
        for code in codes:
            total_code_counts[code] += 1

# Stel de long-tail codes samen
long_tail_codes = {code for code, cnt in total_code_counts.items() if cnt < LONG_TAIL_THRESHOLD}

# Stap 2: Analyseer alleen combinaties die (ook) long-tail bevatten
code_pairs = Counter()
code_triplets = Counter()
code_counts = Counter()

with open('assets/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        codes = sorted([int(c) for c in data['rechtsfeitcodes']])
        # Selecteer alleen codes die in de long-tail zitten
        lt_codes = [c for c in codes if c in long_tail_codes]
        if not lt_codes:
            continue  # alleen doorgaan als er een long-tail code aanwezig is

        # Tel individuele long-tail codes
        for code in lt_codes:
            code_counts[code] += 1

        # Tel paren van long-tail codes – alleen paren waarvan beide codes long-tail zijn
        for pair in combinations(lt_codes, 2):
            code_pairs[pair] += 1

        # Tel triplets van long-tail codes
        if len(lt_codes) >= 3:
            for triplet in combinations(lt_codes, 3):
                code_triplets[triplet] += 1

# Print meest voorkomende combinaties (alleen long-tail codes)
print("Meest voorkomende long-tail code-paren:")
for (c1, c2), count in code_pairs.most_common(20):
    print(f"  {c1} + {c2}: {count} keer")