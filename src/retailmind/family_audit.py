"""Exploratory product-code family diagnostics, not an identity classifier."""
from collections import defaultdict
from itertools import combinations
import re

# Presentation-only words can obscure potentially related descriptions.
STOP = {'A', 'AN', 'AND', 'OF', 'THE', 'IN', 'WITH', 'FOR', 'SET', 'PACK',
        'RED', 'BLUE', 'GREEN', 'PINK', 'BLACK', 'WHITE', 'YELLOW', 'PURPLE',
        'SMALL', 'LARGE', 'MINI', 'BIG'}


def tokens(description):
    return {word for word in re.findall(r'[A-Z]+', description.upper()) if word not in STOP and len(word) > 1}


def similarity(first, second):
    a, b = tokens(first), tokens(second)
    return len(a & b) / len(a | b) if a and b else None


def audit(records):
    groups = defaultdict(list)
    for item in records:
        base = item.get('candidate_code_base')
        description = item.get('description')
        if base and description:
            groups[str(base)].append(item)
    report = []
    for base, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        scores = [similarity(a['description'], b['description']) for a, b in combinations(entries, 2)]
        scores = [score for score in scores if score is not None]
        normalized = [item['stock_code'].upper() for item in entries]
        report.append({'candidate_code_base': base, 'n_products': len(entries),
                       'min_token_similarity': min(scores) if scores else None,
                       'case_only_collision': len(normalized) != len(set(normalized)),
                       'codes': ' | '.join(item['stock_code'] for item in entries),
                       'descriptions': ' | '.join(item['description'] for item in entries)})
    return report
