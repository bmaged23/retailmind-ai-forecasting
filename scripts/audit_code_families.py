"""Analyze possible stock-code groups using descriptions known at cutoff."""
from pathlib import Path
from collections import Counter
import csv
import json
import random
from retailmind.family_audit import audit

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/generated'
    with (folder / 'product-features-all.csv').open(newline='', encoding='utf-8') as handle:
        groups = audit(csv.DictReader(handle))
    output = folder / 'candidate-family-audit.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=groups[0].keys())
        writer.writeheader(); writer.writerows(groups)
    eligible = sorted(groups, key=lambda x: x['candidate_code_base'])
    selection = random.Random(722).sample(eligible, min(30, len(eligible)))
    with (folder / 'candidate-family-random-review.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=groups[0].keys())
        writer.writeheader(); writer.writerows(selection)
    summary = {'groups': len(groups), 'products': sum(g['n_products'] for g in groups),
               'case_collision_groups': sum(g['case_only_collision'] for g in groups),
               'zero_token_overlap_groups': sum(g['min_token_similarity'] == 0 for g in groups),
               'no_comparable_tokens_groups': sum(g['min_token_similarity'] is None for g in groups),
               'warning': 'Token overlap is only a review heuristic; codes and descriptions do not prove common product identity.'}
    (folder / 'candidate-family-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('\nPotential case-only collisions:')
    for group in [g for g in groups if g['case_only_collision']][:10]:
        print(group['candidate_code_base'], group['codes'], group['descriptions'])

if __name__ == '__main__': main()
