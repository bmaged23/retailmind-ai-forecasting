"""Evaluate reviewed title counts without training any forecasting model."""
from pathlib import Path
import csv
import json
from retailmind.product_features import candidate_pack_size

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = {}
    for split in ('development', 'heldout', 'random-heldout'):
        with (ROOT / 'docs/reviews' / f'pack-{split}-labels.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        matched = false_positive = missed = incorrect = ambiguous_output = 0
        for row in rows:
            predicted = candidate_pack_size(row['description'])
            label = row['review_status']
            if label == 'explicit_count':
                expected = int(float(row['review_pack_size']))
                if predicted == expected: matched += 1
                elif predicted is None: missed += 1
                else: incorrect += 1
            elif label == 'no_explicit_count':
                false_positive += int(predicted is not None)
            elif label == 'ambiguous':
                ambiguous_output += int(predicted is not None)
            else: raise ValueError('Unknown review label')
        positive = sum(r['review_status'] == 'explicit_count' for r in rows)
        negative = sum(r['review_status'] == 'no_explicit_count' for r in rows)
        ambiguous = len(rows) - positive - negative
        result[split] = {'rows': len(rows), 'explicit_count_rows': positive, 'correct_counts': matched,
                         'missed_counts': missed, 'incorrect_counts': incorrect, 'no_explicit_count_rows': negative,
                         'false_positives': false_positive, 'ambiguous_rows': ambiguous, 'ambiguous_with_prediction': ambiguous_output,
                         'reviewer': 'single assistant reviewer; not independently adjudicated',
                         'sample': 'simple random sample of remaining described products' if split == 'random-heldout' else 'stratified by text patterns, not representative'}
    path = ROOT / 'reports/generated/pack-review-evaluation.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
