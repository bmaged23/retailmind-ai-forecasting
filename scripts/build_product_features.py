"""Generate a reviewable pre-cutoff metadata feature catalogue (no training)."""
from pathlib import Path
import argparse
from datetime import date
import csv
import json
from retailmind.product_features import snapshot
from retailmind.workbook import iter_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Profile all panel codes; descriptive only')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/benchmark.json').read_text())
    codes = set()
    if args.all:
        with (ROOT / 'data/processed/daily_sales.csv').open(newline='', encoding='utf-8') as handle:
            codes.update(row['stock_code'] for row in csv.DictReader(handle))
    else:
        for name in ('cohort.csv', 'cohort-intermittent.csv'):
            with (ROOT / 'reports/generated' / name).open(newline='', encoding='utf-8') as handle:
                codes.update(row['stock_code'] for row in csv.DictReader(handle))
    records = snapshot(iter_rows(ROOT / 'data/raw/online_retail_II.xlsx'), date.fromisoformat(config['cohort_as_of']), codes)
    output = ROOT / ('reports/generated/product-features-all.csv' if args.all else 'reports/generated/product-features.csv')
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    print(f'Wrote {len(records)} pre-cutoff product feature rows to {output}')


if __name__ == '__main__': main()
