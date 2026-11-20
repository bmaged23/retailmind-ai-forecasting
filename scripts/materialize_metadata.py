"""Materialize pre-cutoff feature snapshots; no model fitting or evaluation."""
from datetime import date, datetime
from pathlib import Path
import csv
import json
from retailmind.product_features import snapshot
from retailmind.workbook import iter_rows

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('StockCode', 'Invoice', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'Price', 'UnitPrice', 'Description', 'Country')


def main():
    config = json.loads((ROOT / 'configs/benchmark.json').read_text())
    cutoffs = [date.fromisoformat(item) for item in config['validation_cutoffs']]
    cohorts = {}
    for cohort, filename in [('active', 'cohort.csv'), ('intermittent', 'cohort-intermittent.csv')]:
        with (ROOT / 'reports/generated' / filename).open(newline='', encoding='utf-8') as handle:
            cohorts[cohort] = {row['stock_code'] for row in csv.DictReader(handle)}
    codes = set.union(*cohorts.values())
    events = []
    for row in iter_rows(ROOT / 'data/raw/online_retail_II.xlsx'):
        code = str(row.get('StockCode') or '').strip()
        when = row.get('InvoiceDate')
        if code not in codes or not isinstance(when, (date, datetime)):
            continue
        day = when.date() if isinstance(when, datetime) else when
        if day <= max(cutoffs):
            events.append({key: row[key] for key in FIELDS if key in row})
    out = ROOT / 'reports/generated/feature-snapshots'; out.mkdir(parents=True, exist_ok=True)
    for group, members in cohorts.items():
        for cutoff in cutoffs:
            records = snapshot(events, cutoff, members, require_observed=True)
            if not records:
                raise ValueError(f'No as-of records for {group} at {cutoff}')
            path = out / f'{group}-{cutoff}.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                writer.writeheader(); writer.writerows(records)
            print(f'{group} {cutoff}: {len(records)} of {len(members)} products', flush=True)

if __name__ == '__main__': main()
