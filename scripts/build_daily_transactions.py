"""Build compact daily transaction summaries for the preselected cohorts."""
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
import csv
import json
import math
from retailmind.workbook import iter_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT/'configs/benchmark.json').read_text())
    latest = max(date.fromisoformat(x) for x in cfg['validation_cutoffs'])
    products = set()
    for filename in ['cohort.csv', 'cohort-intermittent.csv']:
        with (ROOT/'reports/generated'/filename).open(newline='') as handle:
            products.update(r['stock_code'] for r in csv.DictReader(handle))
    grouped = defaultdict(lambda: {'units': 0., 'revenue': 0., 'uk_units': 0., 'known_country_units': 0., 'invoices': set()})
    for row in iter_rows(ROOT/'data/raw/online_retail_II.xlsx'):
        code = str(row.get('StockCode') or '').strip()
        when = row.get('InvoiceDate')
        inv = str(row.get('Invoice', row.get('InvoiceNo', '')) or '').strip()
        if code not in products or not inv or inv.upper().startswith('C') or not isinstance(when, (date, datetime)):
            continue
        day = when.date() if isinstance(when, datetime) else when
        if day > latest:
            continue
        try:
            qty = float(row.get('Quantity'))
            price = float(row.get('Price', row.get('UnitPrice')))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(qty) or not math.isfinite(price) or qty <= 0 or price <= 0:
            continue
        record = grouped[(code, day)]
        record['units'] += qty; record['revenue'] += qty * price; record['invoices'].add(inv)
        country = str(row.get('Country') or '').strip().upper()
        if country:
            record['known_country_units'] += qty
        if country == 'UNITED KINGDOM':
            record['uk_units'] += qty
    path = ROOT/'reports/generated/daily-transaction-features.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        fields = ['stock_code','date','units','revenue','invoice_count','uk_units','known_country_units']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (code, day), data in sorted(grouped.items()):
            writer.writerow({'stock_code': code, 'date': day.isoformat(), 'units': data['units'],
                             'revenue': data['revenue'], 'invoice_count': len(data['invoices']),
                             'uk_units': data['uk_units'], 'known_country_units': data['known_country_units']})
    print('Wrote', len(grouped), 'SKU-day transaction summaries to', path)

if __name__ == '__main__': main()
