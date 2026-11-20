"""Aggregate pre-cutoff availability of candidate transaction metadata."""
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
import csv
import json
import re
from retailmind.workbook import iter_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / 'configs/benchmark.json').read_text())
    cutoff = date.fromisoformat(cfg['cohort_as_of'])
    codes = set()
    for filename in ('cohort.csv', 'cohort-intermittent.csv'):
        with (ROOT / 'reports/generated' / filename).open(newline='', encoding='utf-8') as handle:
            codes.update(row['stock_code'] for row in csv.DictReader(handle))
    descriptions, prices, countries = defaultdict(set), defaultdict(set), Counter()
    latest_descriptions = {}
    valid_rows = 0
    for row in iter_rows(ROOT / 'data/raw/online_retail_II.xlsx'):
        sku = str(row.get('StockCode') or '').strip()
        when = row.get('InvoiceDate')
        if sku not in codes or not isinstance(when, (date, datetime)):
            continue
        day = when.date() if isinstance(when, datetime) else when
        if day > cutoff or str(row.get('Invoice') or '').upper().startswith('C'):
            continue
        try:
            quantity = float(row.get('Quantity'))
            price = float(row.get('Price'))
        except (TypeError, ValueError):
            continue
        if quantity <= 0 or price <= 0:
            continue
        valid_rows += 1
        desc = str(row.get('Description') or '').strip()
        if desc:
            descriptions[sku].add(desc.upper())
            if sku not in latest_descriptions or when > latest_descriptions[sku][0]:
                latest_descriptions[sku] = (when, desc.upper())
        prices[sku].add(price)
        countries[str(row.get('Country') or 'Unknown').strip()] += 1
    words = Counter()
    for _, desc in latest_descriptions.values():
        words.update(set(re.findall(r'[A-Z]{3,}', desc)))
    patterns = Counter('digits' if re.fullmatch(r'\d+', code) else 'digits plus letters' if re.fullmatch(r'\d+[A-Za-z]+', code) else 'other alphanumeric' for code in codes)
    report = {'as_of': cutoff.isoformat(), 'selected_skus': len(codes), 'positive_transaction_rows': valid_rows,
              'common_description_words': words.most_common(30), 'stock_code_patterns': dict(patterns),
              'skus_with_description': len(descriptions), 'skus_with_multiple_descriptions': sum(len(x)>1 for x in descriptions.values()),
              'skus_with_price': len(prices), 'skus_with_multiple_prices': sum(len(x)>1 for x in prices.values()),
              'top_customer_countries_by_transaction_rows': countries.most_common(10),
              'interpretation': 'Country is customer destination, not production origin; descriptions and prices can vary; these counts do not validate inferred categories or product quality.'}
    path = ROOT / 'reports/generated/metadata-profile.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
