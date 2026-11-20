"""Inspect product-code exclusions and unusual transaction quantities."""
from collections import Counter
from pathlib import Path
import heapq
import json
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]


def main():
    book = load_workbook(ROOT / 'data/raw/online_retail_II.xlsx', read_only=True, data_only=True)
    excluded = Counter(); descriptions = {}; extremes = []
    try:
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows)
            for values in rows:
                row = dict(zip(headers, values))
                code = str(row.get('StockCode') or '').strip()
                invoice = str(row.get('Invoice') or '').strip()
                qty, price = row.get('Quantity'), row.get('Price')
                if not code or not isinstance(qty, (int, float)) or not isinstance(price, (int, float)) or invoice.upper().startswith('C') or qty <= 0 or price <= 0:
                    continue
                if not code[0].isdigit():
                    excluded[code] += 1
                    descriptions.setdefault(code, str(row.get('Description') or '')[:100])
                else:
                    item = (float(qty), str(row.get('InvoiceDate')), code)
                    if len(extremes) < 15: heapq.heappush(extremes, item)
                    elif item > extremes[0]: heapq.heapreplace(extremes, item)
    finally: book.close()
    report = {'top_excluded_codes': [{'stock_code': code, 'rows': n, 'example_description': descriptions[code]} for code,n in excluded.most_common(20)], 'largest_included_transactions': [{'quantity': q, 'timestamp': d, 'stock_code': c} for q,d,c in sorted(extremes, reverse=True)]}
    path = ROOT / 'reports/generated/raw-review.json'; path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
