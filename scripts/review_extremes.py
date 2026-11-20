"""Summarize very large transactions and same-SKU negative lines, not customer data."""
from pathlib import Path
from collections import defaultdict
import json
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]


def main():
    codes = {'23843', '23166', '37410'}
    records = defaultdict(list)
    book = load_workbook(ROOT/'data/raw/online_retail_II.xlsx', read_only=True, data_only=True)
    try:
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            columns = next(rows)
            for values in rows:
                row = dict(zip(columns, values))
                code = str(row.get('StockCode') or '').strip()
                qty = row.get('Quantity')
                if code in codes and isinstance(qty, (int, float)) and abs(qty) >= 5000:
                    records[code].append({'date': str(row.get('InvoiceDate')), 'quantity': qty, 'cancellation_invoice': str(row.get('Invoice') or '').upper().startswith('C')})
    finally:
        book.close()
    output = ROOT/'reports/generated/extreme-review.json'
    output.write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))

if __name__ == '__main__': main()
