"""Convert Online Retail II transaction rows to a daily gross-sales panel."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
import csv
import json
import math


def clean_rows(rows):
    sales = defaultdict(float)
    counts = Counter()
    for row in rows:
        counts['rows_read'] += 1
        invoice = str(row.get('Invoice', row.get('InvoiceNo', '')) or '').strip()
        code = str(row.get('StockCode', '') or '').strip()
        when = row.get('InvoiceDate')
        try:
            quantity = float(row.get('Quantity'))
            price = float(row.get('Price', row.get('UnitPrice')))
        except (TypeError, ValueError):
            counts['invalid_numeric'] += 1
            continue
        if not isinstance(when, (datetime, date)) or not invoice or not code or not math.isfinite(quantity) or not math.isfinite(price):
            counts['invalid_fields'] += 1
            continue
        # Identical transaction lines can be legitimate; do not deduplicate them.
        if invoice.upper().startswith('C') or quantity < 0:
            counts['returns_or_cancellations'] += 1
            continue
        if quantity == 0 or price <= 0:
            counts['nonpositive_quantity_or_price'] += 1
            continue
        # Only reviewed alphabetic codes are admitted; other codes stay in a
        # review bucket rather than being asserted to be non-products.
        normalized = code.upper()
        if not code[0].isdigit() and not (normalized.startswith('DCGS') or normalized == 'PADS'):
            counts['unclassified_or_service_codes'] += 1
            continue
        if not code[0].isdigit():
            counts['reviewed_alphabetic_product_rows'] += 1
        day = when.date() if isinstance(when, datetime) else when
        sales[(code, day)] += quantity
        counts['included_rows'] += 1
    return sales, counts


def daily_grid(sales):
    """Fill dates only between each SKU's first and last recorded sale."""
    by_product = defaultdict(dict)
    for (code, day), quantity in sales.items():
        by_product[code][day] = quantity
    for code in sorted(by_product):
        series = by_product[code]
        day, end = min(series), max(series)
        while day <= end:
            yield code, day, series.get(day, 0.0)
            day += timedelta(days=1)


def preprocess(workbook, output):
    from .workbook import iter_rows
    sales, audit = clean_rows(iter_rows(workbook))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'daily_sales.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['stock_code', 'date', 'quantity'])
        writer.writerows((code, day.isoformat(), quantity) for code, day, quantity in daily_grid(sales))
    (output / 'audit.json').write_text(json.dumps(dict(audit), indent=2) + '\n')
    return audit
