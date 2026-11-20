"""Conservative, exploratory matching of sales to same-day cancellations.

Matching is a sensitivity analysis, not verified linkage or a production target.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math


def candidate(row):
    """Return an in-memory match key and event; omit identifiers from outputs."""
    code = str(row.get('StockCode') or '').strip()
    normalized = code.upper()
    invoice = str(row.get('Invoice', row.get('InvoiceNo', '')) or '').strip()
    customer = row.get('Customer ID', row.get('CustomerID'))
    when = row.get('InvoiceDate')
    try:
        qty = float(row.get('Quantity'))
        price = float(row.get('Price', row.get('UnitPrice')))
    except (ValueError, TypeError):
        return None
    if (not code or not invoice or customer is None or str(customer).strip() == ''
            or not isinstance(when, datetime) or not math.isfinite(qty)
            or not math.isfinite(price) or not price > 0 or qty == 0):
        return None
    if not (code[0].isdigit() or normalized.startswith('DCGS') or normalized == 'PADS'):
        return None
    is_cancel = invoice.upper().startswith('C') and qty < 0
    if not is_cancel and (invoice.upper().startswith('C') or qty < 0):
        return None
    country = str(row.get('Country') or '').strip().upper()
    key = (code, str(customer).strip(), country, abs(qty), price)
    return key, (when, is_cancel)


def match_unique(rows, max_delay=timedelta(days=1)):
    """Match only mutually unique sale/cancel candidates within max_delay.

    The cancellation must follow the sale. Ambiguous repeated sales or returns
    are not matched. No match is asserted across missing customer IDs.
    """
    if max_delay <= timedelta(0):
        raise ValueError('max_delay must be positive')
    groups = defaultdict(lambda: ([], []))
    stats = Counter()
    for row in rows:
        stats['rows_read'] += 1
        event = candidate(row)
        if event is None:
            continue
        key, (when, is_cancel) = event
        groups[key][int(is_cancel)].append(when)
        stats['candidate_cancellations' if is_cancel else 'candidate_sales'] += 1
    matched = []
    for key, (sales, cancellations) in groups.items():
        if not sales or not cancellations:
            continue
        # Sorted two-pointer windows avoid comparing every sale to every return.
        sales.sort(); cancellations.sort()
        sale_degrees = [0] * len(sales)
        cancel_degrees = [0] * len(cancellations)
        sale_only = [None] * len(sales)
        left = right = 0
        for j, cancel in enumerate(cancellations):
            while left < len(sales) and sales[left] < cancel - max_delay:
                left += 1
            right = max(right, left)
            while right < len(sales) and sales[right] <= cancel:
                right += 1
            cancel_degrees[j] = right - left
            # Increment each overlapping range in O(1); resolve degrees below.
            if left < right:
                sale_degrees[left] += 1
                if right < len(sales):
                    sale_degrees[right] -= 1
                if right - left == 1:
                    sale_only[left] = j
        running = 0
        for i, delta in enumerate(sale_degrees):
            running += delta
            j = sale_only[i]
            if running == 1 and j is not None and cancel_degrees[j] == 1:
                matched.append({'stock_code': key[0], 'sale_date': sales[i].date().isoformat(),
                                'cancellation_date': cancellations[j].date().isoformat(), 'quantity': key[3]})
    stats['matched_pairs'] = len(matched)
    stats['matched_units'] = sum(item['quantity'] for item in matched)
    return matched, stats
