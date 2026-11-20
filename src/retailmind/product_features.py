"""Auditable product metadata features available as of a specified date.

Keywords are literal text indicators, not verified categories or quality labels.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import math
import re

# Exploratory, intentionally small and overlapping; not a complete taxonomy.
TEXT_TERMS = ('BAG', 'BOX', 'CANDLE', 'CHRISTMAS', 'GLASS', 'METAL', 'PAPER',
              'WOOD', 'CERAMIC', 'FABRIC', 'SET', 'PACK', 'HEART', 'RED', 'BLUE', 'WHITE', 'PINK')


# Only obvious plural forms; ambiguous phrases still need human review.
PLURALS = {'BAGS': 'BAG', 'BOXES': 'BOX', 'CANDLES': 'CANDLE', 'HEARTS': 'HEART',
           'SETS': 'SET', 'PACKS': 'PACK', 'PAPERS': 'PAPER'}


def description_tokens(description):
    raw = set(re.findall(r'[A-Z]+', description.upper()))
    return raw | {PLURALS[token] for token in raw if token in PLURALS}


def code_shape(code):
    if re.fullmatch(r'\d+', code):
        return 'numeric'
    if re.fullmatch(r'\d+[A-Za-z]+', code):
        return 'numeric_with_suffix'
    return 'other'


def candidate_pack_size(description):
    """Extract explicit SET/PACK counts and S/n, abstaining on conflicts."""
    description = description.upper()
    matches = re.findall(r'\b(?:SET|PACK)(?:\s*/\s*|\s+OF\s+|\s+)(\d{1,3})\b', description)
    matches += re.findall(r'\bS\s*/\s*(\d{1,3})\b', description)
    sizes = {int(value) for value in matches if 1 <= int(value) <= 100}
    words = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
             'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'TEN': 10}
    for word in re.findall(r'\b(?:SET|PACK)\s+OF\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b', description):
        sizes.add(words[word])
    return next(iter(sizes)) if len(sizes) == 1 else None



def candidate_code_base(code):
    match = re.fullmatch(r'(\d+)[A-Za-z]+', code)
    return match.group(1) if match else None


def snapshot(rows, cutoff, codes, *, lookback_days=90, require_observed=False):
    """Summarize only positive sales recorded by cutoff; no customer IDs retained."""
    if not isinstance(cutoff, date) or isinstance(cutoff, datetime) or lookback_days < 1:
        raise ValueError('cutoff must be a date and lookback_days positive')
    codes = set(codes)
    start = cutoff - timedelta(days=lookback_days - 1)
    latest = {}; units = Counter(); weighted = Counter(); invoices = defaultdict(set); countries = defaultdict(Counter); observed = set()
    for row in rows:
        code = str(row.get('StockCode') or '').strip()
        when = row.get('InvoiceDate')
        invoice = str(row.get('Invoice', row.get('InvoiceNo', '')) or '').strip()
        if code not in codes or not invoice or invoice.upper().startswith('C') or not isinstance(when, (date, datetime)):
            continue
        day = when.date() if isinstance(when, datetime) else when
        if day > cutoff:
            continue
        try:
            qty = float(row.get('Quantity'))
            price = float(row.get('Price', row.get('UnitPrice')))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(qty) or not math.isfinite(price) or qty <= 0 or price <= 0:
            continue
        observed.add(code)
        description = str(row.get('Description') or '').strip().upper()
        if description:
            # Latest timestamp wins; text breaks exact-timestamp ties deterministically.
            timestamp = when if isinstance(when, datetime) else datetime.combine(day, datetime.min.time())
            candidate = (timestamp, description)
            if code not in latest or candidate > latest[code]:
                latest[code] = candidate
        if day < start:
            continue
        units[code] += qty
        weighted[code] += qty * price
        invoices[code].add(invoice)
        countries[code][str(row.get('Country') or '').strip().upper()] += qty
    result = []
    for code in sorted(codes):
        if require_observed and code not in observed:
            continue
        desc = latest.get(code, (None, ''))[1]
        tokens = description_tokens(desc)
        country_total = sum(countries[code].values())
        item = {'stock_code': code, 'as_of': cutoff.isoformat(), 'description': desc, 'description_available': int(bool(desc)),
                'code_shape': code_shape(code), 'candidate_code_base': candidate_code_base(code),
                'candidate_pack_size': candidate_pack_size(desc), 'sales_units_lookback': units[code],
                'weighted_mean_price_lookback': weighted[code] / units[code] if units[code] else None,
                'invoice_count_lookback': len(invoices[code]),
                'uk_units_share_lookback': countries[code]['UNITED KINGDOM'] / country_total if country_total else None}
        item.update({f'text_{term.lower()}': int(term in tokens) if desc else None for term in TEXT_TERMS})
        result.append(item)
    return result
