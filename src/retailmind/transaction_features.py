"""Cutoff-safe historical transaction summaries, with missingness indicators."""
from bisect import bisect_left, bisect_right
from datetime import timedelta
import csv
import math


class TransactionFeatures:
    def __init__(self, rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row['stock_code'], []).append(row)
        self.series = {code: sorted(items, key=lambda x: x['date']) for code, items in grouped.items()}
        self.dates = {code: [r['date'] for r in items] for code, items in self.series.items()}
        self.prefix = {}
        for code, items in self.series.items():
            totals = [[0.0] for _ in range(5)]
            for row in items:
                for values, key in zip(totals, ('units', 'revenue', 'invoice_count', 'uk_units', 'known_country_units')):
                    values.append(values[-1] + row[key])
            self.prefix[code] = totals

    def at(self, code, cutoff):
        """Return log price, log invoice-days, UK share and price/geo missing flags.

        Invoice-days are distinct invoices per product and date, summed over 90
        days; the same invoice spanning dates may be counted more than once.
        """
        items = self.series.get(code, [])
        days = self.dates.get(code, [])
        lo = bisect_left(days, cutoff - timedelta(days=89))
        hi = bisect_right(days, cutoff)
        totals = self.prefix.get(code)
        values = [s[hi] - s[lo] for s in totals] if totals else [0.0] * 5
        units, revenue, invoice_days, uk, known = values
        if not units:
            return [0.0, 0.0, 0.0, 1.0, 1.0]
        price = revenue / units
        return [math.log1p(price), math.log1p(invoice_days), uk / known if known else 0.0,
                0.0, float(known == 0)]


def load(path):
    from datetime import date
    rows = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            rows.append({'stock_code': row['stock_code'], 'date': date.fromisoformat(row['date']),
                         'units': float(row['units']), 'revenue': float(row['revenue']),
                         'invoice_count': int(row['invoice_count']), 'uk_units': float(row['uk_units']),
                         'known_country_units': float(row['known_country_units'])})
    return TransactionFeatures(rows)
