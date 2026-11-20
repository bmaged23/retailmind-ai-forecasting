"""Generate aggregate cancellation sensitivity figures from the workbook."""
from collections import defaultdict
from pathlib import Path
import csv
import json
from retailmind.cancellations import match_unique
from retailmind.workbook import iter_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    pairs, counts = match_unique(iter_rows(ROOT / 'data/raw/online_retail_II.xlsx'))
    adjustments = defaultdict(float)
    for pair in pairs:
        adjustments[(pair['stock_code'], pair['sale_date'])] += pair['quantity']
    gross_total = adjusted_total = gross_peak = adjusted_peak = 0.0
    affected_days = negative_days = 0
    with (ROOT / 'data/processed/daily_sales.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            gross = float(row['quantity'])
            adjustment = adjustments.get((row['stock_code'], row['date']), 0)
            adjusted = gross - adjustment
            if adjustment:
                affected_days += 1
            if adjusted < 0:
                negative_days += 1
            gross_total += gross
            adjusted_total += adjusted
            gross_peak = max(gross_peak, gross)
            adjusted_peak = max(adjusted_peak, adjusted)
    if negative_days:
        raise ValueError(f'{negative_days} adjusted product-days are negative; investigate classification mismatch')
    report = {'method': 'same SKU, non-missing customer, country, absolute quantity and price; sale precedes cancellation by at most 24 hours; mutually unique candidate pair',
              'limitations': ['heuristic, not verified transaction linkage', 'later cancellations can introduce look-ahead bias if applied to historical forecasts', 'not a model-ready adjusted target'],
              **dict(counts), 'matched_same_day_pairs': sum(p['sale_date'] == p['cancellation_date'] for p in pairs),
              'affected_product_days': affected_days, 'gross_units': gross_total, 'adjusted_units': adjusted_total,
              'matched_units_share_pct': counts['matched_units'] / gross_total * 100 if gross_total else None,
              'gross_max_product_day': gross_peak, 'adjusted_max_product_day': adjusted_peak}
    path = ROOT / 'reports/generated/cancellation-sensitivity.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
