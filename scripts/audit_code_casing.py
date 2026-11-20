"""Inspect code-case variants without modifying or combining sales series."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/generated'
    daily = pd.read_csv(ROOT / 'data/processed/daily_sales.csv', dtype={'stock_code': str}, parse_dates=['date'])
    positive = daily[daily.quantity > 0].copy()
    positive['normalized'] = positive.stock_code.str.upper()
    variants = positive.groupby('normalized').stock_code.nunique()
    possible = set(variants[variants > 1].index)
    subset = positive[positive.normalized.isin(possible)]
    by_code = subset.groupby(['normalized', 'stock_code']).agg(first_sale=('date','min'), last_sale=('date','max'),
                                                               transaction_days=('date','size'), units=('quantity','sum')).reset_index()
    paired_days = subset.groupby(['normalized', 'date']).stock_code.nunique()
    overlapping = set(paired_days[paired_days > 1].index.get_level_values('normalized'))
    by_code['first_sale'] = by_code.first_sale.dt.strftime('%Y-%m-%d')
    by_code['last_sale'] = by_code.last_sale.dt.strftime('%Y-%m-%d')
    by_code.to_csv(folder / 'code-case-variants.csv', index=False)
    metadata = pd.read_csv(folder / 'product-features-all.csv', dtype={'stock_code': str}).fillna({'description': ''})
    metadata['normalized'] = metadata.stock_code.str.upper()
    counts = metadata[metadata.normalized.isin(possible) & (metadata.description != '')].groupby('normalized').description.nunique()
    summary = {'normalized_groups': len(possible), 'distinct_original_codes': len(by_code),
               'groups_with_same_day_sales_for_multiple_variants': len(overlapping),
               'groups_with_descriptions_at_cutoff': len(counts),
               'groups_with_multiple_descriptions_at_cutoff': int((counts > 1).sum()),
               'note': 'Same-day sales or matching descriptions do not prove identical products; no codes were merged. Date ranges use the full dataset and are descriptive only.'}
    (folder / 'code-case-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(by_code.head(20).to_string(index=False))

if __name__ == '__main__': main()
