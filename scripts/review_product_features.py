"""Reproducible feature-value profiling and random samples for human review."""
from pathlib import Path
import argparse
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'reports/generated'
    prefix = 'product-features-all.csv' if args.all else 'product-features.csv'
    label = '-all' if args.all else ''
    frame = pd.read_csv(folder / prefix, dtype={'stock_code': str, 'candidate_code_base': str})
    profile = {}
    for name in frame.columns:
        series = frame[name]
        valid = series.dropna()
        info = {'missing': int(series.isna().sum()), 'unique_non_null': int(valid.nunique())}
        if valid.nunique() <= 20:
            counts = valid.astype(str).value_counts().sort_index()
            info['value_counts'] = {str(k): int(v) for k, v in counts.items()}
        elif pd.api.types.is_numeric_dtype(series):
            info['quantiles'] = {str(k): float(v) for k, v in valid.quantile([0, .25, .5, .75, 1]).items()}
        profile[name] = info
    (folder / f'product-feature-profile{label}.json').write_text(json.dumps(profile, indent=2) + '\n')
    sample = frame.sample(n=min(50 if args.all else 20, len(frame)), random_state=42).copy()
    tags = [c for c in frame if c.startswith('text_')]
    sample['literal_tags'] = sample[tags].apply(lambda r: ', '.join(k[5:] for k,v in r.items() if v == 1), axis=1)
    columns = ['stock_code', 'description', 'code_shape', 'candidate_code_base', 'candidate_pack_size', 'weighted_mean_price_lookback', 'uk_units_share_lookback', 'literal_tags']
    sample[columns].to_csv(folder / f'product-feature-sample{label}.csv', index=False)
    print(sample[columns].to_string(index=False))
    print('\nFeature profile written to', folder / f'product-feature-profile{label}.json')

if __name__ == '__main__': main()
