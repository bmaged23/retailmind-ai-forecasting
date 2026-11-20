"""Write a descriptive audit and a cohort selected using only past observations."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def select_cohort(panel, cfg):
    asof = pd.Timestamp(cfg['cohort_as_of'])
    start = asof - pd.Timedelta(days=cfg['cohort_lookback_days'] - 1)
    # Select using positive observations at or before the selection date only.
    # Panel zero rows can extend to the last sale in the future, so their presence
    # must never decide whether a product was eligible at the cutoff.
    history = panel[(panel.date <= asof) & (panel.quantity > 0)]
    first = history.groupby('stock_code').date.min().rename('first_sale')
    past = history[history.date >= start]
    candidates = past.groupby('stock_code').agg(sale_days=('quantity', 'size'), units=('quantity', 'sum'), last_sale=('date', 'max')).join(first).reset_index()
    recent_start = asof - pd.Timedelta(days=27)
    eligible = candidates[(candidates.first_sale <= start) & (candidates.sale_days >= cfg['minimum_sale_days']) & (candidates.last_sale >= recent_start)]
    cohort = eligible.sort_values(['units', 'stock_code'], ascending=[False, True]).head(cfg['maximum_products']).copy()
    return cohort, len(candidates), len(eligible)


def select_intermittent(panel, cfg):
    asof = pd.Timestamp(cfg['cohort_as_of'])
    start = asof - pd.Timedelta(days=cfg['cohort_lookback_days'] - 1)
    policy = cfg['intermittent_cohort']
    history = panel[(panel.date <= asof) & (panel.quantity > 0)]
    first = history.groupby('stock_code').date.min().rename('first_sale')
    recent = history[history.date >= start]
    candidates = recent.groupby('stock_code').agg(sale_days=('quantity', 'size'), units=('quantity', 'sum'), last_sale=('date', 'max')).join(first).reset_index()
    eligible = candidates[(candidates.first_sale <= start) & candidates.sale_days.between(policy['minimum_sale_days'], policy['maximum_sale_days']) & (candidates.last_sale >= asof - pd.Timedelta(days=policy['recent_sale_days'] - 1))]
    return eligible.sort_values(['units', 'stock_code'], ascending=[False, True]).head(policy['maximum_products']).copy(), len(eligible)


def main():
    cfg = json.loads((ROOT / 'configs/benchmark.json').read_text())
    panel = pd.read_csv(ROOT / 'data/processed/daily_sales.csv', dtype={'stock_code': str}, parse_dates=['date'])
    cohort, candidate_count, eligible_count = select_cohort(panel, cfg)
    intermittent, intermittent_eligible = select_intermittent(panel, cfg)
    if set(cohort.stock_code) & set(intermittent.stock_code):
        raise ValueError('Cohorts must be disjoint')
    out = ROOT / 'reports/generated'; out.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out / 'cohort.csv', index=False)
    intermittent.to_csv(out / 'cohort-intermittent.csv', index=False)
    observed = set(panel.date.unique())
    calendar = pd.date_range(panel.date.min(), panel.date.max(), freq='D')
    missing = [d.strftime('%Y-%m-%d') for d in calendar if d not in observed]
    positive = panel.loc[panel.quantity > 0, 'quantity']
    availability = []
    lookup = panel.set_index(['stock_code', 'date']).index
    for phase in ['validation', 'calibration', 'test']:
        for cutoff_str in cfg[phase + '_cutoffs']:
            cutoff = pd.Timestamp(cutoff_str)
            targets = pd.date_range(cutoff + pd.Timedelta(days=1), periods=cfg['horizon_days'])
            complete = sum(all((sku, d) in lookup for d in targets) for sku in cohort.stock_code)
            availability.append({'phase': phase, 'cutoff': cutoff_str, 'complete_target_skus': int(complete), 'missing_target_skus': int(len(cohort)-complete)})
    report = {'panel_start': str(panel.date.min().date()), 'panel_end': str(panel.date.max().date()), 'missing_global_dates': missing,
              'positive_product_day_quantiles': {str(k): float(v) for k,v in positive.quantile([.5,.9,.99,.999,1]).items()},
              'cohort_as_of': cfg['cohort_as_of'], 'candidates_with_any_history': candidate_count, 'eligible_before_limit': eligible_count, 'selected_products': len(cohort), 'intermittent_eligible': intermittent_eligible, 'intermittent_selected': len(intermittent), 'target_availability': availability}
    (out / 'data-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
