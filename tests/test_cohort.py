import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('validate_data', Path(__file__).resolve().parents[1] / 'scripts/validate_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cohort_is_invariant_to_future_observations():
    cfg = {'cohort_as_of': '2011-04-30', 'cohort_lookback_days': 90, 'minimum_sale_days': 14, 'maximum_products': 2}
    dates = pd.date_range('2011-01-01', '2011-04-30')
    panel = pd.DataFrame({'stock_code': 'A', 'date': dates, 'quantity': [1.0] * len(dates)})
    expected = module.select_cohort(panel, cfg)[0].stock_code.tolist()
    future = pd.DataFrame({'stock_code': ['B', 'A'], 'date': pd.to_datetime(['2011-05-01', '2011-05-01']), 'quantity': [100000, 0]})
    assert module.select_cohort(pd.concat([panel, future]), cfg)[0].stock_code.tolist() == expected == ['A']


def test_intermittent_cohort_uses_only_pre_cutoff_sales():
    cfg = {'cohort_as_of': '2011-04-30', 'cohort_lookback_days': 90,
           'intermittent_cohort': {'minimum_sale_days': 3, 'maximum_sale_days': 13, 'maximum_products': 100, 'recent_sale_days': 28}}
    days = pd.to_datetime(['2011-01-01', '2011-03-01', '2011-03-20', '2011-04-10', '2011-04-20'])
    panel = pd.DataFrame({'stock_code': 'X', 'date': days, 'quantity': 1.0})
    expected, _ = module.select_intermittent(panel, cfg)
    future = pd.DataFrame({'stock_code': ['X'], 'date': pd.to_datetime(['2011-05-01']), 'quantity': [100000]})
    actual, _ = module.select_intermittent(pd.concat([panel, future]), cfg)
    assert expected.stock_code.tolist() == actual.stock_code.tolist() == ['X']
