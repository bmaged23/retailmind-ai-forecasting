from datetime import date, timedelta
from retailmind.benchmark import evaluate_fold, historical_panel


def test_history_not_affected_by_future_data():
    start = date(2011, 1, 1); cutoff = start + timedelta(days=100)
    panel = {'sku': {start + timedelta(days=i): float(i % 7) for i in range(101)}}
    before = historical_panel(panel, ['sku'], cutoff)
    panel['sku'][cutoff + timedelta(days=10)] = 10000
    assert historical_panel(panel, ['sku'], cutoff) == before


def test_common_mask_and_forecasts():
    start = date(2011, 1, 1); cutoff = start + timedelta(days=100)
    series = {start + timedelta(days=i): float(i % 7 + 1) for i in range(130)}
    rows, forecasts, missing, metadata = evaluate_fold({'a': series, 'b': {**series, cutoff + timedelta(days=1): 1}}, ['a', 'b', 'absent'], cutoff, 90, 28)
    assert {r['model'] for r in rows} == {'seasonal_naive', 'lightgbm'}
    assert len(rows) == 4 and len(forecasts) == 112
    assert missing == [{'stock_code': 'absent', 'reason': 'insufficient_history'}]
    assert metadata['scored_products'] == 2
    assert all(r['prediction'] >= 0 for r in forecasts)
