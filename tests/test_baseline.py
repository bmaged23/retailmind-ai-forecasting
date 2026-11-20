from datetime import date, datetime, timedelta
import pytest
from retailmind.data import clean_rows, daily_grid
from retailmind.forecast import backtest, metrics, seasonal_naive


def test_cleaning_and_grid():
    base = {'Invoice': '1', 'StockCode': '123', 'InvoiceDate': datetime(2011, 1, 1), 'Quantity': 2, 'Price': 1}
    rows = [base, base.copy(), {**base, 'Invoice': 'C2', 'Quantity': -2}, {**base, 'Invoice': '3', 'InvoiceDate': datetime(2011, 1, 3)}]
    sales, audit = clean_rows(rows)
    assert audit['included_rows'] == 3
    assert audit['returns_or_cancellations'] == 1
    assert [r[2] for r in daily_grid(sales)] == [4, 0, 2]


def test_forecast_uses_only_history():
    assert seasonal_naive(list(range(14)), 10) == [7, 8, 9, 10, 11, 12, 13, 7, 8, 9]
    start = date(2011, 1, 1)
    panel = {'a': {start + timedelta(days=i): float(i) for i in range(130)}}
    cutoff = start + timedelta(days=95)
    before = backtest(panel, [cutoff])[0]
    panel['a'][cutoff + timedelta(days=1)] = 999
    after = backtest(panel, [cutoff])[0]
    assert before['mae'] != after['mae']
    assert seasonal_naive([panel['a'][cutoff + timedelta(days=i)] for i in range(-89, 1)]) == seasonal_naive(list(map(float, range(6, 96))))


def test_undefined_metrics_and_cohort():
    assert metrics([0, 0], [0, 0], [0] * 90)['wape'] is None
    assert metrics([0, 0], [0, 0], [0] * 90)['mase'] is None
    assert backtest({'x': {date(2011, 1, 1): 1}}, [date(2011, 1, 1)]) == []
    with pytest.raises(ValueError):
        seasonal_naive([1], 28)


def test_reviewed_alphabetic_products_and_service_codes():
    base = {'Invoice': '1', 'StockCode': 'DCGS0058', 'InvoiceDate': datetime(2011, 1, 1), 'Quantity': 2, 'Price': 1}
    sales, audit = clean_rows([base, {**base, 'StockCode': 'PADS'}, {**base, 'StockCode': 'POST'}, {**base, 'StockCode': 'UNREVIEWED'}])
    assert len(sales) == 2
    assert audit['reviewed_alphabetic_product_rows'] == 2
    assert audit['unclassified_or_service_codes'] == 2
