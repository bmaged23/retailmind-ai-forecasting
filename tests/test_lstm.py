from datetime import date, timedelta
import pytest

torch = pytest.importorskip('torch')
from retailmind.lstm_model import train, predict


def test_lstm_ignores_future_and_predicts_nonnegative():
    start = date(2010, 1, 1)
    cutoff = start + timedelta(days=160)
    series = {start + timedelta(days=i): float(i % 7 + 1) for i in range(170)}
    panel = {'sku': series}
    history = [series[cutoff - timedelta(days=i)] for i in range(89, -1, -1)]
    model = train(panel, cutoff, epochs=1)
    first = predict(model, history, horizon=3)
    series[cutoff + timedelta(days=1)] = 100000
    second = predict(train(panel, cutoff, epochs=1), history, horizon=3)
    assert first == pytest.approx(second)
    assert len(first) == 3 and min(first) >= 0


def test_lstm_early_stopping_metadata():
    start = date(2010, 1, 1)
    panel = {'sku': {start + timedelta(days=i): 2.0 for i in range(160)}}
    model = train(panel, start + timedelta(days=159), epochs=2, patience=1)
    assert 1 <= model.best_epoch <= model.epochs_run <= 2
    assert model.best_validation_loss >= 0


def test_enriched_lstm_does_not_use_future_transaction_features():
    from retailmind.transaction_features import TransactionFeatures
    start = date(2010, 1, 1); cutoff = start + timedelta(days=159)
    series = {start + timedelta(days=i): float(i % 7 + 1) for i in range(170)}
    entry = {'stock_code':'sku','date':cutoff-timedelta(days=2),'units':2.,'revenue':6.,
             'invoice_count':1,'uk_units':2.,'known_country_units':2.}
    future = {**entry, 'date':cutoff+timedelta(days=1),'revenue':99999.}
    history = [series[cutoff-timedelta(days=i)] for i in range(89,-1,-1)]
    a = TransactionFeatures([entry]); b = TransactionFeatures([entry, future])
    first = predict(train({'sku':series},cutoff,epochs=1,metadata=a),history,horizon=3,cutoff=cutoff,metadata_values=a.at('sku',cutoff))
    second = predict(train({'sku':series},cutoff,epochs=1,metadata=b),history,horizon=3,cutoff=cutoff,metadata_values=b.at('sku',cutoff))
    assert first == pytest.approx(second)
