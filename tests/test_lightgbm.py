from datetime import date, timedelta
from retailmind.lightgbm_model import predict, train


def test_training_ignores_future_and_recursive_prediction():
    start = date(2010, 1, 1)
    cutoff = start + timedelta(days=70)
    panel = {'sku': {start + timedelta(days=i): float(i % 7) for i in range(90)}}
    first = train(panel, cutoff)
    history = [panel['sku'][cutoff - timedelta(days=i)] for i in range(27, -1, -1)]
    expected = predict(first, history, cutoff, 28)
    for day in panel['sku']:
        if day > cutoff:
            panel['sku'][day] = 99999
    actual = predict(train(panel, cutoff), history, cutoff, 28)
    assert actual == expected
    assert len(actual) == 28 and min(actual) >= 0
