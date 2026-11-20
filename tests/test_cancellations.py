from datetime import datetime, timedelta
from retailmind.cancellations import match_unique


def row(qty, offset=0, customer=123, invoice='1'):
    return {'Invoice': invoice, 'StockCode': '23843', 'Quantity': qty, 'Price': 1.0,
            'InvoiceDate': datetime(2011, 1, 1, 12) + timedelta(minutes=offset),
            'Customer ID': customer, 'Country': 'UK'}


def test_unique_match_and_exclude_missing_or_late():
    pairs, stats = match_unique([row(10), row(-10, 12, invoice='C2'), row(20, customer=None), row(-20, 20, customer=None, invoice='C3'), row(30), row(-30, 1500, invoice='C4')])
    assert len(pairs) == 1 and stats['matched_units'] == 10
    assert pairs[0]['quantity'] == 10
    assert 'customer' not in str(pairs).lower()


def test_ambiguous_pairs_are_not_matched():
    assert not match_unique([row(10), row(10, 1), row(-10, 2, invoice='C2')])[0]
    assert not match_unique([row(10), row(-10, 1, invoice='C2'), row(-10, 2, invoice='C3')])[0]


def test_does_not_match_returns_before_sale():
    assert not match_unique([row(-10, 0, invoice='C2'), row(10, 1)])[0]


def test_different_product_code_casing_does_not_match():
    sale = {**row(10), 'StockCode': 'DCGS001'}
    cancellation = {**row(-10, 1, invoice='C2'), 'StockCode': 'dcgs001'}
    assert not match_unique([sale, cancellation])[0]
