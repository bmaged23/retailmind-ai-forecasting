from datetime import date, datetime
from retailmind.product_features import snapshot, candidate_pack_size, candidate_code_base


def test_snapshot_ignores_future_rows_and_no_identifiers_leak():
    base = {'StockCode': '123A', 'Invoice': '1', 'InvoiceDate': datetime(2011, 4, 20),
            'Quantity': 2, 'Price': 3, 'Description': 'RED GLASS BAG', 'Country': 'United Kingdom', 'Customer ID': 'private'}
    future = {**base, 'Invoice': '2', 'InvoiceDate': datetime(2011, 5, 2), 'Price': 900, 'Description': 'METAL BOX'}
    first = snapshot([base], date(2011, 4, 30), ['123A'])
    assert snapshot([base, future], date(2011, 4, 30), ['123A']) == first
    item = first[0]
    assert item['weighted_mean_price_lookback'] == 3 and item['uk_units_share_lookback'] == 1
    assert item['description_available'] == 1
    assert item['text_glass'] == 1 and item['text_metal'] == 0
    assert item['code_shape'] == 'numeric_with_suffix'
    assert 'private' not in str(item) and 'customer' not in str(item).lower()


def test_empty_history_is_missing_not_zero_price():
    item = snapshot([], date(2011, 4, 30), ['A'])[0]
    assert item['description_available'] == 0
    assert item['text_glass'] is None
    assert item['weighted_mean_price_lookback'] is None
    assert item['uk_units_share_lookback'] is None


def test_pack_and_code_patterns_are_conservative():
    assert candidate_pack_size('SET/6 BEAD COASTERS') == 6
    assert candidate_pack_size('PACK OF 12 RED BAGS') == 12
    assert candidate_pack_size('S/3 TEAR DROP GLASS') == 3
    assert candidate_pack_size('SET 5 MINI MAGNETS') == 5
    assert candidate_pack_size('SET OF THREE VINTAGE WRAPS') == 3
    assert candidate_pack_size('SET OF THREE / SET OF 4') is None
    assert candidate_pack_size('5 HOOK HANGER') is None
    assert candidate_pack_size('SET/6 AND PACK OF 12') is None
    assert candidate_pack_size('BOX 6 X 12 CM') is None
    assert candidate_code_base('35095A') == '35095'
    assert candidate_code_base('DCGS0058') is None


def test_most_recent_description_not_lexically_largest():
    base = {'StockCode':'1', 'Invoice':'10', 'Quantity':1, 'Price':2, 'Country':'United Kingdom'}
    early = {**base, 'InvoiceDate':datetime(2011, 4, 29, 9), 'Description':'Z BOX'}
    late = {**base, 'InvoiceDate':datetime(2011, 4, 29, 17), 'Description':'A GLASS'}
    assert snapshot([late, early], date(2011, 4, 30), ['1'])[0]['description'] == 'A GLASS'


def test_obvious_plural_terms_are_recognized_without_substring_matching():
    from retailmind.product_features import description_tokens
    tokens = description_tokens('BOXES AND CANDLES IN WOODLAND')
    assert 'BOX' in tokens and 'CANDLE' in tokens
    assert 'WOOD' not in tokens


def test_future_only_product_excluded_when_strict_asof_required():
    future = {'StockCode':'A', 'Invoice':'1', 'InvoiceDate':datetime(2011, 5, 1), 'Quantity':1, 'Price':2, 'Description':'BOX'}
    assert snapshot([future], date(2011, 4, 30), ['A'], require_observed=True) == []
    assert snapshot([future], date(2011, 5, 1), ['A'], require_observed=True)[0]['description'] == 'BOX'


def test_cutoff_snapshots_change_only_when_new_observations_arrive():
    base = {'StockCode': 'A', 'Invoice': '11', 'Quantity': 2, 'Price': 3,
            'Description': 'RED BOX', 'Country': 'United Kingdom'}
    earlier = {**base, 'InvoiceDate': datetime(2011, 5, 31, 23, 59)}
    later = {**base, 'Invoice': '12', 'InvoiceDate': datetime(2011, 6, 1),
             'Quantity': 1, 'Price': 9, 'Description': 'BLUE BAG', 'Country': 'France'}
    first = snapshot([earlier, later], date(2011, 5, 31), ['A'], require_observed=True)[0]
    second = snapshot([earlier, later], date(2011, 6, 1), ['A'], require_observed=True)[0]
    assert first['description'] == 'RED BOX' and first['weighted_mean_price_lookback'] == 3
    assert first['invoice_count_lookback'] == 1 and first['uk_units_share_lookback'] == 1
    assert second['description'] == 'BLUE BAG' and second['weighted_mean_price_lookback'] == 5
    assert second['invoice_count_lookback'] == 2 and second['uk_units_share_lookback'] == 2 / 3
