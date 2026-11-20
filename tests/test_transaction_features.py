from datetime import date
import math
import pytest
from retailmind.transaction_features import TransactionFeatures


def test_only_previous_90_days_and_future_excluded():
    rows = [{'stock_code':'A','date':date(2011,1,1),'units':2.,'revenue':6.,'invoice_count':1,'uk_units':2.,'known_country_units':2.},
            {'stock_code':'A','date':date(2011,4,1),'units':1.,'revenue':100.,'invoice_count':1,'uk_units':0.,'known_country_units':1.}]
    store = TransactionFeatures(rows)
    assert store.at('A',date(2011,1,31)) == pytest.approx([math.log1p(3),math.log1p(1),1.,0.,0.])
    assert store.at('A',date(2011,6,30)) == [0.,0.,0.,1.,1.]
    rows[1]['revenue'] = 999999
    assert TransactionFeatures(rows).at('A',date(2011,1,31)) == store.at('A',date(2011,1,31))
