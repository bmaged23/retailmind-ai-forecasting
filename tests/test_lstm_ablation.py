from datetime import date, timedelta
import pytest
from retailmind.lstm_ablation import extra_features, train, predict
from retailmind.transaction_features import TransactionFeatures


def test_feature_variants_use_only_supplied_history_and_metadata():
    h=[0.]*27+[7.]; d=date(2011,6,1)
    assert extra_features('sales',h,d)==[]
    assert len(extra_features('calendar',h,d))==4
    assert extra_features('calendar_frequency',h,d)[-3:] == pytest.approx([1/7,1/28,0])
    assert len(extra_features('calendar_frequency_price',h,d,[1.,2.,.5,0.,0.]))==9


def test_ablation_fit_and_predict_ignore_future_transaction_changes():
    start=date(2010,1,1); cutoff=start+timedelta(days=159)
    sales={'A':{start+timedelta(days=i):float(i%7+1) for i in range(170)}}
    item={'stock_code':'A','date':cutoff-timedelta(days=3),'units':2.,'revenue':6.,'invoice_count':1,'uk_units':2.,'known_country_units':2.}
    future={**item,'date':cutoff+timedelta(days=1),'revenue':100000.}
    a,b=TransactionFeatures([item]),TransactionFeatures([item,future]);h=[sales['A'][cutoff-timedelta(days=i)] for i in range(89,-1,-1)]
    model1,_=train(sales,cutoff,'calendar_frequency_price',metadata=a,epochs=1,device='cpu')
    model2,_=train(sales,cutoff,'calendar_frequency_price',metadata=b,epochs=1,device='cpu')
    pred1=predict(model1,h,cutoff,3,'calendar_frequency_price',metadata_values=a.at('A',cutoff),device='cpu')
    pred2=predict(model2,h,cutoff,3,'calendar_frequency_price',metadata_values=b.at('A',cutoff),device='cpu')
    assert pred1==pytest.approx(pred2)
