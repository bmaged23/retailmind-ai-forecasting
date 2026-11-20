"""Shared frozen-cohort, pre-test context, GPU and forecast-output utilities."""
from datetime import date, timedelta
from pathlib import Path
import csv
import json
import numpy as np
import torch
from retailmind.forecast import load_panel
from retailmind.gpu_monitor import GPUMonitor

ROOT=Path(__file__).resolve().parents[2]
TEST_DIR=ROOT/'artifacts/foundation_models/test_2011-10-05_to_2011-11-01'
VALIDATION_DIR=ROOT/'artifacts/validation_comparison_2011-07-01_to_2011-07-28'
RUN_VALIDATION=__import__('os').environ.get('RETAILMIND_RUN_SPLIT')=='validation'
RUN_DIR=VALIDATION_DIR if RUN_VALIDATION else TEST_DIR
BASE_DIR=ROOT/'artifacts/train-validation-test-2011-11-01'
if RUN_VALIDATION:
    _comparison=json.loads((ROOT/'configs/validation_comparison.json').read_text())
    ORIGIN=date.fromisoformat(_comparison['selection']['forecast_origin'])
    HORIZON=int(_comparison['horizon_days'])
    CONTEXT=int(_comparison['context_days'])
else:
    ORIGIN=date(2011,10,4)
    HORIZON=28
    CONTEXT=90

def frozen_cohort():
    if RUN_VALIDATION:
        path=VALIDATION_DIR/'cohort.json'
        if not path.exists():
            raise FileNotFoundError('Run prepare_validation_cohort.py first')
        import json
        return json.loads(path.read_text())['codes']
    path=BASE_DIR/'test-forecasts.csv'
    with path.open(newline='') as f:
        return sorted({r['stock_code'] for r in csv.DictReader(f) if r['model']=='lightgbm'})

def load_panel_through(path,end):
    """Read only daily-sale rows through ``end``; later targets are not parsed."""
    panel={}
    with Path(path).open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            day=date.fromisoformat(row['date'])
            if day>end:continue
            panel.setdefault(row['stock_code'],{})[day]=float(row['quantity'])
    return panel

def load_contexts():
    panel=(load_panel_through(ROOT/'data/processed/daily_sales.csv',ORIGIN)
           if RUN_VALIDATION else load_panel(ROOT/'data/processed/daily_sales.csv'))
    codes=frozen_cohort();contexts=[]
    start=ORIGIN-timedelta(days=CONTEXT-1)
    for code in codes:
        series=panel.get(code,{})
        observed=[d for d,q in series.items() if d<=ORIGIN and q>0]
        if not observed:raise ValueError(f'No pre-test sale history for frozen product {code}')
        values=[series.get(start+timedelta(days=i),0.) for i in range(CONTEXT)]
        contexts.append(np.asarray(values,dtype=np.float32))
    return panel,codes,contexts

def training_series(panel):
    """Return sequences containing no target after validation end."""
    if RUN_VALIDATION:
        fit_end=date.fromisoformat(_comparison['tuning']['fit_targets_through'])
        valid_end=date.fromisoformat(_comparison['tuning']['checkpoint_validation_end'])
    else:
        fit_end=date(2011,8,31);valid_end=date(2011,9,28)
    fit=[];validation=[]
    for series in panel.values():
        train=dense_series(series,fit_end)
        valid=dense_series(series,valid_end)
        if train is not None and len(train)>=CONTEXT+HORIZON:fit.append(train)
        if valid is not None and len(valid)>=CONTEXT+HORIZON:validation.append(valid)
    return fit,validation

def dense_series(series,end):
    values={d:q for d,q in series.items() if d<=end}
    if not values:return None
    first=min(values)
    return np.asarray([values.get(first+timedelta(days=i),0.)
                       for i in range((end-first).days+1)],dtype=np.float32)

def start_gpu_monitor(cap_mib=1536,threshold_mib=1900):
    monitor=GPUMonitor(max_used_mib=threshold_mib).start()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to silently substitute CPU inference')
    total=torch.cuda.get_device_properties(0).total_memory
    fraction=min(1.,cap_mib*1024**2/total)
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    torch.backends.cudnn.benchmark=False
    torch.cuda.reset_peak_memory_stats()
    monitor.check()
    return monitor

def write_forecasts(model_name,codes,predictions,elapsed_seconds):
    if RUN_VALIDATION:
        directory=VALIDATION_DIR/'forecasts'
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'{model_name}.csv'
        per_series=elapsed_seconds/len(codes)
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['model','stock_code','target_date','horizon','prediction','inference_seconds_per_series'])
            w.writeheader()
            for code,pred in zip(codes,predictions):
                values=np.asarray(pred,dtype=float).reshape(-1)
                if len(values)!=HORIZON:raise ValueError(f'{model_name} returned {len(values)} horizons')
                for h,value in enumerate(values,1):
                    w.writerow({'model':model_name,'stock_code':code,
                      'target_date':(ORIGIN+timedelta(days=h)).isoformat(),
                      'horizon':h,'prediction':max(0.,float(value)),
                      'inference_seconds_per_series':per_series})
        return path
    directory=TEST_DIR/'forecasts'/model_name
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{model_name}.csv'
    per_series=elapsed_seconds/len(codes)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','stock_code','target_date','horizon','prediction','inference_seconds_per_series'])
        w.writeheader()
        for code,pred in zip(codes,predictions):
            values=np.asarray(pred,dtype=float).reshape(-1)
            if len(values)!=HORIZON:raise ValueError(f'{model_name} returned {len(values)} horizons')
            for h,value in enumerate(values,1):
                w.writerow({'model':model_name,'stock_code':code,
                  'target_date':(ORIGIN+timedelta(days=h)).isoformat(),
                  'horizon':h,'prediction':max(0.,float(value)),
                  'inference_seconds_per_series':per_series})
    return path

def save_run_metadata(model_name,**values):
    directory=RUN_DIR/'metadata';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{model_name}.json'
    import json
    path.write_text(json.dumps(values,indent=2)+'\n',encoding='utf-8')
    return path
