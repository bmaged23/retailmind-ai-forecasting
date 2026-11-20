"""Fit selected global baselines on all qualifying historical products/windows."""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import sys
import numpy as np
import torch
import lightgbm as lgb
from retailmind.forecast import load_panel
from retailmind.gpu_monitor import GPUMonitor
from retailmind import lightgbm_model
from retailmind.lstm_ablation import build_network, train, predict
from retailmind.transaction_features import load as load_metadata

ROOT=Path(__file__).resolve().parents[1]
CUTOFF=date(2011,12,9)
OUTPUT=ROOT/'artifacts/full-history-2011-12-09'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def coverage(panel, context):
    products=[]; early=[]
    for code, series in panel.items():
        if not series: continue
        first=min(series);last=min(max(series),CUTOFF)
        first_target=first+timedelta(days=context)
        if first_target<=min(last,CUTOFF-timedelta(days=28)):products.append(code)
        if max(first_target,CUTOFF-timedelta(days=27))<=last:early.append(code)
    return products,early


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest={'status':'in_progress','cutoff':CUTOFF.isoformat(),
              'purpose':'Fit all available qualifying product histories; previous chronological test periods are included in fitting',
              'limits':['No untouched test period remains for these full-history models',
                        'Observed sales are not necessarily true demand',
                        'Some products lack required history and cannot contribute',
                        'All historical eligible windows used; internal last 28 days reserved for early stopping',
                        'Full-history fitting is not a new unbiased accuracy evaluation']}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start_all=perf_counter();panel_path=ROOT/'data/processed/daily_sales.csv';price_path=ROOT/'reports/generated/daily-price-all.csv'
    print('Loading full product panel and historical prices...',flush=True)
    panel=load_panel(panel_path);metadata=load_metadata(price_path)
    missing_dates=sum((max(series)-min(series)).days+1!=len(series) for series in panel.values())
    if missing_dates:raise ValueError(f'{missing_dates} product series contain date gaps; coverage assumptions invalid')
    lgb_products,lgb_early=coverage(panel,28);lstm_products,lstm_early=coverage(panel,90)
    print(f'Panel: {len(panel)} codes; LightGBM {len(lgb_products)} training products; LSTM {len(lstm_products)} training products',flush=True)
    print('Training calendar LightGBM on CPU...',flush=True)
    started=perf_counter();booster=lightgbm_model.train(panel,CUTOFF,max_windows_per_product=None,calendar=True)
    lgb_seconds=perf_counter()-started
    booster.save_model(str(OUTPUT/'lightgbm.txt'))
    reloaded=lgb.Booster(model_file=str(OUTPUT/'lightgbm.txt'))
    examples=[code for code,s in panel.items() if CUTOFF in s and all(CUTOFF-timedelta(days=i) in s for i in range(90))]
    if not examples:raise ValueError('No product has a complete 90-day history at the final cutoff')
    example=sorted(examples)[0];history=[panel[example][CUTOFF-timedelta(days=i)] for i in range(89,-1,-1)]
    ref=lightgbm_model.predict(booster,history,CUTOFF,28,calendar=True)
    check=lightgbm_model.predict(reloaded,history,CUTOFF,28,calendar=True)
    if not np.allclose(ref,check,rtol=1e-7,atol=1e-7):raise ValueError('LightGBM round-trip check failed')
    print(f'LightGBM finished in {lgb_seconds:.1f}s; best iteration {booster.best_iteration}; artifact round-trip passed',flush=True)
    monitor=GPUMonitor(max_used_mib=1900).start()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; LSTM full training aborted')
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,1024**3/total),0)
        torch.backends.cudnn.benchmark=False;torch.cuda.reset_peak_memory_stats();monitor.check()
        print('Training sales + price LSTM on GPU with batch size 512...',flush=True)
        started=perf_counter();network,info=train(panel,CUTOFF,'sales_price',metadata=metadata,max_windows=None,batch_size=512,monitor=monitor)
        lstm_seconds=perf_counter()-started;phase_gpu=monitor.summarize_since(started)
        torch.save(network.state_dict(),OUTPUT/'lstm-state.pt')
        restored=build_network(2);restored.load_state_dict(torch.load(OUTPUT/'lstm-state.pt',map_location='cpu',weights_only=True));restored.eval()
        reference=predict(network,history,CUTOFF,28,'sales_price',metadata_values=metadata.at(example,CUTOFF),device='cuda',monitor=monitor)
        reconstructed=predict(restored,history,CUTOFF,28,'sales_price',metadata_values=metadata.at(example,CUTOFF),device='cpu')
        if not np.allclose(reference,reconstructed,rtol=1e-4,atol=1e-3):raise ValueError('LSTM state round-trip mismatch')
        peak_reserved=torch.cuda.max_memory_reserved()/1024**2;gpu=monitor.finish();monitor=None
        print(f'LSTM finished in {lstm_seconds:.1f}s; best epoch {info["best_epoch"]}; artifact round-trip passed',flush=True)
    finally:
        if monitor:
            monitor.stop_event.set()
            if monitor.thread:monitor.thread.join(timeout=6)
    manifest.update({'status':'complete','dataset_product_codes':len(panel),'training_scope':{'lightgbm_products':len(lgb_products),
        'lightgbm_early_stop_products':len(lgb_early),'lstm_products_expected':len(lstm_products),'lstm_early_stop_products':len(lstm_early),
        'all_qualifying_windows':True,'stop_period_start':(CUTOFF-timedelta(days=27)).isoformat()},
        'lightgbm':{'feature_set':'sales patterns + known calendar','device':'cpu','training_seconds':lgb_seconds,
                    'best_iteration':booster.best_iteration,'model_path':'lightgbm.txt'},
        'lstm':{'feature_set':'sales history + historical price','context':90,'layers':1,'hidden_units':32,'price_features':2,
                'batch_size':512,'max_epochs':20,'patience':3,'learning_rate':0.001,'seed':42,'training_seconds':lstm_seconds,
                'training_info':info,'model_path':'lstm-state.pt','training_gpu_samples':phase_gpu},
        'gpu':{'device':torch.cuda.get_device_name(0),'allocator_cap_gib':1,'watchdog_mib':1900,'peak_pytorch_reserved_mib':peak_reserved,
               'sampled_device':gpu,'note':'Sampling may miss short peaks; PyTorch allocator cap excludes some driver allocations'},
        'smoke_check':{'sample_stock_code':example,'horizon':28,'lightgbm_reload':'passed','lstm_reload':'passed'},
        'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'lightgbm':lgb.__version__},
        'hashes':{str(path.relative_to(ROOT)):sha(path) for path in [panel_path,price_path,ROOT/'src/retailmind/lstm_ablation.py',
                    ROOT/'src/retailmind/lightgbm_model.py',ROOT/'src/retailmind/transaction_features.py',Path(__file__),
                    OUTPUT/'lightgbm.txt',OUTPUT/'lstm-state.pt']},'total_pipeline_seconds':perf_counter()-start_all})
    if info['training_products']!=len(lstm_products):raise ValueError('LSTM training-product count failed coverage check')
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Completed full-history training; manifest:',OUTPUT/'manifest.json',flush=True)
    print(json.dumps({'lightgbm_seconds':lgb_seconds,'lstm_seconds':lstm_seconds,'lstm_info':info,'gpu':gpu},indent=2),flush=True)

if __name__=='__main__':main()
