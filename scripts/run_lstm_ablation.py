"""Compare four GPU LSTM feature variants on frozen validation cohorts."""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json
import platform
import sys
import torch
from retailmind.benchmark import historical_panel
from retailmind.forecast import load_panel, metrics
from retailmind.gpu_monitor import GPUMonitor
from retailmind.lstm_ablation import VARIANTS, train, predict
from retailmind.transaction_features import load as load_metadata

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields or rows[0].keys());writer.writeheader();writer.writerows(rows)


def main():
    cfg_path=ROOT/'configs/benchmark.json'; cfg=json.loads(cfg_path.read_text())
    folder=ROOT/'reports/generated'; panel_path=ROOT/'data/processed/daily_sales.csv'; metadata_path=folder/'daily-transaction-features.csv'
    cohorts={}
    for name,file in [('active','cohort.csv'),('intermittent','cohort-intermittent.csv')]:
        with (folder/file).open(newline='',encoding='utf-8') as f: cohorts[name]=[r['stock_code'] for r in csv.DictReader(f)]
        if len(cohorts[name])!=100 or len(set(cohorts[name]))!=100: raise ValueError('Cohort invalid; regenerate first')
    panel=load_panel(panel_path,products=set(cohorts['active'])|set(cohorts['intermittent']))
    metadata=load_metadata(metadata_path)
    monitor=GPUMonitor(max_used_mib=1900).start()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,1024**3/total),0)
        torch.backends.cudnn.benchmark=False;torch.cuda.reset_peak_memory_stats()
        rows=[]; forecasts=[]; missing=[]; runs=[]; selected={}
        context=cfg['context_days'];horizon=cfg['horizon_days']
        for group, cohort in cohorts.items():
            for phase in (0, 1):
                if phase == 0:
                    variants = ('sales', 'calendar', 'calendar_frequency')
                else:
                    totals = {v: [] for v in variants}
                    for item in forecasts:
                        if item['cohort'] == group and item['variant'] in totals:
                            totals[item['variant']].append(abs(item['actual'] - item['prediction']))
                    winner = min(totals, key=lambda v: sum(totals[v]) / len(totals[v]))
                    selected[group] = {'base_variant': winner, 'criterion': 'pooled validation MAE',
                                       'caution': 'price addition is exploratory on the same validation periods'}
                    variants = (winner + '_price',)
                for cutoff_text in cfg['validation_cutoffs']:
                    cutoff=date.fromisoformat(cutoff_text); historical=historical_panel(panel,cohort,cutoff);eligible=[]
                    for code in cohort:
                        h=historical.get(code,{})
                        hd=[cutoff-timedelta(days=i) for i in range(context-1,-1,-1)]
                        td=[cutoff+timedelta(days=i) for i in range(1,horizon+1)]
                        if not all(d in h for d in hd):
                            if phase == 0: missing.append({'cohort':group,'cutoff':cutoff_text,'stock_code':code,'reason':'insufficient_history'})
                        elif not all(d in panel.get(code,{}) for d in td):
                            if phase == 0: missing.append({'cohort':group,'cutoff':cutoff_text,'stock_code':code,'reason':'missing_target'})
                        else:
                            eligible.append((code,[h[d] for d in hd],[panel[code][d] for d in td]))
                    if not eligible: raise ValueError('No eligible evaluation products')
                    for variant in variants:
                        print(f'{group} {cutoff} {variant}: training',flush=True)
                        started=perf_counter();model, info=train(historical,cutoff,variant,metadata=metadata,context=context,batch_size=64,monitor=monitor)
                        elapsed=perf_counter()-started; gpu=monitor.summarize_since(started)
                        runs.append({'cohort':group,'cutoff':cutoff_text,'variant':variant,'training_seconds':elapsed,
                                     'seconds_per_training_product':elapsed/info['training_products'],**info,'gpu_training_samples':gpu,
                                     'scored_products':len(eligible),'cohort_size':len(cohort)})
                        for code,history,actual in eligible:
                            start=perf_counter();pred=predict(model,history,cutoff,horizon,variant,metadata_values=metadata.at(code,cutoff),context=context,monitor=monitor)
                            latency=perf_counter()-start
                            rows.append({'cohort':group,'cutoff':cutoff_text,'variant':variant,'stock_code':code,'inference_seconds':latency,**metrics(actual,pred,history)})
                            for step,(a,p) in enumerate(zip(actual,pred),1):
                                forecasts.append({'cohort':group,'cutoff':cutoff_text,'variant':variant,'stock_code':code,'horizon':step,
                                                  'target_date':(cutoff+timedelta(days=step)).isoformat(),'actual':a,'prediction':p})
                        del model
                        print(f'{group} {cutoff} {variant}: {elapsed:.1f}s, {info["training_products"]} training products, best epoch {info["best_epoch"]}',flush=True)
        torch.cuda.synchronize(); stats=monitor.finish();monitor=None
        output=folder/'lstm-feature-ablation';output.mkdir(parents=True,exist_ok=True)
        write_csv(output/'metrics.csv',rows);write_csv(output/'forecasts.csv',forecasts)
        write_csv(output/'exclusions.csv',missing,['cohort','cutoff','stock_code','reason'])
        meta={'phase':'validation','variants_tested':sorted({r['variant'] for r in runs}),'selected_base_per_cohort':selected,'batch_size':64,'seed':42,'device':torch.cuda.get_device_name(0),
              'torch_version':torch.__version__,'python':sys.version,'platform':platform.platform(),
              'pytorch_allocator_cap_gib':1,'watchdog_mib':1900,'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
              'device_sampling':stats,'runs':runs,'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [cfg_path,panel_path,metadata_path,folder/'cohort.csv',folder/'cohort-intermittent.csv',ROOT/'src/retailmind/lstm_ablation.py',ROOT/'src/retailmind/gpu_monitor.py',Path(__file__)]},
              'limitations':['exploratory validation only; historical data previously inspected','intermittent products with incomplete target windows excluded',
                             'GPU samples can miss brief peaks; PyTorch allocator cap does not cover all CUDA allocations',
                             'price summaries frozen at forecast origin, not future realized prices','frequency features frozen at origin during recursive prediction']}
        (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('Wrote results to',output,flush=True)
    finally:
        if monitor:
            monitor.stop_event.set()
            if monitor.thread: monitor.thread.join(timeout=6)

if __name__=='__main__':main()
