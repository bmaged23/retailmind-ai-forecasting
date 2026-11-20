"""Chronological train/validation/test fit and standalone HTML evaluation report."""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import csv, json, math, statistics
import sys
import numpy as np
import torch
import lightgbm as lgb
from retailmind.forecast import load_panel, metrics, seasonal_naive
from retailmind.lightgbm_model import train_split, predict as predict_lgb
from retailmind.lstm_ablation import build_network, extra_features, predict as predict_lstm
from retailmind.gpu_monitor import GPUMonitor
from retailmind.transaction_features import load as load_metadata

ROOT=Path(__file__).resolve().parents[1]
TRAIN_END=date(2011,8,31)
VALID_START=date(2011,9,1)
VALID_END=date(2011,9,28)
REFIT_END=VALID_END
TEST_ORIGIN=date(2011,10,4)
HORIZON=28
CONTEXT=90
OUT=ROOT/'artifacts/train-validation-test-2011-11-01'
REPORT=ROOT/'reports/model-evaluation.html'
METADATA=None

def make_examples(panel, start, end, variant='sales_price'):
    xs=[]; aux=[]; ys=[]; product_count=0
    for code, series in panel.items():
        if not series: continue
        first=min(series);last=max(series)
        dense=np.zeros((last-first).days+1,dtype=np.float32)
        observed=np.zeros(dense.shape,dtype=bool)
        for day,value in series.items():
            idx=(day-first).days;dense[idx]=value;observed[idx]=True
        if len(dense)<=CONTEXT: continue
        windows=np.lib.stride_tricks.sliding_window_view(dense,CONTEXT+1)
        target_dates=[first+timedelta(days=CONTEXT+i) for i in range(len(windows))]
        indices=np.fromiter((i for i,d in enumerate(target_dates) if start<=d<=end and observed[i+CONTEXT]),dtype=np.int64)
        if not len(indices): continue
        selected=windows[indices]
        inputs=selected[:,:CONTEXT]
        scales=np.maximum(inputs.mean(axis=1),1.)
        xs.append(np.log1p(inputs/scales[:,None]).astype(np.float32,copy=False))
        ys.append(np.log1p(selected[:,CONTEXT]/scales).astype(np.float32,copy=False))
        product_aux=[]
        for idx in indices:
            target=target_dates[int(idx)]
            product_aux.append(extra_features(variant,windows[int(idx),:CONTEXT],target,
                                              METADATA.at(code,target-timedelta(days=1))))
        aux.append(np.asarray(product_aux,dtype=np.float32))
        product_count+=1
        if product_count%500==0: print(f'Built windows for {product_count} products ({sum(len(z) for z in ys):,} examples)',flush=True)
    if not xs:return [],[],[]
    return np.concatenate(xs),np.concatenate(aux),np.concatenate(ys)

def train_lstm(panel, *, monitor=None):
    """Validation selects epoch; final fit includes train+validation labels."""
    from torch import nn
    torch.manual_seed(42);torch.set_num_threads(1)
    prep_started=perf_counter()
    print('Building vectorized LSTM training and validation windows...',flush=True)
    x,a,y=make_examples(panel,date.min,TRAIN_END)
    vx,va,vy=make_examples(panel,VALID_START,VALID_END)
    if len(x)==0 or len(vx)==0: raise ValueError('No LSTM train/validation examples')
    def tensors(xx,aa,yy):
        return torch.utils.data.TensorDataset(torch.from_numpy(np.asarray(xx,dtype=np.float32)).unsqueeze(-1),
          torch.from_numpy(np.asarray(aa,dtype=np.float32)),torch.tensor(yy,dtype=torch.float32))
    tr=tensors(x,a,y); val=tensors(vx,va,vy)
    def loader_for(data,shuffle):
        return torch.utils.data.DataLoader(data,batch_size=512,shuffle=shuffle,
          generator=torch.Generator().manual_seed(42) if shuffle else None,num_workers=4,
          pin_memory=True,persistent_workers=True,prefetch_factor=2)
    loader=loader_for(tr,True);vloader=loader_for(val,False)
    model=build_network(2).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    prep_seconds=perf_counter()-prep_started
    print(f'LSTM windows ready: {len(tr):,} training, {len(val):,} validation in {prep_seconds:.1f}s; CUDA training uses 4 loader workers.',flush=True)
    selection_started=perf_counter()
    best=float('inf');best_epoch=0;best_state=None;stale=0
    for epoch in range(20):
        model.train()
        for seq,extra,label in loader:
            if monitor: monitor.check()
            seq,extra,label=seq.cuda(non_blocking=True),extra.cuda(non_blocking=True),label.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(seq,extra),label);loss.backward();optimizer.step()
        model.eval();total=0.
        with torch.inference_mode():
            for seq,extra,label in vloader:
                total+=nn.functional.mse_loss(model(seq.cuda(non_blocking=True),extra.cuda(non_blocking=True)),label.cuda(non_blocking=True),reduction='sum').item()
        score=total/len(val)
        if score<best-1e-6:
            best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=3: break
    selection_seconds=perf_counter()-selection_started
    del loader,vloader
    import gc;gc.collect()
    # Final refit reuses the already-built train and validation tensors; no test
    # examples are present and the expensive window construction runs only once.
    final_data=torch.utils.data.ConcatDataset([tr,val])
    refit_started=perf_counter()
    final_loader=loader_for(final_data,True)
    model=build_network(2).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(best_epoch):
        model.train()
        for seq,extra,label in final_loader:
            if monitor: monitor.check()
            seq,extra,label=seq.cuda(non_blocking=True),extra.cuda(non_blocking=True),label.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(seq,extra),label);loss.backward();optimizer.step()
    model.eval()
    refit_seconds=perf_counter()-refit_started
    result={'best_epoch_validation':best_epoch,'epochs_run_validation':epoch+1,
            'refit_epochs':best_epoch,'training_windows':len(tr),'validation_windows':len(val),
            'validation_loss':best,'data_loader_workers':4,'batch_size':512,
            'window_preparation_seconds':prep_seconds,'validation_selection_seconds':selection_seconds,
            'refit_seconds':refit_seconds}
    return model,result

def series_window(series, cutoff, days):
    start=cutoff-timedelta(days=days-1)
    return [series.get(start+timedelta(days=i),0.) for i in range(days)]

def main():
    global METADATA
    if '--report-only' in sys.argv:
        result=json.loads((OUT/'metrics.json').read_text())
        with (OUT/'test-forecasts.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        forecasts=[{**r,'horizon':int(r['horizon']),'actual':float(r['actual']),
                    'prediction':float(r['prediction'])} for r in rows]
        scored=sorted({r['stock_code'] for r in forecasts})
        make_report(result['test'],forecasts,scored,result['lightgbm'],result['lstm_selection'],
          result['training_seconds']['lightgbm_select_and_refit'],
          result['lstm_selection']['validation_selection_seconds'],
          result['lstm_selection']['refit_seconds'],result['gpu'],
          result['lstm_peak_pytorch_reserved_mib'],
          missing=result['excluded_incomplete_test_labels'],
          candidates=result['forecast_candidates_selected_pre_test'])
        return
    OUT.mkdir(parents=True,exist_ok=True)
    panel=load_panel(ROOT/'data/processed/daily_sales.csv')
    METADATA=load_metadata(ROOT/'reports/generated/daily-price-all.csv')
    # Materialize only observations available through validation. Test labels are
    # absent from every fitting/model-selection call.
    fit_panel={c:{d:q for d,q in s.items() if d<=VALID_END} for c,s in panel.items()}
    started=perf_counter();lgb_model,lgb_info=train_split(fit_panel,TRAIN_END,VALID_START,VALID_END,calendar=True);lgb_seconds=perf_counter()-started
    # train_split returns the validation-selected round count refit through Sep 28.
    lgb_model.save_model(str(OUT/'lightgbm.txt'))
    monitor=GPUMonitor(max_used_mib=1900).start()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable for 2 GB-capped LSTM')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,1024**3/total),0)
    torch.backends.cudnn.benchmark=False;torch.cuda.reset_peak_memory_stats();monitor.check()
    started=perf_counter();lstm_model,lstm_info=train_lstm(fit_panel,monitor=monitor);lstm_select_seconds=perf_counter()-started
    selected_epochs=lstm_info['best_epoch_validation']
    lstm_final={'refit_epochs':selected_epochs,'training_windows':lstm_info['training_windows']+lstm_info['validation_windows']}
    lstm_refit_seconds=lstm_info['refit_seconds']
    torch.save(lstm_model.state_dict(),OUT/'lstm-state.pt')
    # Candidate products are chosen using only history through the forecast origin.
    # Build and persist all predictions before reading any held-out target labels.
    candidate=[]
    for code,series in panel.items():
        if all(TEST_ORIGIN-timedelta(days=i) in series for i in range(CONTEXT-1,-1,-1)):
            candidate.append(code)
    print(f'Forecasting {len(candidate)} products selected from pre-test history...',flush=True)
    forecasts=[]
    for code in candidate:
        series=panel[code]
        th=series_window(series,TEST_ORIGIN,CONTEXT)
        price_values=METADATA.at(code,TEST_ORIGIN)
        for name,fn in [('seasonal_naive',lambda:seasonal_naive(th,HORIZON)),('lightgbm',lambda:predict_lgb(lgb_model,th,TEST_ORIGIN,HORIZON,calendar=True)),('lstm',lambda:predict_lstm(lstm_model,th,TEST_ORIGIN,HORIZON,'sales_price',metadata_values=price_values,device='cuda',monitor=monitor))]:
            start=perf_counter();pred=fn();latency=perf_counter()-start
            for step,p in enumerate(pred,1):forecasts.append({'model':name,'stock_code':code,'target_date':(TEST_ORIGIN+timedelta(days=step)).isoformat(),'horizon':step,'prediction':p,'inference_seconds':latency})
    # From here onward test labels are accessed solely for final scoring.
    scoreable={c for c in candidate if all(TEST_ORIGIN+timedelta(days=i) in panel[c] for i in range(1,HORIZON+1))}
    missing=len(candidate)-len(scoreable)
    for row in forecasts:
        code=row['stock_code']
        if code in scoreable: row['actual']=panel[code][date.fromisoformat(row['target_date'])]
    forecasts=[r for r in forecasts if r['stock_code'] in scoreable]
    per_product=[]
    for code in sorted(scoreable):
        history=series_window(panel[code],TEST_ORIGIN,CONTEXT)
        rows=[r for r in forecasts if r['stock_code']==code]
        for model_name in ['seasonal_naive','lightgbm','lstm']:
            model_rows=[r for r in rows if r['model']==model_name]
            actual=[r['actual'] for r in model_rows];pred=[r['prediction'] for r in model_rows]
            per_product.append({'model':model_name,'stock_code':code,'inference_seconds':model_rows[0]['inference_seconds'],**metrics(actual,pred,history)})
    gpu=monitor.finish();peak_reserved=torch.cuda.max_memory_reserved()/1024**2
    def aggregate(rows):
        result=[]
        for name in ['seasonal_naive','lightgbm','lstm']:
            group=[r for r in rows if r['model']==name]
            result.append({'model':name,'products':len(group),**{m:statistics.mean(r[m] for r in group if r[m] is not None) for m in ('mae','rmse','wape','mase')},'mean_inference_seconds':statistics.mean([r['inference_seconds'] for r in group]) if 'inference_seconds' in group[0] else None})
        return result
    test_metrics=aggregate(per_product)
    for metric in test_metrics:
        model_rows=[r for r in forecasts if r['model']==metric['model']]
        total_actual=sum(r['actual'] for r in model_rows)
        metric['wape']=sum(abs(r['actual']-r['prediction']) for r in model_rows)/total_actual if total_actual else None
    with (OUT/'test-forecasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=forecasts[0].keys());w.writeheader();w.writerows(forecasts)
    (OUT/'metrics.json').write_text(json.dumps({'train_period':[None,TRAIN_END.isoformat()],'validation_period':[VALID_START.isoformat(),VALID_END.isoformat()],'test_period':[(TEST_ORIGIN+timedelta(days=1)).isoformat(),(TEST_ORIGIN+timedelta(days=HORIZON)).isoformat()],'forecast_candidates_selected_pre_test':len(candidate),'scored_products':len(scoreable),'excluded_incomplete_test_labels':missing,'test':test_metrics,'lightgbm':lgb_info,'lstm_selection':lstm_info,'lstm_refit':lstm_final,'training_seconds':{'lightgbm_select_and_refit':lgb_seconds,'lstm_total_pipeline_including_window_preparation_and_refit':lstm_select_seconds,'lstm_window_preparation':lstm_info['window_preparation_seconds'],'lstm_validation_selection':lstm_info['validation_selection_seconds'],'lstm_refit':lstm_refit_seconds},'gpu':gpu,'lstm_peak_pytorch_reserved_mib':peak_reserved},indent=2)+'\n')
    make_report(test_metrics,forecasts,sorted(scoreable),lgb_info,lstm_info,lgb_seconds,
      lstm_info['validation_selection_seconds'],lstm_refit_seconds,gpu,peak_reserved,
      missing=missing,candidates=len(candidate))
    print(json.dumps({'test':test_metrics,'products':len(scoreable),'lightgbm_seconds':lgb_seconds,
      'lstm_selection_seconds':lstm_info['validation_selection_seconds'],
      'lstm_refit_seconds':lstm_refit_seconds,'gpu':gpu},indent=2),flush=True)

def make_report(test_metrics,forecasts,eligible,lgb_info,lstm_info,lgbsec,lstmselect,lstmrefit,gpu,peak,*,missing=0,candidates=0):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Test MAE by model','Test WAPE by model','Test error by forecast horizon','Example product forecasts'))
    names=[r['model'] for r in test_metrics]
    fig.add_trace(go.Bar(x=names,y=[r['mae'] for r in test_metrics],marker_color=['#8793a6','#44d7b6','#7c8cff']),row=1,col=1)
    fig.add_trace(go.Bar(x=names,y=[r['wape']*100 for r in test_metrics],marker_color=['#8793a6','#44d7b6','#7c8cff']),row=1,col=2)
    for model in names:
        group=[r for r in forecasts if r['model']==model]
        perday=[]
        for h in range(1,HORIZON+1):
            perday.append(statistics.mean(abs(r['actual']-r['prediction']) for r in group if r['horizon']==h))
        fig.add_trace(go.Scatter(x=list(range(1,HORIZON+1)),y=perday,name=model,mode='lines'),row=2,col=1)
    sample=eligible[0]
    actual=[r for r in forecasts if r['stock_code']==sample and r['model']=='lightgbm']
    fig.add_trace(go.Scatter(x=[r['horizon'] for r in actual],y=[r['actual'] for r in actual],name=f'Actual {sample}',mode='lines+markers'),row=2,col=2)
    for model in ['seasonal_naive','lightgbm','lstm']:
        rows=[r for r in forecasts if r['stock_code']==sample and r['model']==model]
        fig.add_trace(go.Scatter(x=[r['horizon'] for r in rows],y=[r['prediction'] for r in rows],name=f'{model} forecast',mode='lines'),row=2,col=2)
    fig.update_layout(template='plotly_dark',height=800,barmode='group',legend=dict(orientation='h',y=-.15))
    fig.update_xaxes(title_text='Model',row=1,col=1);fig.update_yaxes(title_text='Units / product / day',row=1,col=1)
    fig.update_yaxes(title_text='Percent',row=1,col=2);fig.update_xaxes(title_text='Forecast day',row=2,col=1);fig.update_yaxes(title_text='Mean absolute error',row=2,col=1)
    table=''.join(f"<tr><td>{r['model']}</td><td>{r['products']}</td><td>{r['mae']:.3f}</td><td>{r['rmse']:.3f}</td><td>{r['wape']:.2%}</td><td>{r['mase']:.3f}</td><td>{r['mean_inference_seconds']:.4f}</td></tr>" for r in test_metrics)
    html=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RetailMind | Model evaluation</title><style>body{{margin:0;background:#08111f;color:#e7eef8;font:15px system-ui}}main{{max-width:1450px;margin:auto;padding:32px}}h1{{font-size:38px}}p{{color:#b8c7d9;line-height:1.6}}.card{{background:#111d2e;border:1px solid #26374c;border-radius:14px;padding:20px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:11px;border-bottom:1px solid #26374c;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{border-left:4px solid #44d7b6;padding-left:16px}}#chart{{width:100%}}</style></head><body><main><h1>RetailMind Forecast Model Evaluation</h1><p>Chronological held-out evaluation of global LightGBM and LSTM against a seasonal-naive baseline. One 28-day test forecast per scoreable product.</p><div class="card note"><b>Split design</b><p>Training labels end {TRAIN_END}; validation {VALID_START}–{VALID_END} selects LightGBM boosting rounds and LSTM epoch count. Both models are then refit through {REFIT_END}. Test forecast origin: {TEST_ORIGIN}; test targets {TEST_ORIGIN+timedelta(days=1)}–{TEST_ORIGIN+timedelta(days=HORIZON)}. No test labels were available to fitting or model selection. Candidate products ({candidates}) were selected using history known by the forecast origin; {missing} candidates lacked complete recorded test labels and were excluded only at final scoring. {len(eligible)} products are scored. This measures observed recorded sales, not latent demand.</p></div><div class="card"><h2>Test metrics</h2><table><tr><th>Model</th><th>Products</th><th>MAE</th><th>RMSE</th><th>WAPE</th><th>MASE</th><th>Mean inference sec/product</th></tr>{table}</table></div><div class="card" id="chart"></div><div class="card"><h2>Training and hardware</h2><p>LightGBM selected {lgb_info['best_iteration']} boosting rounds; selection plus refit took {lgbsec:.1f}s. LSTM validation selected epoch {lstm_info['best_epoch_validation']} of {lstm_info['epochs_run_validation']}; validation selection took {lstmselect:.1f}s and final refit took {lstmrefit:.1f}s. LSTM: one 32-unit layer, 90-day sales sequence plus historical 90-day price features, Adam lr=0.001, batch size 512, max 20 epochs, patience 3. GPU sampled peak memory {gpu['peak_device_memory_mib']} MiB; PyTorch reserved peak {peak:.1f} MiB. Device-wide memory is sampled every 0.25s and the watchdog abort threshold is 1900 MiB; brief spikes between samples may be missed.</p></div><div class="card"><h2>Interpretation limits</h2><p>Source is historical Online Retail II (2009–2011). Sales are treated as recorded gross quantities. Missing target rows are excluded rather than assumed zero. Inventory, lost sales, and availability are not observed. Results concern this fixed eligible set and one test origin. Test horizon is short and does not establish present-day performance.</p></div><p>MAE and RMSE use units per product per day; WAPE is pooled absolute error divided by pooled actual units; MASE uses the product's training-history seasonal scale and reported here as mean across products.</p></main><script>__PLOTLY__</script><script>document.getElementById('chart').innerHTML=__FIG__;</script></body></html>'''
    # Plotly's standalone HTML div is embedded so the report works offline.
    import plotly.io as pio
    div=pio.to_html(fig,full_html=False,include_plotlyjs='inline')
    html=html.replace('<script>__PLOTLY__</script><script>document.getElementById(\'chart\').innerHTML=__FIG__;</script>',div)
    REPORT.write_text(html,encoding='utf-8')
    print('HTML report:',REPORT,flush=True)

if __name__=='__main__':main()
