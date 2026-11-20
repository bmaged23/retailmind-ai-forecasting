"""Score frozen zero-shot/adapted forecasts on the shared October test."""
from collections import defaultdict
from pathlib import Path
import csv, json, math, statistics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
from common import BASE_DIR, TEST_DIR, HORIZON

REPORT=TEST_DIR/'foundation_model_evaluation.html'
SCORE=TEST_DIR/'scorecard.json'
PREVIEW=TEST_DIR/'portfolio_forecast_preview.png'
PILOT_FILES={'chronos2_lora_pilot_100_steps',
             'chronos2_lora_interval100_validation_pilot'}
PILOT_MODELS={'chronos2-lora'}

def main():
    actuals={};baseline={}
    with (BASE_DIR/'test-forecasts.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            key=(row['stock_code'],row['target_date'])
            actuals.setdefault(key,float(row['actual']))
            if row['model'] in {'seasonal_naive','lightgbm','lstm'}:
                baseline.setdefault(row['model'],{})[key]={
                    'prediction':float(row['prediction']),
                    'inference_seconds':float(row['inference_seconds'])}
    expected=set(actuals)
    results={}
    for path in sorted((TEST_DIR/'forecasts').rglob('*.csv')):
        if path.stem in PILOT_FILES:continue
        values={};name=None
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                name=row['model']
                key=(row['stock_code'],row['target_date'])
                if key in values:raise ValueError(f'Duplicate forecast {name} {key}')
                values[key]={'prediction':float(row['prediction']),
                  'inference_seconds':float(row.get('inference_seconds_per_series') or 0.)}
        if name in PILOT_MODELS:continue
        if not values:continue
        if set(values)!=expected:
            raise ValueError(f'{name} has {len(values)} rows/keys, expected the frozen {len(expected)} test rows')
        results[name]=values
    if not results:raise ValueError('No complete foundation-model forecasts are available yet')
    results={**baseline,**results}
    summary=[];horizon_error={}
    target_days=sorted({k[1] for k in expected})
    metadata_dir=TEST_DIR/'metadata'
    for name,rows in results.items():
        errors=[abs(actuals[k]-rows[k]['prediction']) for k in expected]
        squared=[(actuals[k]-rows[k]['prediction'])**2 for k in expected]
        total_actual=sum(actuals.values())
        latency_by_product=defaultdict(list)
        for (code,_),r in rows.items():latency_by_product[code].append(r['inference_seconds'])
        mean_latency=statistics.mean(
              [statistics.mean(v) for v in latency_by_product.values()])
        metadata_name=('chronos2_lora_full_series_epoch' if name=='chronos2_zero_shot'
                       else name)
        metadata_path=metadata_dir/f'{metadata_name}.json'
        if mean_latency==0 and metadata_path.exists():
            metadata=json.loads(metadata_path.read_text())
            inference=metadata.get('inference_seconds',{})
            phase='zero_shot' if name.endswith('zero_shot') else 'lora'
            seconds=inference.get(phase)
            if seconds is not None:
                product_count=metadata.get('products',metadata.get('cohort_products',1))
                mean_latency=seconds/product_count
        summary.append({'model':name,'products':len(latency_by_product),'forecast_rows':len(rows),
          'mae':statistics.mean(errors),'rmse':math.sqrt(statistics.mean(squared)),
          'wape':sum(errors)/total_actual if total_actual else None,
          'mean_inference_seconds_per_series':mean_latency})
        horizon_error[name]=[]
        for h in range(1,HORIZON+1):
            # Use the ordered dates from the frozen contiguous 28-day window.
            day=target_days[h-1]
            horizon_error[name].append(statistics.mean(
              abs(actuals[k]-rows[k]['prediction']) for k in expected if k[1]==day))
    summary.sort(key=lambda r:r['mae'])
    make_report(summary,actuals,results,horizon_error)
    content={'test_period':['2011-10-05','2011-11-01'],'forecast_origin':'2011-10-04',
      'products':len({k[0] for k in expected}),'rows_per_model':len(expected),
      'models':summary,'fine_tuning_pairs':paired_deltas(summary),
      'note':'The user requested reuse of the prior test window. This is a paired comparison on that historical split, not a newly untouched final benchmark.'}
    SCORE.write_text(json.dumps(content,indent=2)+'\n')
    print(json.dumps(content,indent=2),flush=True)
    print('HTML report:',REPORT,flush=True)

def paired_deltas(summary):
    by={r['model']:r for r in summary};out=[]
    pairs=[('chronos2_zero_shot','chronos2_lora_full_series_epoch'),
           ('timestral209m_zero_shot','timestral209m_lora'),
           ('timesfm25_zero_shot','timesfm25_lora'),
           ('timesfm3_zero_shot','timesfm3_head_tuned'),
           ('tirex2_zero_shot','tirex2_head_tuned'),
           ('toto2_22m_zero_shot','toto2_22m_custom_lora'),
           ('toto1_zero_shot','toto1_finetuned')]
    for zero,tuned in pairs:
        if zero in by and tuned in by:
            out.append({'zero_shot':zero,'fine_tuned':tuned,
              'mae_change':by[tuned]['mae']-by[zero]['mae'],
              'wape_change':by[tuned]['wape']-by[zero]['wape']})
    return out

def make_report(summary,actuals,results,horizon_error):
    names=[r['model'] for r in summary]
    fig=make_subplots(rows=2,cols=2,subplot_titles=('MAE on shared test','WAPE on shared test',
      'Absolute error by forecast day','Example high-volume SKU'))
    fig.add_trace(go.Bar(x=names,y=[r['mae'] for r in summary],
      marker_color=['#39dcb6' if 'lora' in n or 'fine' in n or 'tuned' in n else '#7288ff' for n in names]),row=1,col=1)
    fig.add_trace(go.Bar(x=names,y=[100*r['wape'] for r in summary],
      marker_color=['#39dcb6' if 'lora' in n or 'fine' in n or 'tuned' in n else '#7288ff' for n in names]),row=1,col=2)
    for name in names:
        fig.add_trace(go.Scatter(x=list(range(1,HORIZON+1)),y=horizon_error[name],
          name=name,mode='lines'),row=2,col=1)
    total_by_product=defaultdict(float)
    for (code,_),value in actuals.items():total_by_product[code]+=value
    sample=max(total_by_product,key=total_by_product.get)
    days=sorted({k[1] for k in actuals})
    fig.add_trace(go.Scatter(x=list(range(1,HORIZON+1)),
      y=[actuals[(sample,day)] for day in days],name=f'Actual {sample}',mode='lines+markers'),row=2,col=2)
    for name in names:
        rows=results[name]
        if (sample,days[0]) not in rows:continue
        fig.add_trace(go.Scatter(x=list(range(1,HORIZON+1)),
          y=[rows[(sample,day)]['prediction'] for day in days],name=name,mode='lines'),row=2,col=2)
    fig.update_layout(template='plotly_dark',height=850,barmode='group',
      legend=dict(orientation='h',y=-.2),margin=dict(b=120))
    fig.update_xaxes(title_text='Model',row=1,col=1);fig.update_yaxes(title_text='Units / SKU / day',row=1,col=1)
    fig.update_yaxes(title_text='Percent',row=1,col=2);fig.update_xaxes(title_text='Forecast day',row=2,col=1)
    fig.update_yaxes(title_text='Mean absolute error',row=2,col=1)
    aggregate_actual=defaultdict(float)
    aggregate_prediction={name:defaultdict(float) for name in names}
    for (code,day),actual in actuals.items():
        aggregate_actual[day]+=actual
        for name in names:
            aggregate_prediction[name][day]+=results[name][(code,day)]['prediction']
    days=sorted(aggregate_actual)
    wide=go.Figure()
    wide.add_trace(go.Scatter(x=days,y=[aggregate_actual[d] for d in days],
      name='Ground truth',mode='lines+markers',line=dict(color='#f4b942',width=4)))
    for name in names:
        wide.add_trace(go.Scatter(x=days,y=[aggregate_prediction[name][d] for d in days],
          name=name,mode='lines',line=dict(width=2)))
    wide.update_layout(template='plotly_dark',height=520,margin=dict(l=65,r=30,t=45,b=70),
      title='Portfolio daily sales: ground truth vs model forecasts',
      xaxis_title='Date',yaxis_title='Total units across 2,693 products',
      hovermode='x unified',legend=dict(orientation='h',y=-.24))
    # Put Plotly.js in the first chart block. Otherwise its Plotly.newPlot call
    # runs before the later chart block loads the library and remains blank.
    wide_chart=pio.to_html(wide,full_html=False,include_plotlyjs='inline')
    fig_preview,ax=plt.subplots(figsize=(18,7),layout='constrained')
    x=list(range(len(days)))
    ax.plot(x,[aggregate_actual[d] for d in days],label='Ground truth',
      color='#d79a00',linewidth=3.5,marker='o',markersize=3)
    for name in names:
        ax.plot(x,[aggregate_prediction[name][d] for d in days],label=name,linewidth=2)
    ax.set_title('Daily portfolio sales: ground truth vs model forecasts',fontsize=17)
    ax.set_xlabel('Date');ax.set_ylabel('Total units across 2,693 products')
    tick_positions=list(range(0,len(days),4))
    if tick_positions[-1]!=len(days)-1:tick_positions.append(len(days)-1)
    ax.set_xticks(tick_positions,[days[i][5:] for i in tick_positions])
    ax.grid(alpha=.25);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False)
    fig_preview.savefig(PREVIEW,dpi=150,facecolor='white')
    plt.close(fig_preview)
    rows=''.join(f"<tr><td>{r['model']}</td><td>{r['products']:,}</td><td>{r['mae']:.3f}</td><td>{r['rmse']:.3f}</td><td>{r['wape']:.2%}</td><td>{r['mean_inference_seconds_per_series']:.4f}</td></tr>" for r in summary)
    chart=pio.to_html(fig,full_html=False,include_plotlyjs=False)
    toto_custom_note=('<p class="warn">Toto 2.0 custom LoRA is an experimental research adaptation using the model\'s differentiable forward pass. Datadog does not currently publish an official Toto 2.0 fine-tuning workflow; do not treat this as vendor-supported fine-tuning.</p>' if 'toto2_22m_custom_lora' in results else '')
    head_tuning_note=('<p class="warn">TimesFM 3.0 and TiRex-2 tuned runs are experimental output-head fine-tuning with the pretrained backbones frozen. They are not vendor-provided LoRA workflows. TimesFM 3.0 fine-tuned weights are not distributable under its non-commercial license.</p>' if 'timesfm3_head_tuned' in results or 'tirex2_head_tuned' in results else '')
    toto_custom_note += head_tuning_note
    html=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailMind | Foundation model comparison</title><style>body{{margin:0;background:#07111f;color:#e8eff8;font:15px system-ui}}main{{max-width:1500px;margin:auto;padding:30px}}h1{{font-size:38px}}p{{color:#b6c5d8;line-height:1.65}}.card{{background:#111c2d;border:1px solid #26364d;border-radius:14px;padding:20px;margin:16px 0}}.note{{border-left:4px solid #39dcb6;padding-left:16px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:11px;border-bottom:1px solid #293a50;text-align:right}}td:first-child,th:first-child{{text-align:left}}.warn{{color:#ffdc99}}</style></head><body><main><h1>Foundation Model Comparison</h1><p>Zero-shot and fine-tuned foundation model forecasts, compared with LightGBM and LSTM baselines on the same 2,693 products and October 5–November 1, 2011 test dates.</p><div class="card note"><b>Protocol</b><p>Forecast origin October 4; 90-day context; 28-day horizon. Fine-tuning uses training data through August 31; validation data through September 28 selects the checkpoint and drives early stopping. Saved model forecasts are joined with test labels only for scoring.</p>{toto_custom_note}<p class="warn">This test period has already been used for the earlier LightGBM/LSTM evaluation and is reused here at the user's request. Treat this as a paired comparison on a known split, not a newly untouched final benchmark.</p></div><div class="card"><h2>Scores</h2><table><tr><th>Model / method</th><th>Products</th><th>MAE</th><th>RMSE</th><th>WAPE</th><th>Inference sec / product</th></tr>{rows}</table><p>MAE and RMSE are units per product per day. WAPE is pooled absolute error divided by pooled actual units. Lower is better for all three.</p></div><div class="card"><h2>Daily portfolio forecast</h2><p>Daily unit totals summed across all 2,693 products. Hover over the lines to compare the ground truth and each model.</p>{wide_chart}</div><div class="card">{chart}</div><div class="card"><h2>Interpretation</h2><p>Recorded gross sales are the target; inventory, lost sales, and product availability are absent from the source data. The dataset is historical. Score changes from fine-tuning are empirical for this split and may not generalize to a later period.</p></div></main></body></html>'''
    REPORT.write_text(html,encoding='utf-8')

if __name__=='__main__':main()
