"""Standalone interactive EDA for the historical engineered features."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]


def main():
    f = pd.read_csv(ROOT/'reports/generated/feature-eda.csv',dtype={'stock_code':str})
    figs = [px.histogram(f,x='log_price',color='cohort',barmode='overlay',nbins=35,title='Historical price after log1p transform'),
            px.histogram(f,x='log_invoice_days',color='cohort',barmode='overlay',nbins=35,title='Historical invoice activity after log1p transform'),
            px.histogram(f,x='uk_share',color='cohort',barmode='overlay',nbins=30,title='Share of units sent to UK customers'),
            px.scatter(f,x='log_invoice_days',y='log_price',color='cohort',hover_data=['stock_code','cutoff','uk_share'],title='Price and transaction activity')]
    chart_json={}
    for i,fig in enumerate(figs):
        fig.update_layout(template='plotly_dark',paper_bgcolor='#111b2e',plot_bgcolor='#111b2e',font={'color':'#dce9f7'},height=370,margin={'l':60,'r':20,'t':60,'b':60})
        chart_json[f'c{i}']=fig.to_json()
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailMind | Feature EDA</title><style>body{background:#080f1e;color:#e7eef9;font:15px system-ui;margin:0}main{max-width:1400px;margin:auto;padding:32px}a{color:#5eead4}h1{font-size:42px;margin:15px 0}p{color:#acbdd1;line-height:1.5}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.panel{background:#111b2e;border:1px solid #25344b;border-radius:14px;padding:15px}.note{border-left:4px solid #5eead4;padding-left:18px}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:16px}}</style><script>__PLOTLY__</script></head><body><main><a href="data-exploration.html">← Data dashboard</a><h1>Feature engineering review</h1><p>Two validation origins × 100 active and 100 intermittent products. Points are product/cutoff snapshots, not independent products. No test-period features or model results are shown.</p><div class="panel note"><strong>What the models can know</strong><p>Price and invoice activity summarize the 90 days ending at the forecast cutoff; price is log1p-transformed. Invoice-days count distinct product invoices per day and sum daily counts, so an invoice spanning multiple dates can be counted twice. The UK share refers to customer destination, not manufacture. Missing values are encoded separately with indicators. Calendar values are known in advance.</p><p>Price and geography are present in all 400 selected snapshots. This is not a claim of completeness for the entire dataset or earlier training windows. The two cohorts were intentionally selected; distributions are not representative of all products.</p></div><div class="grid" style="margin-top:15px"><div class="panel" id="c0"></div><div class="panel" id="c1"></div><div class="panel" id="c2"></div><div class="panel" id="c3"></div></div><p>The charts are descriptive and do not establish causal relationships. Customer country is not production origin; product quality is not observed. See the accompanying notebook for tables and missingness.</p></main><script>const charts=__CHARTS__;Object.entries(charts).forEach(([id,raw])=>{let c=JSON.parse(raw);Plotly.newPlot(id,c.data,c.layout,{responsive:true,displaylogo:false})});</script></body></html>'''
    page=page.replace('__PLOTLY__',get_plotlyjs()).replace('__CHARTS__',json.dumps(chart_json).replace('</','<\\/'))
    output=ROOT/'reports/feature-eda.html';output.write_text(page,encoding='utf-8');print(output)

if __name__=='__main__': main()
