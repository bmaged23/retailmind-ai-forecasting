"""Reproducible exploratory analysis and standalone HTML dashboard."""
from pathlib import Path
import hashlib
import html
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]


def analyze():
    path = ROOT / 'data/processed/daily_sales.csv'
    frame = pd.read_csv(path, dtype={'stock_code': str}, parse_dates=['date'])
    audit = json.loads((ROOT / 'data/processed/audit.json').read_text())
    daily = frame.groupby('date', as_index=False).agg(units=('quantity', 'sum'), active_skus=('quantity', lambda s: (s > 0).sum()))
    product = frame.groupby('stock_code').agg(units=('quantity', 'sum'), days=('quantity', 'size'), sale_days=('quantity', lambda s: (s > 0).sum()), max_daily=('quantity', 'max')).reset_index()
    product['zero_pct'] = 100 * (1 - product.sale_days / product.days)
    product = product.sort_values('units', ascending=False).reset_index(drop=True)
    product['share_pct'] = product.units / product.units.sum() * 100
    daily['weekday'] = daily.date.dt.day_of_week
    week = daily.groupby('weekday').agg(mean_units=('units', 'mean'), days=('units', 'size')).reset_index()
    return frame, audit, daily, product, week


def create_dashboard(frame, audit, daily, product, week):
    dates = [d.strftime('%Y-%m-%d') for d in daily.date]
    total = product.units.sum()
    top10 = 100 * product.head(10).units.sum() / total
    zero = 100 * (frame.quantity == 0).mean()
    raw = audit['rows_read']
    fig = go.Figure()
    fig.add_scatter(x=dates, y=daily.units, name='Units sold', line={'color':'#5eead4','width':1.5}, fill='tozeroy', fillcolor='rgba(45,212,191,.08)')
    fig.update_layout(title='Daily recorded gross units', xaxis_title='Date', yaxis_title='Units', xaxis_rangeslider_visible=True)
    bars = go.Figure(go.Bar(x=product.head(15).stock_code, y=product.head(15).units, marker_color='#8b8cf8'))
    bars.update_layout(title='Top 15 products by gross units', xaxis_title='Stock code', yaxis_title='Units')
    weekday = go.Figure(go.Bar(x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], y=week.set_index('weekday').mean_units.reindex(range(7),fill_value=0), marker_color='#38bdf8', customdata=week.set_index('weekday').days.reindex(range(7),fill_value=0), hovertemplate='%{x}<br>Mean %{y:,.0f} units<br>%{customdata} recorded calendar days<extra></extra>'))
    weekday.update_layout(title='Average daily units by weekday', yaxis_title='Mean units per day')
    hist = go.Figure(go.Histogram(x=product.zero_pct, xbins={'start':0,'end':100,'size':5}, marker_color='#fb923c'))
    hist.update_layout(title='How often does a product have zero recorded sales?', xaxis_title='Zero-sales days within observed product span (%)', yaxis_title='Products')
    charts=[]
    for name, chart in [('trend',fig),('products',bars),('week',weekday),('zeros',hist)]:
        chart.update_layout(template='plotly_dark', paper_bgcolor='#111b2e', plot_bgcolor='#111b2e', font={'family':'Arial','color':'#dbe7f7'}, margin={'l':60,'r':24,'t':65,'b':60}, height=380)
        charts.append((name, chart.to_json()))
    rows = product.head(200).copy()
    records = [{'sku':str(r.stock_code),'units':round(float(r.units),2),'days':int(r.days),'sale_days':int(r.sale_days),'zero':round(float(r.zero_pct),1),'max':round(float(r.max_daily),2)} for r in rows.itertuples()]
    catalog_path = ROOT / 'reports/generated/product-features-all.csv'
    if not catalog_path.exists():
        catalog_path = ROOT / 'reports/generated/product-features.csv'
    catalog = []
    if catalog_path.exists():
        with catalog_path.open(newline='', encoding='utf-8') as handle:
            import csv
            for item in csv.DictReader(handle):
                catalog.append({'sku': item['stock_code'], 'description': item['description'],
                                'shape': item['code_shape'], 'base': item.get('candidate_code_base', ''), 'pack': item.get('candidate_pack_size', ''), 'price': item['weighted_mean_price_lookback'],
                                'tags': ', '.join(k[5:] for k, value in item.items() if k.startswith('text_') and value == '1')})
    catalog_json = json.dumps(catalog).replace('</', '<\\/')
    charts_json = json.dumps(dict(charts)).replace('</','<\\/')
    records_json = json.dumps(records).replace('</','<\\/')
    body = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailMind | Data Explorer</title><style>
    *{box-sizing:border-box}body{margin:0;background:#080f1e;color:#e6edf8;font:15px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:36px}header{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:30px}.eyebrow{color:#5eead4;text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:700}h1{font-size:clamp(30px,4vw,48px);margin:5px 0;letter-spacing:-.04em}h2{font-size:22px;margin:0 0 8px}p{color:#aabbd0;margin:5px 0}.badge{border:1px solid #345268;color:#9be6dc;border-radius:40px;padding:8px 14px;font-size:12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:#111b2e;border:1px solid #25344b;border-radius:14px;padding:20px}.card span{color:#91a5be;font-size:12px}.card strong{display:block;font-size:28px;letter-spacing:-.03em}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}.full{grid-column:1/-1}.chart{width:100%;min-height:380px}.note{font-size:12px;color:#93a7be}.insights{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}.insights li{margin:9px 0;color:#b7c7d9}.tag{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#5eead4;font-weight:700}.warn{color:#fbbf80}input{background:#0b1424;color:white;border:1px solid #354861;border-radius:8px;padding:11px;width:min(100%,350px)}.tablewrap{overflow:auto;max-height:540px;margin-top:16px}table{width:100%;border-collapse:collapse;font-size:13px;text-align:right}th{position:sticky;top:0;background:#17243a;color:#99abc3;cursor:pointer}th,td{padding:12px;border-bottom:1px solid #25344b}th:first-child,td:first-child{text-align:left}footer{color:#8396ae;font-size:12px;margin:30px 0}a{color:#5eead4}@media(max-width:800px){main{padding:18px}.cards,.grid,.insights{grid-template-columns:1fr 1fr}header{align-items:start;flex-direction:column}}@media(max-width:500px){.cards,.grid,.insights{grid-template-columns:1fr}.full{grid-column:auto}}
    </style><script>__PLOTLY__</script></head><body><main><header><div><div class="eyebrow">RetailMind AI / research briefing</div><h1>Inside the sales data.</h1><p>A first look at the signals, gaps and risks behind a 28-day forecasting benchmark.</p></div><div class="badge">Exploratory analysis · 2009–2011</div></header>
    <section class="cards"><div class="card"><span>TRANSACTION LINES</span><strong>__RAW__</strong><span>Two UCI workbook sheets</span></div><div class="card"><span>PRODUCTS IN PANEL</span><strong>__SKUS__</strong><span>After current cleaning rules</span></div><div class="card"><span>RECORDED GROSS UNITS</span><strong>__UNITS__</strong><span>Returns excluded</span></div><div class="card"><span>ZERO-SALES PRODUCT DAYS</span><strong>__ZERO__%</strong><span>Within first-to-last sale span</span></div></section>
    <section class="grid"><div class="panel full"><div id="trend" class="chart"></div><p class="note">Drag the range slider to zoom. Daily totals include only retained product transactions; an absent source day is not verified as zero demand.</p></div><div class="panel"><div id="products" class="chart"></div><p class="note">Totals reflect full-period recorded gross units, including unusually large orders.</p></div><div class="panel"><div id="week" class="chart"></div><p class="note">Mean over dates present in the cleaned panel, not proof of a weekly causal effect. Missing calendar dates are not silently counted as zero.</p></div><div class="panel full"><div id="zeros" class="chart"></div><p class="note">Zeros are inserted between each product’s first and last recorded sale. They may reflect no sale, stockout, unavailability, or incomplete records.</p></div></section>
    <section class="insights"><div class="panel"><div class="tag">What makes the data useful</div><h2>A rich forecasting test bed</h2><ul><li>__RAW__ transaction lines span two years and thousands of products, with real variation across days and items.</li><li>The top 10 products account for <b>__TOP__%</b> of retained units, exposing concentration that aggregate metrics can conceal.</li><li>Daily product histories enable rolling backtests and separate analysis of intermittent sales.</li></ul></div><div class="panel"><div class="tag warn">What needs care</div><h2>Sales are not true demand</h2><ul><li>No stock, supplier lead times or lost sales are observed. A zero does not prove there was no demand.</li><li>Old data, wholesale transactions and exceptional orders limit transfer to current retail. The two largest orders have same-SKU, equal negative cancellations minutes later; gross totals retain the positive entries.</li><li>The current product-code filter is a heuristic; the first/last-sale grid and full-period descriptive analysis can introduce selection bias.</li><li>This exploration uses the entire historical dataset. Freeze chronological validation and test periods before tuning, and do not select models from these charts.</li></ul></div></section>
    <section class="panel"><div class="tag">Explore the data</div><h2>Top 200 products</h2><p>Search a stock code and click a column heading to sort. Days are measured only within each product’s observed first-to-last sale span.</p><input id="search" type="search" placeholder="Search stock code…" aria-label="Search stock code"><div class="tablewrap"><table><thead><tr><th data-key="sku">Stock code ↕</th><th data-key="units">Gross units ↕</th><th data-key="days">Span days ↕</th><th data-key="sale_days">Sale days ↕</th><th data-key="zero">Zero days % ↕</th><th data-key="max">Largest day ↕</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note" id="count"></p></section>
    <section class="panel" style="margin-top:16px"><div class="tag">Feature engineering / as of 30 April 2011</div><h2>Product descriptions &amp; candidate signals</h2><p><a href="code-family-audit.html">Explore the stock-code family and case audit →</a></p><p>Search the product catalogue by code or description. Codes were gathered retrospectively; descriptions and historical features are limited to the cutoff. Tags are literal words; possible family bases and pack sizes require review. None establish product quality or origin. Price is the historical quantity-weighted mean, not a future price.</p><input id="catalog-search" type="search" placeholder="Search a code or description…" aria-label="Search product feature catalogue"><div class="tablewrap"><table><thead><tr><th>Stock code</th><th>Description</th><th>Code format</th><th>Possible family base</th><th>Possible pack size</th><th>Historic price (GBP)</th><th>Text tags</th></tr></thead><tbody id="catalog-rows"></tbody></table></div><p class="note" id="catalog-count"></p></section>
    <section class="panel" style="margin-top:16px"><div class="tag">Cleaning audit</div><h2>From transactions to the panel</h2><p>__INCLUDED__ included · __RETURNS__ returns/cancellations · __NONPRODUCT__ service/unclassified codes · __NONPOSITIVE__ nonpositive price/quantity. Categories follow pipeline precedence; the audit counts rows, not units.</p></section>
    <footer>Source: Chen (2012), <a href="https://doi.org/10.24432/C5CG6D">Online Retail II, UCI</a>, CC BY 4.0. This descriptive report is not a model evaluation or business savings claim. Reproduce using the accompanying notebook and project data.</footer></main><script>
    const charts=__CHARTS__;Object.entries(charts).forEach(([id,raw])=>{const p=JSON.parse(raw);Plotly.newPlot(id,p.data,p.layout,{responsive:true,displaylogo:false});});
    const catalog=__CATALOG__;function renderCatalog(){const q=document.getElementById('catalog-search').value.toLowerCase();const filtered=catalog.filter(x=>(x.sku+' '+x.description).toLowerCase().includes(q));const body=document.getElementById('catalog-rows');body.replaceChildren();filtered.forEach(x=>{const tr=document.createElement('tr');[x.sku,x.description,x.shape,x.base||'—',x.pack||'—',x.price?Number(x.price).toFixed(2):'—',x.tags||'—'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});body.appendChild(tr)});document.getElementById('catalog-count').textContent=`Showing ${filtered.length} of ${catalog.length} catalogue entries.`}document.getElementById('catalog-search').addEventListener('input',renderCatalog);renderCatalog();
    const data=__ROWS__;let key='units',descending=true;function render(){let q=document.getElementById('search').value.toLowerCase();let items=data.filter(x=>x.sku.toLowerCase().includes(q)).sort((a,b)=>{let av=a[key],bv=b[key];return (typeof av==='string'?av.localeCompare(bv):av-bv)*(descending?-1:1)});let body=document.getElementById('rows');body.replaceChildren();items.forEach(x=>{let tr=document.createElement('tr');[x.sku,x.units.toLocaleString(),x.days.toLocaleString(),x.sale_days.toLocaleString(),x.zero.toFixed(1)+'%',x.max.toLocaleString()].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});body.appendChild(tr)});document.getElementById('count').textContent=`Showing ${items.length} of ${data.length} products in this table.`}document.getElementById('search').addEventListener('input',render);document.querySelectorAll('th[data-key]').forEach(th=>th.addEventListener('click',()=>{descending=key===th.dataset.key?!descending:true;key=th.dataset.key;render()}));render();</script></body></html>'''
    replacements = {'__PLOTLY__':get_plotlyjs(),'__RAW__':f'{raw:,}','__SKUS__':f'{len(product):,}','__UNITS__':f'{total:,.0f}','__ZERO__':f'{zero:.1f}','__TOP__':f'{top10:.1f}','__INCLUDED__':f"{audit.get('included_rows',0):,}",'__RETURNS__':f"{audit.get('returns_or_cancellations',0):,}",'__NONPRODUCT__':f"{audit.get('unclassified_or_service_codes',0):,}",'__NONPOSITIVE__':f"{audit.get('nonpositive_quantity_or_price',0):,}",'__CHARTS__':charts_json,'__ROWS__':records_json,'__CATALOG__':catalog_json}
    for key,value in replacements.items(): body=body.replace(key,value)
    path=ROOT/'reports/data-exploration.html';path.parent.mkdir(exist_ok=True);path.write_text(body,encoding='utf-8')
    return path


if __name__ == '__main__':
    args=analyze()
    print(create_dashboard(*args))
