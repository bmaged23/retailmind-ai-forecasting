import csv, collections, html, math
from pathlib import Path
base=Path('artifacts/validation_comparison_2011-07-01_to_2011-07-28/forecasts'); dst=Path('docs/images'); dst.mkdir(exist_ok=True)
models=[('Timestral LoRA','timestral209m_lora','#38bdf8'),('Chronos-2 LoRA','chronos2_lora','#a78bfa'),('LSTM','lstm','#fb923c'),('TimesFM 3 tuned','timesfm3_head_tuned','#34d399')]
series={}; codes=set()
for name,file,color in models:
 d=collections.defaultdict(float)
 for r in csv.DictReader(open(base/(file+'.csv'))): d[r['target_date']]+=float(r['prediction']); codes.add(r['stock_code'])
 series[name]=d
actual=collections.defaultdict(float)
for r in csv.DictReader(open('data/processed/daily_sales.csv')):
 if r['stock_code'] in codes and '2011-07-01'<=r['date']<='2011-07-28': actual[r['date']]+=float(r['quantity'])
dates=sorted(series[models[0][0]]); allvals=list(actual.values())+[v for d in series.values() for v in d.values()]; maxy=max(allvals)*1.1

def chart(title,sub,dates,sets,maxy,filename,band=None):
 W,H=1100,450; L,R,T,B=80,35,100,70; pw=W-L-R; ph=H-T-B
 x=lambda i:L+i*pw/(len(dates)-1); y=lambda v:T+ph-(v/maxy)*ph
 a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" rx="22" fill="#0f172a"/><text x="35" y="43" fill="#f8fafc" font-family="Arial" font-weight="bold" font-size="24">{html.escape(title)}</text><text x="35" y="72" fill="#94a3b8" font-family="Arial" font-size="14">{html.escape(sub)}</text>']
 for i in range(5):
  v=maxy*i/4; yy=y(v); a.append(f'<line x1="{L}" y1="{yy}" x2="{W-R}" y2="{yy}" stroke="#29374b"/><text x="{L-10}" y="{yy+5}" text-anchor="end" fill="#94a3b8" font-family="Arial" font-size="12">{v:.0f}</text>')
 if band:
  low,high=band; pts=[f'{x(i):.1f},{y(v):.1f}' for i,v in enumerate(high)]+[f'{x(i):.1f},{y(v):.1f}' for i,v in reversed(list(enumerate(low)))]; a.append(f'<polygon points="{" ".join(pts)}" fill="#38bdf8" opacity=".18"/>')
 for j,(name,vals,color) in enumerate(sets):
  pts=' '.join(f'{x(i):.1f},{y(v):.1f}' for i,v in enumerate(vals));a.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{3 if j==0 else 2}" stroke-linejoin="round"/><circle cx="{L+j*190}" cy="{H-25}" r="5" fill="{color}"/><text x="{L+10+j*190}" y="{H-20}" fill="#cbd5e1" font-family="Arial" font-size="12">{html.escape(name)}</text>')
 for i in (0,7,14,21,27):a.append(f'<text x="{x(i):.1f}" y="{H-B+22}" text-anchor="middle" fill="#94a3b8" font-family="Arial" font-size="12">{dates[i][5:]}</text>')
 a.append('</svg>'); (dst/filename).write_text(''.join(a))
chart('Daily portfolio sales: actual vs models','Common 4,366-product cohort • July 2011 selection window • units/day',dates,[('Actual',[actual[d] for d in dates],'#f8fafc')]+[(n,[series[n][d] for d in dates],c) for n,_,c in models],maxy,'forecast-actual.svg')
rows=[r for r in csv.DictReader(open('artifacts/milestone2_uncertainty/timestral209m_lora/evaluation-intervals.csv')) if r['stock_code']=='21976']; rows.sort(key=lambda r:r['target_date']); dates=[r['target_date'] for r in rows]; actuals=[float(r['actual']) for r in rows]; mid=[float(r['p50']) for r in rows]; lo=[float(r['calibrated_lower']) for r in rows]; hi=[float(r['calibrated_upper']) for r in rows]
chart('Forecast intervals for product 21976','Illustrative SKU • 12 Nov – 9 Dec 2011 • shaded calibrated daily P10–P90 bounds',dates,[('Actual',actuals,'#f8fafc'),('Timestral P50',mid,'#38bdf8')],max(max(actuals),max(hi))*1.1,'forecast-interval.svg',(lo,hi))
print('portfolio actual total',sum(actual.values()),'series',len(codes),'interval rows',len(rows))
