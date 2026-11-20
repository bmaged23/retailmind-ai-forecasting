"""Build per-product daily positive-sales prices across the full workbook."""
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
import csv
import math
from retailmind.workbook import iter_rows

ROOT=Path(__file__).resolve().parents[1]
CUTOFF=date(2011,12,9)


def main():
    with (ROOT/'data/processed/daily_sales.csv').open(newline='',encoding='utf-8') as f:
        products={row['stock_code'] for row in csv.DictReader(f)}
    grouped=defaultdict(lambda:[0.,0.]);accepted=0
    for row in iter_rows(ROOT/'data/raw/online_retail_II.xlsx'):
        code=str(row.get('StockCode') or '').strip();when=row.get('InvoiceDate')
        invoice=str(row.get('Invoice',row.get('InvoiceNo','')) or '').strip()
        if code not in products or not invoice or invoice.upper().startswith('C') or not isinstance(when,(date,datetime)):continue
        day=when.date() if isinstance(when,datetime) else when
        if day>CUTOFF:continue
        try:q=float(row.get('Quantity'));p=float(row.get('Price',row.get('UnitPrice')))
        except (TypeError,ValueError):continue
        if not math.isfinite(q) or not math.isfinite(p) or q<=0 or p<=0:continue
        pair=grouped[(code,day)];pair[0]+=q;pair[1]+=q*p;accepted+=1
    path=ROOT/'reports/generated/daily-price-all.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['stock_code','date','units','revenue','invoice_count','uk_units','known_country_units'])
        for (code,day),(units,revenue) in sorted(grouped.items()):writer.writerow([code,day.isoformat(),units,revenue,0,0,0])
    print(f'Wrote {len(grouped)} product-days from {accepted} rows across {len(products)} panel codes to {path}',flush=True)

if __name__=='__main__':main()
