"""Prepare an exploratory feature table for validation origins only."""
from datetime import date, timedelta
from pathlib import Path
import csv
import json
from retailmind.features import calendar_features
from retailmind.transaction_features import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'reports/generated'
    config = json.loads((ROOT/'configs/benchmark.json').read_text())
    store = load(folder/'daily-transaction-features.csv')
    rows = []
    for group, filename in [('active','cohort.csv'),('intermittent','cohort-intermittent.csv')]:
        with (folder/filename).open(newline='',encoding='utf-8') as handle:
            products = [r['stock_code'] for r in csv.DictReader(handle)]
        for cutoff in map(date.fromisoformat, config['validation_cutoffs']):
            for product in products:
                price, invoices, uk, price_missing, uk_missing = store.at(product,cutoff)
                holiday = calendar_features(cutoff + timedelta(days=1))[5]
                rows.append({'cohort':group,'stock_code':product,'cutoff':cutoff.isoformat(),
                             'log_price':price,'log_invoice_days':invoices,'uk_share':uk,
                             'price_missing':int(price_missing),'uk_missing':int(uk_missing),'next_day_bank_holiday':int(holiday)})
    path = folder/'feature-eda.csv'
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Wrote',len(rows),'cohort/cutoff rows')

if __name__=='__main__': main()
