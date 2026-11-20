import argparse
import csv
from datetime import date
from pathlib import Path
from .data import preprocess
from .forecast import backtest, load_panel


def main():
    parser = argparse.ArgumentParser(prog='retailmind')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('workbook')
    prep.add_argument('--output', default='data/processed')
    evaluate = commands.add_parser('backtest')
    evaluate.add_argument('panel')
    evaluate.add_argument('--cutoff', action='append', required=True, type=date.fromisoformat)
    evaluate.add_argument('--output', default='reports/generated/seasonal_naive.csv')
    evaluate.add_argument('--context', type=int, default=90)
    evaluate.add_argument('--horizon', type=int, default=28)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(dict(preprocess(args.workbook, args.output)))
    else:
        rows = backtest(load_panel(args.panel), args.cutoff, args.context, args.horizon)
        if not rows:
            parser.error('No products have complete history and targets for every cutoff')
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows)} product/fold results to {output}')
