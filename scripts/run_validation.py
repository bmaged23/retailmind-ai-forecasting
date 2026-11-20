"""Run the proposed validation folds; test and calibration folds remain untouched."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import lightgbm
from retailmind.transaction_features import load as load_transaction_features
from retailmind.benchmark import evaluate_fold
from retailmind.forecast import load_panel
from datetime import date

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError("No rows or field names supplied")
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm', action='store_true', help='Include the optional CPU LSTM')
    parser.add_argument('--cohort', choices=['active', 'intermittent'], default='active')
    parser.add_argument('--calendar', action='store_true', help='Add known-calendar LightGBM')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--enriched', action='store_true', help='Add time-safe price, invoice activity, geography and calendar features')
    args = parser.parse_args()
    config_path = ROOT / 'configs/benchmark.json'
    panel_path = ROOT / 'data/processed/daily_sales.csv'
    cohort_path = ROOT / ('reports/generated/cohort.csv' if args.cohort == 'active' else 'reports/generated/cohort-intermittent.csv')
    config = json.loads(config_path.read_text())
    with cohort_path.open(newline='', encoding='utf-8') as handle:
        cohort = [row['stock_code'] for row in csv.DictReader(handle)]
    if len(cohort) != (config['maximum_products'] if args.cohort == 'active' else config['intermittent_cohort']['maximum_products']) or len(cohort) != len(set(cohort)):
        raise ValueError('Cohort is missing, duplicated or incomplete; run scripts/validate_data.py')
    panel = load_panel(panel_path, products=set(cohort))
    gpu = None
    if args.device == 'cuda':
        if not args.lstm: parser.error('CUDA is only used by the LSTM; include --lstm')
        import torch
        from retailmind.gpu_monitor import GPUMonitor
        gpu = GPUMonitor(max_used_mib=1900).start()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; GPU run aborted')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 1024 * 1024 * 1024 / total), 0)
        torch.backends.cudnn.benchmark = False
        gpu.check()
    metadata_path = ROOT / 'reports/generated/daily-transaction-features.csv'
    feature_store = load_transaction_features(metadata_path) if args.enriched else None
    rows, predictions, exclusions, folds = [], [], [], []
    for text in config['validation_cutoffs']:
        cutoff = date.fromisoformat(text)
        metrics, forecasts, missing, timing = evaluate_fold(panel, cohort, cutoff, config['context_days'], config['horizon_days'], include_lstm=args.lstm, lstm_config=config.get('lstm'), include_calendar=args.calendar, metadata=feature_store, device=args.device, gpu_monitor=gpu, batch_size=args.batch_size)
        rows.extend(metrics); predictions.extend(forecasts)
        exclusions.extend({'cutoff': text, **x} for x in missing)
        folds.append({'cutoff': text, **timing, 'excluded': len(missing)})
    gpu_stats = gpu.finish() if gpu else None
    result_folder = 'validation-enriched-gpu' if args.enriched and args.device == 'cuda' else 'validation-enriched' if args.enriched else 'validation'
    out = ROOT / 'reports/generated' / result_folder / args.cohort; out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'metrics.csv', rows); write_csv(out / 'forecasts.csv', predictions)
    write_csv(out / 'exclusions.csv', exclusions, ['cutoff', 'stock_code', 'reason'])
    metadata = {'phase': 'validation', 'cohort': args.cohort, 'enriched': args.enriched, 'transaction_features_sha256': sha256(metadata_path) if args.enriched else None, 'calendar_included': args.calendar, 'lstm_included': args.lstm, 'config_sha256': sha256(config_path), 'panel_sha256': sha256(panel_path),
                'cohort_sha256': sha256(cohort_path), 'python': sys.version, 'platform': platform.platform(),
                'lightgbm_version': lightgbm.__version__, 'device': args.device, 'gpu_stats': gpu_stats, 'batch_size_override': args.batch_size, 'seed': 42, 'folds': folds,
                'source_sha256': {name: sha256(ROOT / 'src/retailmind' / name) for name in ['benchmark.py', 'forecast.py', 'lightgbm_model.py', 'lstm_model.py', 'features.py', 'transaction_features.py', 'gpu_monitor.py']},
                'caveats': ['exploratory; not final test results', 'single-process warm model per-SKU timing; CPU peak RAM not measured; GPU peak is sampled and may miss brief spikes',
                            'full dataset previously inspected in dashboard', 'post-last-sale target days treated as missing']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__': main()
