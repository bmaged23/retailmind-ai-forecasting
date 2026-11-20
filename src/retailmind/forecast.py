"""A seasonal-naive control and calendar-based rolling evaluation."""
from datetime import date, timedelta
import csv
import math
from time import perf_counter


def seasonal_naive(history, horizon=28, season=7):
    if horizon < 1 or season < 1 or len(history) < season:
        raise ValueError('Insufficient history or invalid horizon/season')
    return [history[-season + (i % season)] for i in range(horizon)]


def metrics(actual, predicted, history, season=7):
    if not actual or len(actual) != len(predicted):
        raise ValueError('Actual and forecast must be nonempty and equal length')
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    scale = sum(abs(history[i] - history[i-season]) for i in range(season, len(history))) / (len(history)-season) if len(history) > season else 0
    total = sum(abs(a) for a in actual)
    return {'mae': sum(errors)/len(errors), 'rmse': math.sqrt(sum(e*e for e in errors)/len(errors)), 'wape': sum(errors)/total if total else None, 'mase': sum(errors)/len(errors)/scale if scale else None}


def load_panel(path, products=None):
    result = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if products is not None and row['stock_code'] not in products:
                continue
            result.setdefault(row['stock_code'], {})[date.fromisoformat(row['date'])] = float(row['quantity'])
    return result


def backtest(panel, cutoffs, context=90, horizon=28):
    if context < 7 or horizon < 1:
        raise ValueError('Context must be at least 7 and horizon positive')
    results = []
    # Identical fixed cohort across all requested cutoffs.
    eligible = []
    for code, series in sorted(panel.items()):
        needed = (cutoff + timedelta(days=i) for cutoff in cutoffs for i in range(1-context, horizon+1))
        if all(day in series for day in needed):
            eligible.append(code)
    for cutoff in cutoffs:
        for code in eligible:
            series = panel[code]
            history = [series[cutoff + timedelta(days=i)] for i in range(1-context, 1)]
            actual = [series[cutoff + timedelta(days=i)] for i in range(1, horizon+1)]
            started = perf_counter()
            prediction = seasonal_naive(history, horizon)
            latency = perf_counter() - started
            results.append({'stock_code': code, 'cutoff': cutoff.isoformat(), 'model': 'seasonal_naive', 'horizon': horizon, 'inference_seconds': latency, **metrics(actual, prediction, history)})
    return results
