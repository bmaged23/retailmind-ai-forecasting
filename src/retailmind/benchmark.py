"""Common, chronological validation harness for point forecasts."""
from datetime import date, timedelta
from time import perf_counter
from .forecast import metrics, seasonal_naive
from . import lightgbm_model


def historical_panel(panel, cohort, cutoff):
    """Build history as known at cutoff; after first sale, absent dates mean no recorded sales.

    This assumption says nothing about availability or underlying demand. No future
    panel rows are consulted, including to infer an item's last active day.
    """
    result = {}
    for code in cohort:
        series = panel.get(code, {})
        positive = [day for day, value in series.items() if day <= cutoff and value > 0]
        if not positive:
            continue
        first = min(positive)
        result[code] = {first + timedelta(days=i): series.get(first + timedelta(days=i), 0.0)
                        for i in range((cutoff - first).days + 1)}
    return result


def evaluate_fold(panel, cohort, cutoff, context=90, horizon=28, *, include_lstm=False, lstm_config=None, include_calendar=False, metadata=None, device="cpu", gpu_monitor=None, batch_size=None):
    if context < 28 or horizon < 1 or len(cohort) != len(set(cohort)):
        raise ValueError('Invalid context, horizon or duplicate product in cohort')
    historical = historical_panel(panel, cohort, cutoff)
    eligible, missing = [], []
    for code in cohort:
        series = historical.get(code, {})
        history_days = [cutoff - timedelta(days=i) for i in range(context-1, -1, -1)]
        target_days = [cutoff + timedelta(days=i) for i in range(1, horizon+1)]
        if not all(day in series for day in history_days):
            missing.append({'stock_code': code, 'reason': 'insufficient_history'})
        elif not all(day in panel.get(code, {}) for day in target_days):
            missing.append({'stock_code': code, 'reason': 'missing_target'})
        else:
            eligible.append((code, [series[d] for d in history_days], [panel[code][d] for d in target_days]))
    if not eligible:
        raise ValueError('No cohort members have complete history and targets')
    start = perf_counter()
    model = lightgbm_model.train(historical, cutoff)
    train_seconds = perf_counter() - start
    calendar_model = None
    calendar_train_seconds = None
    if include_calendar:
        start = perf_counter()
        calendar_model = lightgbm_model.train(historical, cutoff, calendar=True)
        calendar_train_seconds = perf_counter() - start
    enriched_lightgbm = None
    enriched_lightgbm_seconds = None
    if metadata is not None:
        start = perf_counter()
        enriched_lightgbm = lightgbm_model.train(historical, cutoff, calendar=True, metadata=metadata)
        enriched_lightgbm_seconds = perf_counter() - start
    neural = None
    enriched_neural = None
    enriched_neural_seconds = None
    neural_train_seconds = None
    neural_gpu_stats = None
    enriched_gpu_stats = None
    if include_lstm:
        from . import lstm_model
        start = perf_counter()
        options = lstm_config or {}
        if device not in {'cpu', 'cuda'} or options.get('layers', 1) != 1 or options.get('dropout', 0) != 0:
            raise ValueError('Unsupported LSTM configuration or unsafe GPU setting')
        if options.get('context_days', context) != context:
            raise ValueError('LSTM context must match benchmark context')
        settings = dict(context=context, device=device, monitor=gpu_monitor, hidden_units=options.get('hidden_units', 32),
                        learning_rate=options.get('learning_rate', 0.001), batch_size=batch_size or options.get('batch_size', 64),
                        epochs=options.get('max_epochs', 20), patience=options.get('early_stopping_patience', 3),
                        validation_days=options.get('early_stopping_days', 28), weight_decay=options.get('weight_decay', 0),
                        seed=options.get('seed', 42))
        neural = lstm_model.train(historical, cutoff, **settings)
        neural_train_seconds = perf_counter() - start
        if gpu_monitor: neural_gpu_stats = gpu_monitor.summarize_since(start)
        if metadata is not None:
            start = perf_counter()
            enriched_neural = lstm_model.train(historical, cutoff, metadata=metadata, **settings)
            enriched_neural_seconds = perf_counter() - start
            if gpu_monitor: enriched_gpu_stats = gpu_monitor.summarize_since(start)
    rows, forecasts = [], []
    for code, history, actual in eligible:
        models = [('seasonal_naive', lambda: seasonal_naive(history, horizon)),
                  ('lightgbm', lambda: lightgbm_model.predict(model, history, cutoff, horizon))]
        if calendar_model is not None:
            models.append(('lightgbm_calendar', lambda: lightgbm_model.predict(calendar_model, history, cutoff, horizon, calendar=True)))
        if neural is not None:
            models.append(('lstm', lambda: lstm_model.predict(neural, history, horizon, context=context, device=device, monitor=gpu_monitor)))
        if enriched_lightgbm is not None:
            values = metadata.at(code, cutoff)
            models.append(('lightgbm_enriched', lambda: lightgbm_model.predict(enriched_lightgbm, history, cutoff, horizon, calendar=True, metadata_values=values)))
        if enriched_neural is not None:
            models.append(('lstm_enriched', lambda: lstm_model.predict(enriched_neural, history, horizon, context=context, device=device, cutoff=cutoff, metadata_values=values, monitor=gpu_monitor)))
        for name, fn in models:
            start = perf_counter()
            prediction = fn()
            latency = perf_counter() - start
            if len(prediction) != horizon:
                raise ValueError(f'{name} returned the wrong forecast horizon')
            rows.append({'model': name, 'stock_code': code, 'cutoff': cutoff.isoformat(), 'inference_seconds': latency,
                         **metrics(actual, prediction, history)})
            for step, (observed, predicted) in enumerate(zip(actual, prediction), 1):
                forecasts.append({'model': name, 'stock_code': code, 'cutoff': cutoff.isoformat(),
                                  'target_date': (cutoff + timedelta(days=step)).isoformat(), 'horizon': step,
                                  'actual': observed, 'prediction': predicted})
    return rows, forecasts, missing, {'lightgbm_train_seconds': train_seconds, 'lightgbm_enriched_train_seconds': enriched_lightgbm_seconds, 'lstm_enriched_train_seconds': enriched_neural_seconds, 'lstm_gpu_training_samples': neural_gpu_stats, 'lstm_enriched_gpu_training_samples': enriched_gpu_stats, 'lstm_enriched_best_epoch': enriched_neural.best_epoch if enriched_neural else None, 'lstm_enriched_epochs_run': enriched_neural.epochs_run if enriched_neural else None, 'lightgbm_calendar_train_seconds': calendar_train_seconds, 'lstm_train_seconds': neural_train_seconds, 'lstm_best_epoch': neural.best_epoch if neural else None, 'lstm_epochs_run': neural.epochs_run if neural else None, 'lstm_inner_val_loss': neural.best_validation_loss if neural else None, 'scored_products': len(eligible), 'cohort_size': len(cohort)}
