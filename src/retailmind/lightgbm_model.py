"""Global one-step LightGBM baseline with recursive multi-step inference."""
from datetime import timedelta
from .features import calendar_features, demand_features


def features(history, target_date, *, calendar=False, metadata_values=None):
    result = demand_features(history)
    if calendar:
        result.extend(calendar_features(target_date))
    if metadata_values is not None:
        result.extend(metadata_values)
    return result



def train(panel, cutoff, *, min_history=28, max_windows_per_product=180, seed=42, calendar=False, metadata=None):
    """Fit one pooled model using only observations on or before cutoff.

    Windows with calendar gaps are skipped. Product IDs are not numeric features.
    """
    import lightgbm as lgb
    import numpy as np
    if min_history < 28 or (max_windows_per_product is not None and max_windows_per_product < 1):
        raise ValueError('Invalid training window parameters')
    x, y, vx, vy = [], [], [], []
    validation_start = cutoff - timedelta(days=27)
    for code, series in panel.items():
        dates = sorted(day for day in series if day <= cutoff)
        targets = dates[-max_windows_per_product:] if max_windows_per_product is not None else dates
        for target in targets:
            required = [target - timedelta(days=i) for i in range(min_history, 0, -1)]
            if all(day in series for day in required):
                vector = features([series[day] for day in required], target, calendar=calendar, metadata_values=metadata.at(code, target - timedelta(days=1)) if metadata else None)
                if target >= validation_start:
                    vx.append(vector); vy.append(series[target])
                else:
                    x.append(vector); y.append(series[target])
    if not x or not vx:
        raise ValueError('No complete training and early-stopping windows available')
    dataset = lgb.Dataset(np.asarray(x), label=np.asarray(y))
    validation = lgb.Dataset(np.asarray(vx), label=np.asarray(vy), reference=dataset)
    return lgb.train({'objective': 'regression', 'learning_rate': 0.05,
                      'num_leaves': 15, 'seed': seed, 'num_threads': 1,
                      'verbosity': -1}, dataset, num_boost_round=100,
                     valid_sets=[validation], callbacks=[lgb.early_stopping(10, verbose=False)])


def predict(model, history, cutoff, horizon=28, *, calendar=False, metadata_values=None):
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    import numpy as np
    values = list(history)
    predictions = []
    for step in range(1, horizon + 1):
        day = cutoff + timedelta(days=step)
        value = max(0.0, float(model.predict(np.asarray([features(values, day, calendar=calendar, metadata_values=metadata_values)]), num_threads=1)[0]))
        predictions.append(value)
        values.append(value)
    return predictions


def train_split(panel, train_end, validation_start, validation_end, *, calendar=False,
                validation_days=28, seed=42):
    """Select boosting rounds on a contiguous validation block, then refit through it.

    Only training labels before validation_start enter the selection fit. The
    returned model is refit through validation_end for a later untouched test.
    """
    from datetime import timedelta
    import lightgbm as lgb
    import numpy as np
    x, y, vx, vy = [], [], [], []
    for series in panel.values():
        for target, label in series.items():
            if target > validation_end:
                continue
            required = [target - timedelta(days=i) for i in range(28, 0, -1)]
            if not all(day in series for day in required):
                continue
            row = features([series[d] for d in required], target, calendar=calendar)
            if validation_start <= target <= validation_end:
                vx.append(row); vy.append(label)
            elif target <= train_end:
                x.append(row); y.append(label)
    if not x or not vx:
        raise ValueError('Insufficient chronological LightGBM train/validation rows')
    params = {'objective':'regression','learning_rate':0.05,'num_leaves':15,
              'seed':seed,'num_threads':1,'verbosity':-1}
    base = lgb.Dataset(np.asarray(x), label=np.asarray(y))
    valid = lgb.Dataset(np.asarray(vx), label=np.asarray(vy), reference=base)
    selector = lgb.train(params, base, num_boost_round=100, valid_sets=[valid],
                         callbacks=[lgb.early_stopping(10, verbose=False)])
    rounds = max(1, int(selector.best_iteration))
    refit_x, refit_y = x + vx, y + vy
    final = lgb.train(params, lgb.Dataset(np.asarray(refit_x), label=np.asarray(refit_y)),
                      num_boost_round=rounds)
    return final, {'best_iteration':rounds,'selection_train_rows':len(x),
                   'validation_rows':len(vx),'refit_rows':len(refit_y)}
