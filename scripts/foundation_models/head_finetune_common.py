"""Shared training and validation windows for experimental head adaptation."""
from datetime import date, timedelta

import numpy as np

from common import ROOT, RUN_VALIDATION, ORIGIN, load_panel_through

TRAIN_END = date(2011, 4, 30) if RUN_VALIDATION else date(2011, 8, 31)
VALID_END = date(2011, 5, 28) if RUN_VALIDATION else date(2011, 9, 28)
CONTEXT = 90
HORIZON = 28
MAX_EPOCHS = 6
PATIENCE = 2
VALIDATION_LIMIT = 256
SEED = 42


def dense(series, end):
    values = {day: quantity for day, quantity in series.items() if day <= end}
    if not values:
        return None
    first = min(values)
    return np.asarray(
        [values.get(first + timedelta(days=i), 0.0) for i in range((end - first).days + 1)],
        dtype=np.float32,
    )


def load_training_windows():
    panel = load_panel_through(ROOT / "data/processed/daily_sales.csv", ORIGIN)
    train_series, validation = [], []
    for series in panel.values():
        train = dense(series, TRAIN_END)
        valid = dense(series, VALID_END)
        if train is not None and len(train) >= CONTEXT + HORIZON:
            train_series.append(train)
        if valid is not None and len(valid) >= CONTEXT + HORIZON:
            validation.append((valid[-CONTEXT-HORIZON:-HORIZON], valid[-HORIZON:]))
    rng = np.random.default_rng(SEED)
    if len(validation) > VALIDATION_LIMIT:
        chosen = np.sort(rng.choice(len(validation), VALIDATION_LIMIT, replace=False))
        validation = [validation[int(i)] for i in chosen]
    return train_series, validation, rng


def random_training_window(series, rng):
    max_start = len(series) - CONTEXT - HORIZON
    start = int(rng.integers(max_start + 1)) if max_start > 0 else 0
    return series[start:start + CONTEXT], series[start + CONTEXT:start + CONTEXT + HORIZON]
