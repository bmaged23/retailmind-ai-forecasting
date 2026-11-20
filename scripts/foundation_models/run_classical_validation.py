"""Run seasonal-naive, LightGBM, and LSTM on the frozen July selection cohort."""
from datetime import date, timedelta
from pathlib import Path
import sys
from time import perf_counter
import json

import numpy as np
import torch
from torch import nn

from common import (
    CONTEXT, HORIZON, ORIGIN, ROOT, RUN_VALIDATION, load_contexts,
    load_panel_through, save_run_metadata, start_gpu_monitor, write_forecasts,
)
from retailmind.forecast import seasonal_naive
from retailmind.lightgbm_model import train_split, predict as predict_lgbm
from retailmind.lstm_ablation import build_network, predict as predict_lstm

CONFIG = json.loads((ROOT / "configs/validation_comparison.json").read_text())
FIT_END = date.fromisoformat(CONFIG["tuning"]["fit_targets_through"])
VALID_START = date.fromisoformat(CONFIG["tuning"]["checkpoint_validation_start"])
VALID_END = date.fromisoformat(CONFIG["tuning"]["checkpoint_validation_end"])
BATCH_SIZE = 512
WORKERS = 4
MAX_EPOCHS = 20
PATIENCE = 3
MAX_WINDOWS_PER_PRODUCT = 180


def windows(panel, start, end):
    """Prepare normalized 90-day single-step windows without future rows."""
    features, labels = [], []
    for series in panel.values():
        targets = [day for day in series if start <= day <= end]
        if len(targets) > MAX_WINDOWS_PER_PRODUCT:
            targets = targets[-MAX_WINDOWS_PER_PRODUCT:]
        for target in targets:
            dates = [target - timedelta(days=i) for i in range(CONTEXT, 0, -1)]
            if not all(day in series for day in dates):
                continue
            values = np.asarray([series[day] for day in dates], dtype=np.float32)
            scale = max(float(values.mean()), 1.0)
            features.append(np.log1p(values / scale))
            labels.append(np.log1p(float(series[target]) / scale))
    if not features:
        raise ValueError(f"No LSTM windows for {start} through {end}")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def dataset(x, y):
    return torch.utils.data.TensorDataset(
        torch.from_numpy(x).unsqueeze(-1),
        torch.zeros((len(y), 0), dtype=torch.float32),
        torch.from_numpy(y),
    )


def loader(data, shuffle, seed=42):
    return torch.utils.data.DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=WORKERS, pin_memory=True, persistent_workers=False,
        prefetch_factor=2,
    )


def train_lstm(train_x, train_y, valid_x, valid_y, monitor):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    train_data, valid_data = dataset(train_x, train_y), dataset(valid_x, valid_y)
    train_loader, valid_loader = loader(train_data, True), loader(valid_data, False)
    model = build_network(0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    epoch_times = []
    selection_start = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = perf_counter()
        model.train()
        for seq, extra, label in train_loader:
            monitor.check()
            seq = seq.cuda(non_blocking=True)
            extra = extra.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(seq, extra), label)
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for seq, extra, label in valid_loader:
                monitor.check()
                pred = model(seq.cuda(non_blocking=True), extra.cuda(non_blocking=True))
                total += nn.functional.mse_loss(
                    pred, label.cuda(non_blocking=True), reduction="sum"
                ).item()
        score = total / len(valid_data)
        epoch_times.append(perf_counter() - epoch_start)
        print(f"LSTM selection epoch={epoch} val_mse={score:.6f}", flush=True)
        if score < best_loss - 1e-6:
            best_loss = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    selection_seconds = perf_counter() - selection_start
    del train_loader, valid_loader, optimizer, model
    torch.cuda.empty_cache()

    # Refit from a fresh initialization through the tuning-validation window,
    # using only the epoch count chosen above. July remains untouched.
    torch.manual_seed(42)
    combined = dataset(
        np.concatenate((train_x, valid_x)),
        np.concatenate((train_y, valid_y)),
    )
    combined_loader = loader(combined, True)
    model = build_network(0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    refit_start = perf_counter()
    for epoch in range(best_epoch):
        model.train()
        for seq, extra, label in combined_loader:
            monitor.check()
            seq = seq.cuda(non_blocking=True)
            extra = extra.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(seq, extra), label)
            loss.backward()
            optimizer.step()
        print(f"LSTM refit epoch={epoch + 1}/{best_epoch}", flush=True)
    refit_seconds = perf_counter() - refit_start
    model.eval()
    del combined_loader, optimizer, train_data, valid_data, combined
    torch.cuda.empty_cache()
    return model, {
        "best_epoch": best_epoch,
        "epochs_run_for_selection": len(epoch_times),
        "best_validation_mse": best_loss,
        "train_windows": len(train_y),
        "validation_windows": len(valid_y),
        "refit_windows": len(train_y) + len(valid_y),
        "batch_size": BATCH_SIZE,
        "data_loader_workers": WORKERS,
        "learning_rate": 0.001,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "epoch_seconds": epoch_times,
        "selection_seconds": selection_seconds,
        "refit_seconds": refit_seconds,
    }


def main():
    if not RUN_VALIDATION:
        raise RuntimeError("Set RETAILMIND_RUN_SPLIT=validation")
    _, codes, contexts = load_contexts()
    # Training panel is cut at the checkpoint-selection window end; July rows
    # are neither parsed nor passed to any fitting code.
    panel = load_panel_through(ROOT / "data/processed/daily_sales.csv", VALID_END)
    lstm_only = "--lstm-only" in sys.argv
    if lstm_only:
        forecast_dir = ROOT / "artifacts/validation_comparison_2011-07-01_to_2011-07-28/forecasts"
        for candidate in ("seasonal_naive", "lightgbm"):
            if not (forecast_dir / f"{candidate}.csv").exists():
                raise FileNotFoundError(f"Cannot skip baseline run; {candidate} forecast is missing")
        lgb_training_seconds = None
        lgb_inference_seconds = None
        seasonal_seconds = None
    else:
        fit_start = perf_counter()
        lgb_model, lgb_info = train_split(
            panel, FIT_END, VALID_START, VALID_END, calendar=False,
        )
        lgb_training_seconds = perf_counter() - fit_start

        seasonal_predictions = []
        seasonal_start = perf_counter()
        for history in contexts:
            seasonal_predictions.append(seasonal_naive(history, HORIZON))
        seasonal_seconds = perf_counter() - seasonal_start
        write_forecasts("seasonal_naive", codes, seasonal_predictions, seasonal_seconds)

        lgb_predictions = []
        lgb_infer_start = perf_counter()
        for history in contexts:
            lgb_predictions.append(predict_lgbm(
                lgb_model, history, ORIGIN, HORIZON, calendar=False,
            ))
        lgb_inference_seconds = perf_counter() - lgb_infer_start
        write_forecasts("lightgbm", codes, lgb_predictions, lgb_inference_seconds)
        save_run_metadata(
            "lightgbm", method="global LightGBM, sales-only recursive",
            products=len(codes), fit_targets_end=FIT_END.isoformat(),
            checkpoint_validation=[VALID_START.isoformat(), VALID_END.isoformat()],
            forecast_origin=ORIGIN.isoformat(), selection_period=["2011-07-01", "2011-07-28"],
            selection_targets_loaded=False, context_days=CONTEXT, horizon_days=HORIZON,
            training_seconds=lgb_training_seconds, inference_seconds=lgb_inference_seconds,
            **lgb_info,
        )

    train_x, train_y = windows(panel, date.min, FIT_END)
    valid_x, valid_y = windows(panel, VALID_START, VALID_END)
    del panel
    monitor = start_gpu_monitor(cap_mib=1536, threshold_mib=1900)
    lstm_start = perf_counter()
    lstm_model, lstm_info = train_lstm(train_x, train_y, valid_x, valid_y, monitor)
    lstm_training_seconds = perf_counter() - lstm_start
    del train_x, train_y, valid_x, valid_y
    torch.cuda.empty_cache()
    lstm_predictions = []
    inference_start = perf_counter()
    for history in contexts:
        monitor.check()
        lstm_predictions.append(predict_lstm(
            lstm_model, history, ORIGIN, HORIZON, "sales",
            metadata_values=None, context=CONTEXT, device="cuda", monitor=monitor,
        ))
    lstm_inference_seconds = perf_counter() - inference_start
    write_forecasts("lstm", codes, lstm_predictions, lstm_inference_seconds)
    gpu = monitor.finish()
    if not lstm_only:
        save_run_metadata(
            "seasonal_naive", method="weekly seasonal naive", products=len(codes),
            forecast_origin=ORIGIN.isoformat(), selection_period=["2011-07-01", "2011-07-28"],
            selection_targets_loaded=False, context_days=CONTEXT, horizon_days=HORIZON,
            inference_seconds=seasonal_seconds,
        )
    save_run_metadata(
        "lstm", method="sales-only global LSTM; early stopping on May, refit through May",
        products=len(codes), fit_targets_end=FIT_END.isoformat(),
        checkpoint_validation=[VALID_START.isoformat(), VALID_END.isoformat()],
        forecast_origin=ORIGIN.isoformat(), selection_period=["2011-07-01", "2011-07-28"],
        selection_targets_loaded=False, context_days=CONTEXT, horizon_days=HORIZON,
        training_seconds=lstm_training_seconds, inference_seconds=lstm_inference_seconds,
        gpu=gpu, peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        **lstm_info,
    )
    print(json.dumps({
        "products": len(codes), "lightgbm_training_seconds": lgb_training_seconds,
        "lstm_training_seconds": lstm_training_seconds,
        "lstm_inference_seconds": lstm_inference_seconds,
        "seasonal_naive_inference_seconds": seasonal_seconds,
        "gpu": gpu,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
