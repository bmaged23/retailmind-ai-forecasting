"""Generate fixed-model Timestral P10/P50/P90 forecasts for Milestone 2."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import numpy as np
import torch
from timestral import TimestralForecaster
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/milestone2_uncertainty.json"
OUT = ROOT / "artifacts/milestone2_uncertainty/timestral209m_lora"


def load_panel_through(path: Path, end: date) -> dict[str, dict[date, float]]:
    """Read only observations available by forecast origin."""
    panel: dict[str, dict[date, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            if day <= end:
                panel.setdefault(row["stock_code"], {})[day] = float(row["quantity"])
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["calibration", "evaluation"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    split = config[args.split]
    origin = date.fromisoformat(split["forecast_origin"])
    target_start = date.fromisoformat(split["target_start"])
    target_end = date.fromisoformat(split["target_end"])
    context_days = int(config["context_days"])
    horizon = int(config["horizon_days"])
    if (target_start != origin + timedelta(days=1)
            or (target_end - target_start).days + 1 != horizon):
        raise ValueError(f"Invalid {args.split} dates in {CONFIG_PATH}")

    cohort = json.loads((ROOT / config["cohort_file"]).read_text(encoding="utf-8"))
    codes = cohort["codes"]
    panel = load_panel_through(ROOT / config["source_data"], origin)
    context_start = origin - timedelta(days=context_days - 1)
    contexts, sale_days = [], []
    for code in codes:
        series = panel.get(code, {})
        if not any(quantity > 0 for day, quantity in series.items() if day <= origin):
            raise ValueError(f"Frozen cohort product has no history by {origin}: {code}")
        values = [series.get(context_start + timedelta(days=i), 0.0)
                  for i in range(context_days)]
        contexts.append(np.asarray(values, dtype=np.float32))
        sale_days.append(sum(value > 0 for value in values))

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from retailmind.gpu_monitor import GPUMonitor

    cap_mib = config["resources"]["gpu_allocator_cap_mib"]
    watchdog_mib = int(config["resources"]["gpu_watchdog_mib"])
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if cap_mib is not None:
            cap_mib = int(cap_mib)
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, cap_mib * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0
            )
        torch.backends.cudnn.benchmark = False

    load_started = perf_counter()
    model_snapshot = snapshot_download(
        repo_id=config["model_id"], cache_dir=str(ROOT / ".cache/huggingface/hub")
    )
    base_forecaster = TimestralForecaster.from_pretrained(
        model_snapshot, device=args.device
    )
    base = base_forecaster.model
    adapter = ROOT / config["adapter_path"]
    if not adapter.exists():
        raise FileNotFoundError(f"Fine-tuned adapter not found: {adapter}")
    from peft import PeftModel
    wrapped = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model = wrapped.get_base_model()
    forecaster = TimestralForecaster(model, max_context=context_days)
    load_seconds = perf_counter() - load_started
    # Starting nvidia-smi polling before Timestral loads caused CUDA allocator
    # failures in this WSL session. Measure model-load peak through PyTorch and
    # begin device-level watchdog sampling immediately afterward.
    if args.device == "cuda":
        load_peak_pytorch_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        monitor = GPUMonitor(max_used_mib=watchdog_mib).start()
        monitor.check()
    else:
        load_peak_pytorch_reserved_mib = None
        monitor = None
    levels = np.asarray(forecaster.quantile_levels, dtype=float)
    requested = np.asarray(config["quantile_levels"], dtype=float)
    if requested.min() < levels.min() or requested.max() > levels.max():
        raise ValueError(f"Requested quantiles {requested} outside model levels {levels}")

    out_path = OUT / f"{args.split}-forecasts.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["resources"]["inference_batch_size"])
    started = perf_counter()
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model", "stock_code", "forecast_origin", "target_date", "horizon",
            "context_sale_days", "p10", "p50", "p90", "inference_seconds_per_series",
        ])
        writer.writeheader()
        for offset in range(0, len(codes), batch_size):
            if monitor:
                monitor.check()
            batch = np.asarray(contexts[offset:offset + batch_size], dtype=np.float32)[:, None, :]
            result = forecaster.forecast(batch, prediction_length=horizon)
            raw = result.quantiles.detach().float().cpu().numpy()[:, 0, :, :]
            selected = np.stack([
                np.stack([np.interp(requested, levels, step_values)
                          for step_values in product_forecast], axis=0)
                for product_forecast in raw
            ], axis=0)
            selected = np.maximum.accumulate(np.maximum(selected, 0.0), axis=-1)
            for local_idx, product_quantiles in enumerate(selected):
                code_idx = offset + local_idx
                code = codes[code_idx]
                for step, quantiles in enumerate(product_quantiles, 1):
                    target = origin + timedelta(days=step)
                    writer.writerow({
                        "model": config["model"],
                        "stock_code": code,
                        "forecast_origin": origin.isoformat(),
                        "target_date": target.isoformat(),
                        "horizon": step,
                        "context_sale_days": sale_days[code_idx],
                        "p10": float(quantiles[0]),
                        "p50": float(quantiles[1]),
                        "p90": float(quantiles[2]),
                        "inference_seconds_per_series": "",
                    })
            print(f"Forecasted {min(offset + batch_size, len(codes))}/{len(codes)}", flush=True)

    inference_seconds = perf_counter() - started
    per_product = inference_seconds / len(codes)
    # Fill a constant per-series timing without retaining all forecast rows in RAM.
    temp = out_path.with_suffix(".tmp")
    with out_path.open(newline="", encoding="utf-8") as src, temp.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["inference_seconds_per_series"] = f"{per_product:.12f}"
            writer.writerow(row)
    temp.replace(out_path)
    gpu = monitor.finish() if monitor else None
    metadata = {
        "model": config["model"], "model_id": config["model_id"],
        "model_revision": Path(model_snapshot).name,
        "method": "saved fine-tuned adapter; pretrained backbone frozen at inference",
        "split": args.split, "forecast_origin": origin.isoformat(),
        "device": args.device,
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "target_period": [target_start.isoformat(), target_end.isoformat()],
        "products": len(codes), "forecast_rows": len(codes) * horizon,
        "context_days": context_days, "horizon_days": horizon,
        "native_quantile_levels": levels.tolist(),
        "reported_quantile_levels": requested.tolist(),
        "adapter_path": str(adapter.relative_to(ROOT)),
        "selection_period": ["2011-07-01", "2011-07-28"],
        "selection_targets_loaded_for_this_run": False,
        "load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "mean_inference_seconds_per_product": per_product,
        "gpu_allocator_cap_mib": cap_mib,
        "gpu_watchdog_mib": watchdog_mib,
        "load_peak_pytorch_reserved_mib": load_peak_pytorch_reserved_mib,
        "gpu": gpu,
        "peak_pytorch_reserved_mib": (
            torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else None
        ),
        "forecast_file": str(out_path.relative_to(ROOT)),
    }
    (OUT / f"{args.split}-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
