"""Run July inference from the saved experimental TiRex-2 output head."""
from time import perf_counter
import json
import os

import numpy as np
import torch
from tirex2 import TimeseriesType, load_model

from common import (
    CONTEXT, HORIZON, ORIGIN, ROOT, RUN_VALIDATION, load_contexts,
    save_run_metadata, start_gpu_monitor, write_forecasts,
)


def main() -> None:
    if not RUN_VALIDATION:
        raise RuntimeError("Set RETAILMIND_RUN_SPLIT=validation")
    os.environ.setdefault("HF_HOME", "/mnt/d/retailmind-model-cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    _, codes, contexts = load_contexts()
    monitor = start_gpu_monitor(cap_mib=1800, threshold_mib=1950)
    load_started = perf_counter()
    forecaster = load_model("NX-AI/TiRex-2", device="cuda", cuda_graphs=False)
    load_seconds = perf_counter() - load_started
    adapter_path = (
        ROOT / "artifacts/validation_comparison_2011-07-01_to_2011-07-28"
        / "models/tirex2/experimental_head_context90d/output_patch_embedding.pt"
    )
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    forecaster.model.output_patch_embedding.load_state_dict(state)
    series = [
        TimeseriesType(
            target=torch.as_tensor(values, dtype=torch.float32, device="cuda").unsqueeze(0),
            past_covariates=None,
            future_covariates=None,
        )
        for values in contexts
    ]
    predictions = []
    started = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(series), 32):
            monitor.check()
            forecasts = forecaster.forecast(
                series[offset:offset + 32], prediction_length=HORIZON,
                output_type="numpy", batch_size=32,
            )
            predictions.extend([
                np.asarray(f, dtype=np.float32)[0, 4, :] for f in forecasts
            ])
            if (offset + 32) % 512 < 32:
                print(f"Fine-tuned forecasts {min(offset + 32, len(codes))}/{len(codes)}", flush=True)
    inference_seconds = perf_counter() - started
    path = write_forecasts("tirex2_head_tuned", codes, predictions, inference_seconds)
    gpu = monitor.finish()
    save_run_metadata(
        "tirex2_head_tuned", model_id="NX-AI/TiRex-2",
        method="experimental output-head tuning; frozen pretrained backbone",
        vendor_fine_tuning="TiRex-2 PRO feature; this run uses the open Apache-2.0 PyTorch model",
        checkpoint_path=str(adapter_path), train_targets_end="2011-04-30",
        validation_targets_end="2011-05-28", forecast_origin=ORIGIN.isoformat(),
        selection_targets_loaded=False, products=len(codes), context_days=CONTEXT,
        horizon_days=HORIZON, batch_size=32, best_validation_pinball_loss=1.193349,
        best_epoch=1, epochs=[
            {"epoch": 1, "train_pinball_loss": 1.610370, "validation_pinball_loss": 1.193349},
            {"epoch": 2, "train_pinball_loss": 1.466777, "validation_pinball_loss": 1.194333},
            {"epoch": 3, "train_pinball_loss": 1.464791, "validation_pinball_loss": 1.194690},
        ], early_stopping_patience=2, load_seconds=load_seconds,
        inference_seconds=inference_seconds, gpu=gpu,
        peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        inference_only_recovery=True,
    )
    print(json.dumps({"forecast_file": str(path), "load_seconds": load_seconds,
                      "inference_seconds": inference_seconds, "gpu": gpu}, indent=2), flush=True)


if __name__ == "__main__":
    main()
