"""Generate Chronos-2 zero-shot forecasts for the frozen July cohort."""
from time import perf_counter
import json
import os

import numpy as np
import torch
from chronos import Chronos2Pipeline

from common import (
    HORIZON, ORIGIN, ROOT, RUN_VALIDATION, load_contexts, save_run_metadata,
    start_gpu_monitor, write_forecasts,
)


def main() -> None:
    if not RUN_VALIDATION:
        raise RuntimeError("Set RETAILMIND_RUN_SPLIT=validation")
    os.environ.setdefault("HF_HOME", "/mnt/d/retailmind-model-cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    _, codes, contexts = load_contexts()
    monitor = start_gpu_monitor(cap_mib=1536, threshold_mib=1900)
    started = perf_counter()
    model = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    load_seconds = perf_counter() - started
    predictions = []
    infer_started = perf_counter()
    batch_size = 16
    for offset in range(0, len(contexts), batch_size):
        monitor.check()
        quantiles, _ = model.predict_quantiles(
            contexts[offset:offset + batch_size], prediction_length=HORIZON,
            quantile_levels=[0.5], batch_size=batch_size, context_length=len(contexts[0]),
        )
        for value in quantiles:
            array = value.detach().float().cpu().numpy()
            predictions.append(np.maximum(0.0, array.reshape(-1, HORIZON, 1)[0, :, 0]))
    inference_seconds = perf_counter() - infer_started
    path = write_forecasts("chronos2_zero_shot", codes, predictions, inference_seconds)
    gpu = monitor.finish()
    save_run_metadata(
        "chronos2_zero_shot", model_id="amazon/chronos-2", method="zero-shot",
        products=len(codes), forecast_origin=ORIGIN.isoformat(),
        selection_period=["2011-07-01", "2011-07-28"],
        selection_targets_loaded=False, context_days=len(contexts[0]),
        horizon_days=HORIZON, batch_size=batch_size, device="cuda",
        load_seconds=load_seconds, inference_seconds=inference_seconds,
        gpu=gpu, peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
    )
    print(json.dumps({"forecast_file": str(path), "load_seconds": load_seconds,
                      "inference_seconds": inference_seconds, "gpu": gpu}, indent=2), flush=True)


if __name__ == "__main__":
    main()
