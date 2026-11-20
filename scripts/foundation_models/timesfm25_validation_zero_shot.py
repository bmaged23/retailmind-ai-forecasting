"""Generate TimesFM 2.5 zero-shot forecasts for the frozen July cohort."""
from time import perf_counter
import json
import os

import numpy as np
import torch
from transformers import TimesFm2_5ModelForPrediction

from common import (
    HORIZON, ORIGIN, ROOT, RUN_VALIDATION, load_contexts, save_run_metadata,
    start_gpu_monitor, write_forecasts,
)

MODEL_ID = "google/timesfm-2.5-200m-transformers"
MODEL_CACHE = "/mnt/d/retailmind-model-cache/huggingface"
BATCH_SIZE = 4


def main() -> None:
    if not RUN_VALIDATION:
        raise RuntimeError("Set RETAILMIND_RUN_SPLIT=validation")
    os.environ.setdefault("HF_HOME", MODEL_CACHE)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    _, codes, contexts = load_contexts()
    monitor = start_gpu_monitor(cap_mib=1536, threshold_mib=1900)
    started = perf_counter()
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE, torch_dtype=torch.float32,
    ).to("cuda").eval()
    load_seconds = perf_counter() - started
    predictions = []
    infer_started = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(contexts), BATCH_SIZE):
            monitor.check()
            batch = [torch.as_tensor(x, dtype=torch.float32, device="cuda")
                     for x in contexts[offset:offset + BATCH_SIZE]]
            output = model(
                past_values=batch, forecast_context_len=96,
                force_flip_invariance=False, truncate_negative=False,
            )
            values = output.mean_predictions[:, :HORIZON].float().cpu().numpy()
            predictions.extend(np.maximum(values, 0.0))
    inference_seconds = perf_counter() - infer_started
    path = write_forecasts("timesfm25_zero_shot", codes, predictions, inference_seconds)
    gpu = monitor.finish()
    save_run_metadata(
        "timesfm25_zero_shot", model_id=MODEL_ID, method="zero-shot",
        products=len(codes), forecast_origin=ORIGIN.isoformat(),
        selection_period=["2011-07-01", "2011-07-28"],
        selection_targets_loaded=False, observed_context_days=len(contexts[0]),
        model_context_length=96, horizon_days=HORIZON, batch_size=BATCH_SIZE,
        device="cuda", load_seconds=load_seconds,
        inference_seconds=inference_seconds, gpu=gpu,
        peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
    )
    print(json.dumps({"forecast_file": str(path), "load_seconds": load_seconds,
                      "inference_seconds": inference_seconds, "gpu": gpu}, indent=2), flush=True)


if __name__ == "__main__":
    main()
