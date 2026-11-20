"""Experimental TimesFM 3.0 forecast-head fine-tuning and frozen-test scoring.

Only the final quantile prediction head is trainable. The pretrained backbone
is frozen to stay within the project's 2 GiB GPU budget. Test targets are never
loaded during training or validation.
"""
from pathlib import Path
from time import perf_counter
import json
import os

import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

from common import ROOT, RUN_DIR, load_contexts, save_run_metadata, start_gpu_monitor, write_forecasts
from head_finetune_common import (
    CONTEXT, HORIZON, MAX_EPOCHS, PATIENCE, TRAIN_END, VALID_END,
    load_training_windows, random_training_window,
)

OUT = RUN_DIR
ADAPTER = OUT / "models/timesfm3/experimental_head_context90d"
MODEL_ID = "google/timesfm-3.0-pytorch"
BATCH_SIZE = 1
LR = 1e-5
WEIGHT_DECAY = 0.01


def quantile_loss(prediction, target, quantiles):
    error = target.unsqueeze(-1) - prediction
    q = torch.as_tensor(quantiles, device=prediction.device, dtype=prediction.dtype).view(1, -1)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def make_input(context, model, device):
    patch = model.input_patch_len
    values = torch.as_tensor(context, dtype=torch.float32, device=device)
    if values.numel() != CONTEXT:
        raise ValueError(f"Expected a {CONTEXT}-day context")
    padded_length = ((CONTEXT + patch - 1) // patch) * patch
    if padded_length != CONTEXT:
        values = torch.nn.functional.pad(values, (padded_length - CONTEXT, 0))
    values = values.view(1, 1, padded_length // patch, patch)
    batch, _, n_patches, _ = values.shape
    is_target = torch.ones((batch, 1, n_patches), dtype=torch.bool, device=device)
    is_false = torch.zeros_like(is_target)
    positions = torch.arange(n_patches, device=device).view(1, -1)
    return {
        "values": values,
        "masks": torch.zeros_like(values, dtype=torch.bool),
        "patch_segment_ids": torch.zeros_like(positions),
        "patch_positions": positions,
        "patch_is_target": is_target,
        "patch_is_past_only": is_false,
        "patch_is_past_future_covariate": is_false,
    }


def predict_for_training(context, backbone, device):
    # The public TimesFM 3.0 decoder is inference-only. Its underlying PyTorch
    # training forward returns the quantile predictions for every context patch.
    # The final context patch predicts the next 64 days; use the 28-day target.
    logits = backbone(make_input(context, backbone, device))["logits"]
    return logits[0, 0, -1, :HORIZON, :]


def validation_loss(backbone, rows, device):
    losses = []
    backbone.eval()
    with torch.no_grad():
        for context, target in rows:
            prediction = predict_for_training(context, backbone, device)
            expected = torch.as_tensor(target, dtype=torch.float32, device=device)
            losses.append(float(quantile_loss(prediction, expected, backbone.quantiles)))
    return float(np.mean(losses))


def main():
    os.environ.setdefault("HF_HOME", "/mnt/d/retailmind-model-cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    _, codes, test_contexts = load_contexts()
    train_series, validation, rng = load_training_windows()

    monitor = start_gpu_monitor(cap_mib=1800, threshold_mib=1950)
    device = "cuda"
    load_start = perf_counter()
    forecaster = TimesFM3Evaluator(ModelConfig(
        checkpoint_path=MODEL_ID, per_core_batch_size=BATCH_SIZE, device=device,
    ))
    backbone = forecaster.model
    load_seconds = perf_counter() - load_start
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in backbone.output_head.parameters():
        parameter.requires_grad_(True)
    backbone.eval()
    trainable = [p for p in backbone.output_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    progress_path = ADAPTER / "training_progress.pt"
    print(json.dumps({"model": MODEL_ID, "train_series": len(train_series),
        "validation_series": len(validation), "trainable_parameters": sum(p.numel() for p in trainable),
        "gpu_cap_mib": 1800, "batch_size": BATCH_SIZE}), flush=True)

    best = float("inf")
    best_state = None
    stale = 0
    history = []
    start_epoch = 1
    training_seconds_so_far = 0.0
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        backbone.output_head.load_state_dict(progress["latest_head_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        rng.bit_generator.state = progress["rng_state"]
        best = progress["best_validation_loss"]
        best_state = progress["best_head_state"]
        stale = progress["stale_epochs"]
        history = progress["history"]
        training_seconds_so_far = progress.get("cumulative_training_seconds", 0.0)
        start_epoch = progress["completed_epoch"] + 1
        print(f"Resuming from completed epoch {start_epoch - 1}; best validation pinball={best:.6f}", flush=True)
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = perf_counter()
        order = rng.permutation(len(train_series))
        train_losses = []
        for step, idx in enumerate(order, 1):
            monitor.check()
            context, target = random_training_window(train_series[int(idx)], rng)
            prediction = predict_for_training(context, backbone, device)
            expected = torch.as_tensor(target, dtype=torch.float32, device=device)
            loss = quantile_loss(prediction, expected, backbone.quantiles)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if step % 500 == 0:
                print(f"epoch={epoch} series={step}/{len(order)}", flush=True)

        train_loss = float(np.mean(train_losses))
        val_loss = validation_loss(backbone, validation, device)
        history.append({"epoch": epoch, "train_pinball_loss": train_loss,
                        "validation_pinball_loss": val_loss})
        print(f"epoch={epoch} train_pinball={train_loss:.6f} validation_pinball={val_loss:.6f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in backbone.output_head.state_dict().items()}
            ADAPTER.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ADAPTER / "output_head.pt")
        else:
            stale += 1
        ADAPTER.mkdir(parents=True, exist_ok=True)
        training_seconds_so_far += perf_counter() - epoch_start
        torch.save({
            "completed_epoch": epoch,
            "latest_head_state": {k: v.detach().cpu().clone()
                                  for k, v in backbone.output_head.state_dict().items()},
            "best_head_state": best_state,
            "best_validation_loss": best,
            "stale_epochs": stale,
            "history": history,
            "cumulative_training_seconds": training_seconds_so_far,
            "optimizer_state": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
        }, progress_path)
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation pinball={best:.6f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best validation checkpoint was produced")
    backbone.output_head.load_state_dict(best_state)
    training_seconds = training_seconds_so_far
    del optimizer
    torch.cuda.empty_cache()

    predictions = []
    inference_start = perf_counter()
    for index, context in enumerate(test_contexts, 1):
        monitor.check()
        result = next(iter(forecaster.predict_batch(
            [context], horizon=HORIZON, return_quantiles=True,
            use_symmetric_averaging=False,
        )))
        predictions.append(np.maximum(0.0, np.asarray(result.forecast, dtype=np.float32)))
        if index % 500 == 0:
            print(f"Fine-tuned forecasts {index}/{len(codes)}", flush=True)
    inference_seconds = perf_counter() - inference_start
    forecast_path = write_forecasts("timesfm3_head_tuned", codes, predictions, inference_seconds)
    gpu = monitor.finish()
    metadata = {
        "model_id": MODEL_ID,
        "method": "experimental output-head fine-tuning; frozen pretrained backbone",
        "checkpoint_path": str(ADAPTER / "output_head.pt"),
        "train_targets_end": TRAIN_END.isoformat(), "validation_targets_end": VALID_END.isoformat(),
        "test_targets_loaded": False, "products": len(codes),
        "train_series": len(train_series), "validation_sample_size": len(validation),
        "validation_seed": 42, "context_days": CONTEXT, "horizon_days": HORIZON,
        "model_padded_context_days": ((CONTEXT + backbone.input_patch_len - 1)
                                       // backbone.input_patch_len) * backbone.input_patch_len,
        "batch_size": BATCH_SIZE, "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
        "best_validation_pinball_loss": best, "epochs": history,
        "load_seconds": load_seconds, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "gpu": gpu,
        "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "license_note": "TimesFM 3.0 derivatives and weights are not for distribution; research/non-commercial use only.",
    }
    save_run_metadata("timesfm3_head_tuned", **metadata)
    print(json.dumps({"forecast_file": str(forecast_path), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
