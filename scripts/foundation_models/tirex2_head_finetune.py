"""Experimental TiRex-2 forecast-head fine-tuning and frozen-test scoring.

The open TiRex-2 package exposes inference but no vendor training workflow.
This experiment freezes its pretrained backbone and trains only the output
projection with a differentiable supervised pinball loss.
"""
from time import perf_counter
import json
import os

import numpy as np
import torch
from tirex2 import TimeseriesType, load_model

from common import ROOT, RUN_DIR, load_contexts, save_run_metadata, start_gpu_monitor, write_forecasts
from head_finetune_common import (
    CONTEXT, HORIZON, MAX_EPOCHS, PATIENCE, TRAIN_END, VALID_END,
    load_training_windows, random_training_window,
)

OUT = RUN_DIR
ADAPTER = OUT / "models/tirex2/experimental_head_context90d"
MODEL_ID = "NX-AI/TiRex-2"
BATCH_SIZE = 1
LR = 1e-5
WEIGHT_DECAY = 0.01


def make_series(context, device="cpu"):
    target = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
    return TimeseriesType(target=target, past_covariates=None, future_covariates=None)


def quantile_loss(prediction, target, quantiles):
    error = target.unsqueeze(0) - prediction
    q = quantiles.to(device=prediction.device, dtype=prediction.dtype).view(-1, 1)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def predict_for_training(context, backbone, preserve_grad):
    # This is TiRex-2's differentiable internal prediction path. The public
    # predict()/forecast() methods are decorated with no_grad for inference.
    result = backbone._predict(
        [make_series(context)], HORIZON, preserve_grad=preserve_grad,
        tta_diff=backbone.tta_diff, pad_context=False,
    )[0]
    # TiRex's output formatter returns forecasts on the original series device
    # (CPU here); keep the tiny [quantile, horizon] tensor on the training GPU.
    return result[0].to(next(backbone.parameters()).device)


def validation_loss(backbone, rows, device):
    losses = []
    backbone.eval()
    with torch.no_grad():
        for context, target in rows:
            prediction = predict_for_training(context, backbone, preserve_grad=False).to(device)
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
    forecaster = load_model(MODEL_ID, device=device, cuda_graphs=False)
    backbone = forecaster.model
    load_seconds = perf_counter() - load_start
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in backbone.output_patch_embedding.parameters():
        parameter.requires_grad_(True)
    backbone.eval()
    trainable = [p for p in backbone.output_patch_embedding.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    print(json.dumps({"model": MODEL_ID, "train_series": len(train_series),
        "validation_series": len(validation), "trainable_parameters": sum(p.numel() for p in trainable),
        "gpu_cap_mib": 1800, "batch_size": BATCH_SIZE}), flush=True)

    best = float("inf")
    best_state = None
    stale = 0
    history = []
    training_start = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(len(train_series))
        train_losses = []
        for step, idx in enumerate(order, 1):
            monitor.check()
            context, target = random_training_window(train_series[int(idx)], rng)
            prediction = predict_for_training(context, backbone, preserve_grad=True)
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
                          for k, v in backbone.output_patch_embedding.state_dict().items()}
            ADAPTER.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ADAPTER / "output_patch_embedding.pt")
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation pinball={best:.6f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best validation checkpoint was produced")
    backbone.output_patch_embedding.load_state_dict(best_state)
    training_seconds = perf_counter() - training_start
    del optimizer
    torch.cuda.empty_cache()

    test_series = [make_series(context) for context in test_contexts]
    predictions = []
    inference_start = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(test_series), 32):
            monitor.check()
            forecast_batch = forecaster.forecast(
                test_series[offset:offset + 32], prediction_length=HORIZON,
                output_type="numpy", batch_size=32,
            )
            predictions.extend([np.maximum(0.0, np.asarray(f, dtype=np.float32)[0, 4, :])
                                for f in forecast_batch])
            if (offset + 32) % 512 < 32:
                print(f"Fine-tuned forecasts {min(offset + 32, len(codes))}/{len(codes)}", flush=True)
    inference_seconds = perf_counter() - inference_start
    forecast_path = write_forecasts("tirex2_head_tuned", codes, predictions, inference_seconds)
    gpu = monitor.finish()
    metadata = {
        "model_id": MODEL_ID,
        "method": "experimental output-head fine-tuning; frozen pretrained backbone",
        "vendor_fine_tuning": "TiRex-2 PRO feature; this run uses the open Apache-2.0 PyTorch model",
        "checkpoint_path": str(ADAPTER / "output_patch_embedding.pt"),
        "train_targets_end": TRAIN_END.isoformat(), "validation_targets_end": VALID_END.isoformat(),
        "test_targets_loaded": False, "products": len(codes),
        "train_series": len(train_series), "validation_sample_size": len(validation),
        "validation_seed": 42, "context_days": CONTEXT, "horizon_days": HORIZON,
        "batch_size": BATCH_SIZE, "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
        "best_validation_pinball_loss": best, "epochs": history,
        "load_seconds": load_seconds, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "gpu": gpu,
        "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    save_run_metadata("tirex2_head_tuned", **metadata)
    print(json.dumps({"forecast_file": str(forecast_path), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
