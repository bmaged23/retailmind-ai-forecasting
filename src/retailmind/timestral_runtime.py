"""Load the packaged Timestral LoRA forecaster once per API process."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

_INFERENCE_LOCK = Lock()


@lru_cache(maxsize=1)
def _load_forecaster():
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from timestral import TimestralForecaster

    model_id = os.environ.get("TIMESTRAL_MODEL_ID", "mistralai/Timestral-209M")
    model_dir = os.environ.get("TIMESTRAL_BASE_MODEL_DIR")
    if not model_dir:
        model_dir = snapshot_download(
            repo_id=model_id,
            revision=os.environ.get("TIMESTRAL_MODEL_REVISION"),
            cache_dir=os.environ.get("HF_HOME", "/opt/retailmind-model-cache"),
        )
    device = os.environ.get("TIMESTRAL_DEVICE", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TIMESTRAL_DEVICE=cuda but CUDA is unavailable")
    if device == "cuda":
        cap_mib = int(os.environ.get("TIMESTRAL_MAX_GPU_MIB", "2048"))
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        fraction = min(1.0, cap_mib * 1024**2 / total_bytes)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    adapter = Path(os.environ.get(
        "TIMESTRAL_ADAPTER_DIR", "/app/models/timestral209m_lora"
    ))
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Timestral LoRA weights are missing from {adapter}")

    base = TimestralForecaster.from_pretrained(model_dir, device=device).model
    tuned = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    return TimestralForecaster(tuned.get_base_model(), max_context=90), device


def forecast_quantiles(values: list[float], horizon: int) -> tuple[list[list[float]], str]:
    """Return nonnegative, ordered P10/P50/P90 values for each horizon day."""
    with _INFERENCE_LOCK:
        forecaster, device = _load_forecaster()
        context = np.asarray(values[-90:], dtype=np.float32)[None, None, :]
        result = forecaster.forecast(context, prediction_length=horizon)
        raw = result.quantiles.detach().float().cpu().numpy()[0, 0]
        levels = np.asarray(forecaster.quantile_levels, dtype=float)
        requested = np.asarray([0.1, 0.5, 0.9], dtype=float)
        selected = np.stack([
            np.interp(requested, levels, step_values) for step_values in raw
        ])
    selected = np.maximum.accumulate(np.maximum(selected, 0.0), axis=-1)
    calibrated = selected.copy()
    calibration_path = Path(os.environ.get(
        "TIMESTRAL_CALIBRATION_FILE", "/app/models/timestral209m_lora/calibration.json"
    ))
    if calibration_path.is_file():
        import json
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        for index, row in enumerate(calibrated):
            adjustment = float(calibration["horizon_adjustments"].get(
                str(index + 1), {"adjustment_units": 0.0}
            )["adjustment_units"])
            row[0] = min(row[1], max(0.0, row[0] - adjustment))
            row[2] = max(row[1], row[2] + adjustment)
    result = np.concatenate([selected, calibrated[:, [0, 2]]], axis=1)
    return result.tolist(), device


def model_status() -> dict:
    return {
        "id": "timestral209m_lora",
        "label": "Timestral 209M LoRA",
        "adapter_training_window": "validation adapter trained through 2011-04-30",
        "adapter_path": os.environ.get("TIMESTRAL_ADAPTER_DIR", "/app/models/timestral209m_lora"),
        "base_model_id": os.environ.get("TIMESTRAL_MODEL_ID", "mistralai/Timestral-209M"),
        "base_model_revision": os.environ.get("TIMESTRAL_MODEL_REVISION", "latest"),
        "device": os.environ.get("TIMESTRAL_DEVICE", "auto"),
        "max_gpu_memory_mib": int(os.environ.get("TIMESTRAL_MAX_GPU_MIB", "2048")),
    }


def model_files_missing() -> list[str]:
    base = Path(os.environ.get(
        "TIMESTRAL_BASE_MODEL_DIR",
        "/opt/retailmind-model-cache/models--mistralai--Timestral-209M/snapshots/current",
    ))
    adapter = Path(os.environ.get(
        "TIMESTRAL_ADAPTER_DIR", "/app/models/timestral209m_lora"
    ))
    missing = []
    if not (base / "config.json").is_file():
        missing.append(str(base / "config.json"))
    if not (list(base.glob("*.safetensors")) or list(base.glob("*.bin"))):
        missing.append(f"model weights under {base}")
    if not (adapter / "adapter_model.safetensors").is_file():
        missing.append(str(adapter / "adapter_model.safetensors"))
    return missing
