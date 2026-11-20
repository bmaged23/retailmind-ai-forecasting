"""Record data freshness and reproducibility fingerprints for the local service."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from statistics import mean
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/milestone6_ops.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_data_date(path: Path) -> date:
    latest = None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            if latest is None or day > latest:
                latest = day
    if latest is None:
        raise ValueError(f"No data records found in {path}")
    return latest


def forecast_file_profile(path: Path, model: str, expected_rows: int) -> dict:
    count = 0
    latencies = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("model") and row["model"] != model:
                continue
            count += 1
            raw = row.get("inference_seconds_per_series") or row.get("inference_seconds")
            if raw:
                latencies.append(float(raw))
    return {
        "model_rows": count,
        "expected_model_rows": expected_rows,
        "complete": count == expected_rows,
        "mean_inference_seconds_per_product": mean(latencies) if latencies else None,
    }


def collect_snapshot(root: Path = ROOT, config: dict | None = None) -> dict:
    config = config or json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    data_path = root / config["data_file"]
    last_day = latest_data_date(data_path)
    age = (now.date() - last_day).days
    plan_scorecard = json.loads((root / config["model_plan_scorecard"]).read_text(encoding="utf-8"))
    interval_scorecard = json.loads((root / config["interval_scorecard"]).read_text(encoding="utf-8"))
    forecast_config_path = root / config["forecast_config"]
    forecast_config = json.loads(forecast_config_path.read_text(encoding="utf-8"))
    artifacts = {}
    forecast_files_by_model = forecast_config["forecast_files"]
    plan_summary = {row["model"]: row for row in plan_scorecard["plan_summary"]}
    forecast_profiles = {}
    for model, row in plan_summary.items():
        path = root / forecast_files_by_model[model]
        expected_rows = int(plan_scorecard["products"]) * int(plan_scorecard["days"])
        forecast_profiles[model] = forecast_file_profile(path, model, expected_rows) if path.is_file() else {
            "model_rows": 0, "expected_model_rows": expected_rows, "complete": False,
            "mean_inference_seconds_per_product": None,
        }
    missing_artifacts = []
    for relative in config["forecast_artifact_paths"]:
        path = root / relative
        if not path.is_file():
            missing_artifacts.append(relative)
        artifacts[relative] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    model_versions = {}
    for model, row in plan_summary.items():
        path = root / forecast_files_by_model[model]
        model_versions[model] = {
            "label": row["model_label"],
            "forecast_file": str(path.relative_to(root)),
            "forecast_sha256": sha256_file(path) if path.is_file() else None,
            "historical_mae": row["forecast_mae"],
            "mean_inference_seconds_per_product": forecast_profiles[model]["mean_inference_seconds_per_product"],
            "forecast_rows_complete": forecast_profiles[model]["complete"],
            "forecast_period": plan_scorecard["evaluation_period"],
            "origin": plan_scorecard["forecast_origin"],
        }
    status = "artifact_missing" if missing_artifacts or any(not r["complete"] for r in forecast_profiles.values()) else (
        "fresh" if age <= int(config["max_data_age_days"]) else "stale_input"
    )
    return {
        "checked_at_utc": now.isoformat(),
        "status": status,
        "data": {
            "path": str(data_path.relative_to(root)),
            "latest_record_date": last_day.isoformat(),
            "age_days": age,
            "max_allowed_age_days": int(config["max_data_age_days"]),
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "model_versions": model_versions,
        "interval_reference": {
            "model": interval_scorecard["model"],
            "period": interval_scorecard["evaluation_period"],
            "nominal_coverage": interval_scorecard["nominal_coverage"],
            "observed_coverage": interval_scorecard["overall"]["calibrated_coverage"],
            "mean_width": interval_scorecard["overall"]["calibrated_mean_width"],
        },
        "forecast_artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "forecast_inputs_config_sha256": sha256_file(forecast_config_path),
        "action": "monitor_only_no_forecast_or_retraining_triggered",
        "notes": config["notes"],
    }


def write_snapshot(root: Path, snapshot: dict, output_directory: str) -> None:
    directory = root / output_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one monitoring snapshot and exit")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    args = parser.parse_args()
    if args.interval_hours <= 0:
        parser.error("--interval-hours must be greater than zero")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    while True:
        snapshot = collect_snapshot(ROOT, config)
        write_snapshot(ROOT, snapshot, config["output_directory"])
        print(json.dumps({
            "checked_at_utc": snapshot["checked_at_utc"],
            "status": snapshot["status"],
            "latest_record_date": snapshot["data"]["latest_record_date"],
            "age_days": snapshot["data"]["age_days"],
            "action": snapshot["action"],
        }), flush=True)
        if args.once:
            break
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()
