"""Calibrate Timestral's central 80% intervals on a frozen later window."""
from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/milestone2_uncertainty.json").read_text())
OUT = ROOT / "artifacts/milestone2_uncertainty/timestral209m_lora"


def load_actuals(end: date) -> dict[tuple[str, str], float]:
    actuals = {}
    with (ROOT / CONFIG["source_data"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if date.fromisoformat(row["date"]) <= end:
                actuals[(row["stock_code"], row["date"])] = float(row["quantity"])
    return actuals


def read_forecasts(path: Path, split: str) -> list[dict]:
    period = CONFIG[split]
    start = date.fromisoformat(period["target_start"])
    end = date.fromisoformat(period["target_end"])
    cohort = json.loads((ROOT / CONFIG["cohort_file"]).read_text())["codes"]
    expected = {
        (code, (start.fromordinal(start.toordinal() + step)).isoformat())
        for code in cohort for step in range((end - start).days + 1)
    }
    rows = []
    by_key = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["stock_code"], row["target_date"])
            if key in by_key:
                raise ValueError(f"Duplicate {split} forecast: {key}")
            by_key.add(key)
            if row["forecast_origin"] != period["forecast_origin"]:
                raise ValueError(f"Wrong forecast origin for {key}: {row['forecast_origin']}")
            for name in ("p10", "p50", "p90"):
                row[name] = float(row[name])
            row["horizon"] = int(row["horizon"])
            row["context_sale_days"] = int(row["context_sale_days"])
            rows.append(row)
    if by_key != expected:
        raise ValueError(
            f"{split} forecast key mismatch: {len(expected - by_key)} missing, "
            f"{len(by_key - expected)} extra"
        )
    return rows


def main() -> None:
    alpha = 1.0 - float(CONFIG["interval_coverage_target"])
    interval = CONFIG["quantile_levels"]
    if interval != [0.1, 0.5, 0.9]:
        raise ValueError("This calibration runner expects P10/P50/P90 base quantiles")

    forecasts = read_forecasts(OUT / "calibration-forecasts.csv", "calibration")
    # Calibration labels are loaded only after the complete forecast grid is verified.
    actuals = load_actuals(date.fromisoformat(CONFIG["calibration"]["target_end"]))
    scores_by_horizon: dict[int, list[float]] = {}
    scored = []
    for row in forecasts:
        key = (row["stock_code"], row["target_date"])
        actual = actuals.get(key, 0.0)
        score = max(row["p10"] - actual, actual - row["p90"])
        scores_by_horizon.setdefault(row["horizon"], []).append(score)
        scored.append({**row, "actual": actual, "nonconformity_score": score})

    adjustments = {}
    for horizon, scores in sorted(scores_by_horizon.items()):
        n = len(scores)
        rank = math.ceil((n + 1) * (1 - alpha))
        qhat = sorted(scores)[min(rank, n) - 1]
        adjustments[str(horizon)] = {
            "calibration_products": n,
            "finite_sample_rank_1_based": rank,
            "adjustment_units": qhat,
        }

    coverage = {"raw": {}, "calibrated": {}}
    widths = {"raw": {}, "calibrated": {}}
    for horizon, scores in sorted(scores_by_horizon.items()):
        horizon_rows = [row for row in scored if row["horizon"] == horizon]
        qhat = adjustments[str(horizon)]["adjustment_units"]
        raw_covered, cal_covered, raw_width, cal_width = [], [], [], []
        for row in horizon_rows:
            lo, hi, y = row["p10"], row["p90"], row["actual"]
            adj_lo = min(row["p50"], max(0.0, lo - qhat))
            adj_hi = max(row["p50"], hi + qhat)
            raw_covered.append(lo <= y <= hi)
            cal_covered.append(adj_lo <= y <= adj_hi)
            raw_width.append(hi - lo)
            cal_width.append(adj_hi - adj_lo)
        coverage["raw"][str(horizon)] = sum(raw_covered) / len(raw_covered)
        coverage["calibrated"][str(horizon)] = sum(cal_covered) / len(cal_covered)
        widths["raw"][str(horizon)] = sum(raw_width) / len(raw_width)
        widths["calibrated"][str(horizon)] = sum(cal_width) / len(cal_width)

    cal_path = OUT / "calibration.json"
    result = {
        "model": CONFIG["model"],
        "method": "horizon-wise conformalized quantile regression with signed adjustment",
        "nominal_interval_coverage": 1 - alpha,
        "quantiles": interval,
        "calibration_origin": CONFIG["calibration"]["forecast_origin"],
        "calibration_period": [CONFIG["calibration"]["target_start"], CONFIG["calibration"]["target_end"]],
        "products": len({row["stock_code"] for row in scored}),
        "rows": len(scored),
        "horizon_adjustments": adjustments,
        "calibration_coverage_raw_by_horizon": coverage["raw"],
        "calibration_coverage_adjusted_by_horizon": coverage["calibrated"],
        "calibration_mean_width_raw_by_horizon": widths["raw"],
        "calibration_mean_width_adjusted_by_horizon": widths["calibrated"],
        "guarantee_note": "Empirical horizon-specific calibration only. Product/day dependence and prior exploratory exposure preclude a formal distribution-free guarantee for this retail panel.",
    }
    cal_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    calibrated_path = OUT / "calibration-intervals.csv"
    with calibrated_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "stock_code", "forecast_origin", "target_date", "horizon",
                  "context_sale_days", "actual", "p10", "p50", "p90",
                  "calibrated_lower", "calibrated_upper", "adjustment_units"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            qhat = adjustments[str(row["horizon"])]["adjustment_units"]
            writer.writerow({
                **{name: row[name] for name in fields if name in row},
                "calibrated_lower": min(row["p50"], max(0.0, row["p10"] - qhat)),
                "calibrated_upper": max(row["p50"], row["p90"] + qhat),
                "adjustment_units": qhat,
            })
    print(json.dumps(result, indent=2))
    print(f"Calibration corrections: {cal_path}")
    print(f"Calibration forecast intervals: {calibrated_path}")


if __name__ == "__main__":
    main()
