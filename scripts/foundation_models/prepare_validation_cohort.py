"""Freeze the leakage-safe July 2011 model-selection product cohort."""
from datetime import date
import hashlib
import json
from pathlib import Path

import csv

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/validation_comparison.json"
PANEL = ROOT / "data/processed/daily_sales.csv"
OUT = ROOT / "artifacts/validation_comparison_2011-07-01_to_2011-07-28"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoff = date.fromisoformat(config["tuning"]["fit_targets_through"])
    origin = date.fromisoformat(config["selection"]["forecast_origin"])
    context_days = int(config["context_days"])
    # Parse quantities only for rows up to the tuning cutoff. Later target
    # values are not loaded while the cohort is selected.
    eligible = set()
    with PANEL.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            if day <= cutoff and float(row["quantity"]) > 0:
                eligible.add(row["stock_code"])
    codes = sorted(eligible)
    if not codes:
        raise RuntimeError("The pre-tuning cohort is empty")

    # Materialize no targets here. The bounds simply document the context
    # each model runner must construct, using zero for dates without a row.
    context_start = origin.fromordinal(origin.toordinal() - context_days + 1)
    hasher = hashlib.sha256()
    with PANEL.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cohort.json").write_text(
        json.dumps(
            {
                "config": "configs/validation_comparison.json",
                "sales_panel_sha256": hasher.hexdigest(),
                "cohort_rule": config["cohort_rule"],
                "product_count": len(codes),
                "codes": codes,
                "context_start": context_start.isoformat(),
                "context_end": origin.isoformat(),
                "forecast_origin": origin.isoformat(),
                "selection_period": [
                    config["selection"]["target_start"],
                    config["selection"]["target_end"],
                ],
                "selection_targets_loaded": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Frozen {len(codes):,} products in {OUT / 'cohort.json'}")


if __name__ == "__main__":
    main()
