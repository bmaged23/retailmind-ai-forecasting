# Project code map

- `src/retailmind/workbook.py`: shared, streaming two-sheet workbook reader and schema checks.
- `src/retailmind/data.py`: transaction cleaning and daily product sales construction.
- `src/retailmind/product_features.py`: as-of product metadata and literal title indicators; exploratory only.
- `src/retailmind/features.py`: historical demand and calendar feature functions.
- `src/retailmind/cancellations.py`: conservative cancellation matching for sensitivity analysis; no model target rewriting.
- `src/retailmind/benchmark.py`: common validation folds, forecasts and missing-data reporting.
- `src/retailmind/forecast.py`: preliminary seasonal-naive model and metrics; its current cohort selection is exploratory and must be replaced.
- `src/retailmind/lightgbm_model.py`: pooled recursive LightGBM prototype, not yet integrated into benchmark.
- `src/retailmind/cli.py`: commands for preprocessing and exploratory backtesting.
- `src/retailmind/dashboard_api.py` and `apps/dashboard/index.html`: local decision API and responsive dashboard; scenario/recommendation tools are proposal-only and assistant calls are audited in SQLite.
- `src/retailmind/model_catalog.py`: shared candidate identifiers and display names for the dashboard and scenario reports.
- `scripts/ops/monitor_daily.py` and `configs/milestone6_ops.json`: scheduled input-freshness, artifact-completeness, metric-reference, latency, and checksum snapshots.
- `Dockerfile` and `compose.yaml`: loopback-only local API and monitor deployment; historical data and model outputs are mounted read-only.
- `scripts/foundation_models/`: model-specific zero-shot and experimental fine-tuning runners, shared frozen-cohort scoring, and comparison report generation. Run outputs/checkpoints live under ignored `artifacts/foundation_models/`.
- `scripts/`: reproducible analysis entry points for exploration, validation and audit.
- `notebooks/`: executed exploratory notebook; the dashboard generator is in `scripts/data_exploration.py`.
- `configs/`: proposed evaluation design and constraints.
- `tests/`: behavior and leakage tests.
- `data/` and `reports/generated/`: local source data, processed data and generated measurements; excluded from Git.

The implementation combines a small task-specific package with separate
foundation-model runners because candidate forecasting libraries have different
interfaces and training capabilities. The cancellation analysis streams the
workbook but stores candidate events in memory; do not claim constant-memory
processing for that stage.
