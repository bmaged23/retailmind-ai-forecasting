# Milestone 2: probabilistic forecasts

## Goal

Turn the selected Timestral 209M LoRA point forecast into daily P10/P50/P90
forecasts and calibrate the central 80% interval. The existing Timestral
checkpoint returns quantiles; this milestone calibrates their coverage rather
than training a new model.

## Frozen chronological roles

| Role | Dates | Use |
| --- | --- | --- |
| Adapter fit | Through 2011-04-30 | Existing Timestral LoRA fit |
| Checkpoint validation | 2011-05-01–2011-05-28 | Existing early stopping |
| Model selection | 2011-07-01–2011-07-28 | Select Timestral as the MAE leader |
| Interval calibration | 2011-08-01–2011-08-28 | Estimate per-horizon interval adjustment |
| Later-period evaluation | 2011-11-12–2011-12-09 | Measure coverage, width, and P50 error |

All forecasts use the same April-frozen 4,366-product cohort, a 90-day context,
and a 28-day horizon. The calibration forecast origin is July 31; the later
evaluation origin is November 11. The adapter is frozen for both runs. Each
forecast process reads only sales available through its origin. The calibration
runner checks the complete product/date grid before loading August labels. The
evaluation runner checks its forecast grid and reads the later labels only
after the calibration corrections have been saved.

The evaluation window is a later diagnostic, not a pristine blind benchmark:
earlier project work examined some dates and subsets in these periods. The
source ends on 2011-12-09, and foundation-model pretraining may include public
retail data. Report empirical coverage without claiming a formal guarantee or
current-retail performance.

## Calibration method

For each forecast horizon separately, calculate the conformalized quantile
regression score across calibration products:

```text
score = max(P10 - actual, actual - P90)
```

Use the finite-sample 80th percentile (rank `ceil((n + 1) * 0.8)`) as that
horizon's signed adjustment. A positive value widens the base interval; a
negative value narrows it. Apply it as `lower=min(P50, max(0, P10-adjustment))`
and `upper=max(P50, P90+adjustment)`. P50 is unchanged, and the final bounds
remain nonnegative and ordered around the median.

The correction is pooled across products separately for each horizon. Products
and dates are dependent, demand levels vary widely, and zero recorded sales do
not prove zero unconstrained demand. Therefore the observed coverage is an
empirical diagnostic, not a distribution-free SKU-level guarantee. The report
breaks results out by horizon and by intermittent/active context (13 or fewer
versus at least 14 sale days in the 90-day context).

## Results

Timestral's raw central 80% interval already covered 93.95% on average across
the 28 August calibration horizons (range: 90.82%–97.98%). The finite-sample
signed conformal adjustment was zero at every horizon, due to the calibration
score distribution and ties at zero. Thus the calibrated intervals remained
unchanged; on the later period, their observed coverage was 91.53%, not the
nominal 80%.

| Later-period metric | Result |
| --- | ---: |
| Product-day rows | 122,248 |
| P50 MAE | 3.534 units |
| P50 RMSE | 22.864 units |
| P10–P90 coverage | 91.53% |
| Mean interval width | 10.159 units |
| Mean quantile pinball loss | 1.327 |
| Coverage by horizon | 85.39%–96.34% |

The interval coverage is higher than the nominal 80%. This is useful evidence
that the current intervals are conservative on these historical windows; it
does not establish per-product coverage or production reliability. On the later
period, intermittent products had 95.04% coverage with 0.47-unit mean width;
active products had 84.69% coverage with 29.07-unit mean width.

GPU inference used the GTX 1660 Ti, batch size 16, PyTorch's `cudaMallocAsync`
allocator backend, and a device-wide 1,900 MiB watchdog. The sampled peak during
forecasting was 946 MiB (832 MiB PyTorch reserved). Loading is monitored via the
PyTorch reserved-memory peak before the device watchdog starts; short peaks
between device samples could be missed, so 946 MiB is an observed peak, not a
hard hardware guarantee.

## Run workflow

```sh
uv run --extra timestral python scripts/uncertainty/generate_timestral_interval_forecasts.py calibration --device cuda
uv run --extra timestral python scripts/uncertainty/calibrate_timestral_intervals.py
uv run --extra timestral python scripts/uncertainty/generate_timestral_interval_forecasts.py evaluation --device cuda
uv run --extra timestral python scripts/uncertainty/evaluate_timestral_intervals.py
```

Outputs are local under
`artifacts/milestone2_uncertainty/timestral209m_lora/`:

- `calibration.json`: frozen per-horizon interval adjustments and calibration coverage.
- `calibration-intervals.csv`: calibration forecasts, actuals, and adjusted bounds.
- `evaluation-intervals.csv`: per-product/day P10/P50/P90, calibrated lower/upper, and actuals.
- `scorecard.json`: overall, horizon-level, and demand-segment metrics.
- `milestone2-uncertainty-report.html`: standalone summary with portfolio chart and coverage tables.
- `reports/milestone2-uncertainty.html`: shareable copy of the standalone report for the project repository.

The Timestral base weights are cached under the project's ignored `.cache/`
directory, not in source control. The generated adapter and forecast artifacts
are also excluded from Git. Review the base model and adapter license terms
before publishing or deploying.

[Open the interactive uncertainty report](https://bmaged23.github.io/retailmind-ai-forecasting/reports/milestone2-uncertainty.html).
