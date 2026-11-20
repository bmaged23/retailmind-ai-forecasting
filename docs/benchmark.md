# Forecasting benchmark plan

Status: paired validation model selection and the Milestone 2 interval study
are complete as historical experiments. No pristine final benchmark remains.

## Results already produced

The current report is `artifacts/foundation_models/test_2011-10-05_to_2011-11-01/foundation_model_evaluation.html`,
with machine-readable metrics in the adjacent `scorecard.json`. It scores 15
forecast variants on 2,693 products from the October 5–November 1, 2011 test
window, using a 90-day context and October 4 forecast origin. It includes
seasonal naive, LightGBM, LSTM, and zero-shot and adapted foundation-model
forecasts. The user requested reuse of this previously evaluated split, so
these scores are not a blind final benchmark and should not be described as
unbiased test estimates. The saved product/date forecasts and run metadata are
local generated artifacts and are excluded from Git.

On this reused split, Timestral 209M LoRA has the lowest MAE and WAPE, and the
LSTM has the lowest RMSE. Output-head adaptation improved TimesFM 3.0 MAE and
WAPE modestly but worsened RMSE; TiRex-2 changes were negligible. Fine-tuning
methods and license constraints are documented in
`scripts/foundation_models/README.md`.

1. Validate source workbook, sheet row counts, date coverage and cleaning audit; record a dataset hash and define a fixed product cohort without using test outcomes.
2. Freeze non-overlapping chronological training, validation, calibration and final test roles. Use common forecast origins and a 28-day horizon, initially with 90 days of context. Select and tune only on training/validation.
3. Compare the seasonal-naive control, LightGBM and global LSTM trained only on each fold's training data, and the five foundation models as zero-shot adapters. Evaluate supported fine-tuning as a distinct track. Validate model and weight licenses before downloading or distribution.
4. Preserve product/date-level forecasts and evaluate MAE, RMSE, WAPE and MASE, including per-SKU and per-horizon breakdowns and intermittent-demand segments. Clearly mark undefined metrics. Compare models with identical inputs and distinguish covariate-enhanced experiments.
5. Record hardware, runtime environment, model revision, seed, batch size, warm-up, loading time, forecast latency, memory usage and errors. Measure on the same hardware and report the measurement protocol.
6. Freeze the final test evaluation after model selection, document failures, data vintage, potential pretrained-model contamination and sales-censoring limitations. Timestral 209M LoRA is the July validation leader; no unbiased final winner has been established.

## Validation selection status

The baseline training run records LightGBM's validation-selected iteration and
the LSTM's validation-selected epoch, but `artifacts/train-validation-test-2011-11-01/metrics.json`
contains only October test forecast rows and test metrics. It does not contain
September product-level validation forecasts. Foundation-model metadata records
some checkpoint-selection validation losses, but not common validation
forecasts for the full 2,693-product cohort. Those losses are model-specific
training objectives and are not comparable across architectures. The small
Chronos-2 validation pilot is also not a full-cohort paired evaluation.

The common July comparison and selected validation leader are documented in
`docs/validation-comparison.md`. Timestral 209M LoRA has the lowest pooled MAE
on that window. October scores remain descriptive because the period was
reused. The Milestone 2 calibration and later-period coverage study is in
`docs/milestone2-uncertainty.md`; it does not establish an unbiased final
performance claim. A new untouched dataset would be required for that.

The stricter rolling-origin protocol below remains a proposed follow-up. It
must use a fresh untouched test period; the comparison above does not satisfy
that requirement.

The current baseline CSV records per-series inference time in seconds using a monotonic performance clock. It excludes process/model startup, input loading and metric calculation, and is not yet a complete latency benchmark. Current cohort filtering requires complete histories and targets across all supplied cutoffs; it uses future availability and can introduce survivorship bias. Replace it with a cohort fixed before validation/test and report missing targets before treating results as final.
