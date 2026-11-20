# Leakage-safe validation comparison

## Protocol

Use two chronological windows before the previously reused October test. Fit
through April 30, 2011 and use May 1–28 labels for fine-tuning checkpoint
selection and early stopping. Make the common model-selection forecast from an
origin of June 30 for July 1–28. No July target may be read during fitting,
checkpoint selection, feature construction, or prediction. This gap lets an
adapted model use May for tuning while preserving July as a separate selection
window.

Every candidate receives the same 90 calendar days of sales ending at its
forecast origin and predicts 28 daily gross-sales values. Freeze the product
cohort using information through April 30: a product must already have a
recorded sale by then. Build its 90-day context through June 30, filling dates
without transactions with zero recorded sales. For July labels, also score
missing transaction dates as zero recorded sales; this cannot distinguish zero
demand from unavailable or discontinued products. Load selection labels only
after forecasts for every model have been written. Every
candidate must have forecasts for the exact same product-date keys; otherwise
stop and report failures instead of silently comparing different cohorts.

Compare seasonal naive, LightGBM, LSTM, and each foundation model as zero-shot
and, where technically supported, adapted variants. Existing adapters tuned
with September data cannot be reused; adapted variants must be retrained using
the April/May tuning dates. Models that cannot run under the project's 2 GiB
GPU cap will be marked unsupported with the reason and left outside the scored
ranking.

## Scoring and selection

Use pooled MAE as the primary selection metric. Report RMSE and WAPE as
guardrails, and per-product latency separately with model load time and peak GPU
memory. Clamp negative forecasts to zero. Record the exact model revision, fit
window, checkpoint choice, batch size, software versions, runtime and failures.

The October 5–November 1 period was already reused and is excluded from model
choice. Keep its scores descriptive. After freezing the July selection, evaluate
the selected candidate once on a genuinely untouched later window if available;
otherwise state clearly that no unbiased final test remains.

## Results

The comparison includes all 15 configured candidates: zero-shot, fine-tuned, and
classical baselines. Every model receives the same 90-day observed window ending
June 30 and is scored on the same 4,366 products and 28 July dates (122,248
product-day rows per model). Model-specific implementations may use only part of
the window or left-pad it for patch alignment, but no candidate receives extra
observed days. The scorer validates every forecast key before reading July
labels. By pooled MAE,
`timestral209m_lora` is the selected candidate on this validation window. Its
RMSE and WAPE are also competitive, but the October period has already been
reused and cannot serve as an unbiased final test.

| Rank | Model | MAE | RMSE | WAPE | Inference sec/product |
|---:|---|---:|---:|---:|---:|
| 1 | timestral209m_lora | 2.3314 | 19.4880 | 91.65% | 0.011439 |
| 2 | chronos2_lora | 2.3456 | 19.5513 | 92.21% | 0.006903 |
| 3 | chronos2_zero_shot | 2.3633 | 19.4903 | 92.90% | 0.002926 |
| 4 | toto2_22m_custom_lora | 2.3634 | 19.6624 | 92.90% | 0.002346 |
| 5 | timestral209m_zero_shot | 2.3649 | 19.4727 | 92.96% | 0.007267 |
| 6 | tirex2_head_tuned | 2.4033 | 19.7228 | 94.47% | 0.026389 |
| 7 | tirex2_zero_shot | 2.4086 | 19.6659 | 94.68% | 0.027765 |
| 8 | timesfm25_zero_shot | 2.4096 | 19.4570 | 94.72% | 0.012577 |
| 9 | toto2_22m_zero_shot | 2.4285 | 19.5896 | 95.46% | 0.001822 |
| 10 | timesfm3_head_tuned | 2.5111 | 20.2086 | 98.71% | 0.181880 |
| 11 | timesfm3_zero_shot | 2.5531 | 20.2910 | 100.36% | 0.151134 |
| 12 | lstm | 2.5618 | 19.0183 | 100.70% | 0.019143 |
| 13 | timesfm25_lora | 3.0629 | 18.9711 | 120.40% | 0.018856 |
| 14 | seasonal_naive | 3.4350 | 24.3101 | 135.03% | 0.000004 |
| 15 | lightgbm | 5.1099 | 19.7075 | 200.87% | 0.002079 |

Foundation-model fine-tuning used fit histories through April 30 and a fixed,
seeded sample of 256 May histories for checkpoint selection and early stopping.
The classical LSTM used its recorded May validation workflow. TiRex-2 and
TimesFM 3 adaptations are experimental output-head tuning with frozen
backbones; Toto 2.0 and Timestral use custom LoRA training paths, not official
vendor fine-tuning recipes. See the per-model JSON metadata for epoch histories,
batch sizes, checkpoint losses, inference times, and sampled GPU memory.

The HTML report includes the ranked scorecard and a selectable portfolio-total
chart comparing all model forecasts with recorded July sales. The chart is
cohort-wide; the CSV forecasts contain the per-product detail.

## Runner workflow

1. Freeze the cohort once with `uv run python
   scripts/foundation_models/prepare_validation_cohort.py`. The current source
   panel produces 4,366 eligible products. The generated manifest records the
   source hash and code list without storing target values.
2. Retrain each trainable candidate with fit data through April 30 and May
   checkpoint validation. Zero-shot candidates need no fit. Each forecast
   runner writes one CSV to
   `artifacts/validation_comparison_2011-07-01_to_2011-07-28/forecasts/`, named
   `<candidate>.csv`, with `model,stock_code,target_date,horizon,prediction,
   inference_seconds_per_series` columns. Batch sizes and model revisions are
   recorded in per-model metadata.
3. Run `uv run python
   scripts/foundation_models/score_validation_comparison.py`. It refuses to
   read labels if any configured forecast file is missing, rejects duplicate
   or mismatched product-date keys, and only then loads the panel to produce a
   common scorecard at `artifacts/validation_comparison_2011-07-01_to_2011-07-28/scorecard.json`
   and dashboard at `artifacts/validation_comparison_2011-07-01_to_2011-07-28/validation-comparison.html`.
