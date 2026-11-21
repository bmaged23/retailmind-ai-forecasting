# Engineered feature EDA and enriched validation

The [interactive feature EDA](https://bmaged23.github.io/retailmind-ai-forecasting/reports/feature-eda.html) and [notebook](../notebooks/06_engineered_feature_eda.ipynb) summarize two validation origins for 100 active and 100 intermittent products. These are 400 product/cutoff snapshots, not 400 independent products. The cohorts are selected and the dataset has already been explored; the plots do not represent all products or a blind final test.

The 400 snapshots had no missing 90-day historical price or customer-country information, but earlier windows and other products may be missing. The features include a missingness indicator for both. Historical quantity-weighted price and invoice activity use log1p; UK customer share is bounded; weekday and month use cyclical encodings. Country means customer destination, not product origin. Invoice activity is the sum of distinct invoices within each day over the prior 90 days (invoice-days), not necessarily unique invoices across the entire window. Returns are excluded consistently with the primary gross-sales target. Description and candidate code-family fields are experimental and excluded.

Each training example receives only transaction summaries available through the day before its target. During the recursive 28-day forecast, transaction summaries are frozen at the forecast cutoff; target calendar features are known in advance. An LSTM with a 90-day sales sequence and 32 hidden units receives seven calendar and five transaction summary features at the output head; sales-only LSTM remains a control. LightGBM gets its sales patterns, calendar and the same five transaction features. Target units are not truncated or modified. Both models reserve 28 pre-cutoff days for chronological early stopping. The exploratory validation runner preserves the original sales-only and calendar-only controls, applies common cohort and target masks, runs LightGBM on CPU and the LSTM on GPU for the recorded GPU runs.

Run after `uv sync --locked --extra dev --extra neural` and data preparation:

```sh
uv run --extra dev --extra neural python scripts/build_daily_transactions.py
uv run --extra dev --extra neural python scripts/feature_eda.py
uv run --extra dev --extra neural python scripts/feature_eda_report.py
uv run --extra dev --extra neural python scripts/run_validation.py --lstm --calendar --enriched --cohort active --device cuda --batch-size 32
uv run --extra dev --extra neural python scripts/run_validation.py --lstm --calendar --enriched --cohort intermittent --device cuda --batch-size 32
```

These enriched-feature results are validation only; their own final-test and calibration evaluations have not run. A separate Timestral interval study is documented in `milestone2-uncertainty.md`. Runtime is preliminary warm-inference timing without standardized warmup or parallelism. CPU peak RAM was not measured. The feature engineering choices were made after examining the historical dataset; they cannot be presented as fully blind model selection.

## GPU validation results

Both cohorts were evaluated at two 28-day origins using the same product/fold mask for each model. The active cohort scored 100 products in each fold; the intermittent cohort scored 88 and 85 because target windows for 12 and 15 products were incomplete. These exclusions can bias results towards products with longer observed histories. All scores are exploratory validation, not final test results.

| Model | Active MAE | Active WAPE | Intermittent MAE | Intermittent WAPE |
| --- | ---: | ---: | ---: | ---: |
| Seasonal naive | 42.93 | 127.78% | 8.54 | 153.14% |
| LightGBM, sales | 47.24 | 140.60% | 8.68 | 155.79% |
| LightGBM, sales + calendar | 40.53 | 120.62% | 8.48 | 152.09% |
| LightGBM, enriched | 40.97 | 121.96% | 9.65 | 173.17% |
| LSTM, sales | 30.07 | 89.50% | 6.15 | 110.29% |
| LSTM, enriched | 32.73 | 97.43% | 6.14 | 110.08% |

Enriched combines calendar, historical price, invoice-days and UK customer share; it did not improve active-cohort validation over the sales-only LSTM or calendar-only LightGBM in this run. The intermittent LSTM difference is small; these comparisons do not isolate which feature helped or hurt. The earlier baseline runs had different batch size and modeling settings, so compare results within this run rather than across prior runs.

The NVIDIA GTX 1660 Ti LSTM used batch size 32 and a PyTorch allocator cap of 1 GiB. Device-wide sampled peak memory was 166 MiB for both cohorts. Across individual LSTM training phases, mean sampled GPU utilization ranged from 17.3% to 27.7%, and the highest sampled utilization was 42%. Sampling was approximately every 0.25 seconds and includes CPU-side preparation within training phases; short peaks may be missed. The cap limits PyTorch's allocator, not necessarily all CUDA/driver allocations, so these samples do not prove an absolute hard VRAM ceiling. A device-wide 1,900 MiB watchdog was active. LightGBM ran on CPU.

The LSTM is one global model trained on mixed-product batches. Mean fold training time divided by 100 cohort products was approximately 0.207 seconds/product (sales-only active), 0.106 (enriched active), 0.153 (sales-only intermittent), and 0.149 (enriched intermittent). These are amortized figures, not separate per-product training runs; early stopping chose different epoch counts. Actual fold times, epoch counts, sampled GPU data and per-product warm forecast timings are in `reports/generated/validation-enriched-gpu/summary.json` and per-cohort metadata. Original run metadata mistakenly says no peak memory was measured; it means CPU peak RAM was not measured—the GPU device peak is sampled and recorded separately.
