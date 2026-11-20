# LSTM feature selection: exploratory validation

Run `uv run --extra dev --extra neural python scripts/run_lstm_ablation.py` on a CUDA-enabled host with the cohort and daily transaction files already prepared. Then run `uv run --extra dev --extra neural python scripts/summarize_lstm_ablation.py`. The runner uses two validation origins, a 28-day forecast, 90-day input, batch size 64, one 32-unit LSTM layer, Adam at 0.001, a maximum of 20 epochs, chronological early stopping with patience 3 and seed 42. The test and calibration periods were not used for evaluation. Historical data was previously explored; these are not blind final test results.

The initial three variants were sales-only; sales plus cyclic weekday/month; and sales plus weekday/month and historical sales-frequency measures. Within each cohort, the lowest pooled validation MAE of those three was then selected and historical 90-day weighted price plus a missingness indicator added. This adaptive price comparison reuses the validation periods and is exploratory; a final test must occur only after the model and choices are frozen. Price and frequency are frozen at the forecast origin throughout recursive prediction. Weekday/month are known for the future target dates. No future actual sales or future realized prices are inputs.

| Cohort | Variant | Pooled MAE | WAPE | Mean training seconds per fold |
| --- | --- | ---: | ---: | ---: |
| Active | Sales | 29.994 | 89.28% | 12.34 |
| Active | Sales + calendar | 31.254 | 93.03% | 7.93 |
| Active | Sales + calendar + frequency | 31.117 | 92.62% | 7.95 |
| Active | Sales + historical price | 29.156 | 86.78% | 10.41 |
| Intermittent | Sales | 6.159 | 110.50% | 9.14 |
| Intermittent | Sales + calendar | 6.230 | 111.79% | 9.07 |
| Intermittent | Sales + calendar + frequency | 6.229 | 111.76% | 10.41 |
| Intermittent | Sales + historical price | 6.162 | 110.56% | 8.45 |

In the active cohort, historical price reduced pooled MAE by about 2.8% relative to sales-only on these two validation periods. The price variant had lower per-product MAE for 54/100 and 74/100 products in the two folds. The intermittent difference was approximately 0.003 units per day, too small to interpret as a meaningful improvement. It scored only 88/100 and 85/100 intermittent products because the remaining target windows were missing; exclusions can bias the comparison. One seed and two origins are insufficient to establish robustness. Price may reflect discounts, order size and returns; this is predictive association, not a causal effect.

Sixteen global LSTM fits took a sum of 151.4 seconds of measured training time, including per-fit training-window preparation but excluding separate scoring and overall setup. Each fit trained from windows across 100 products; batches mix examples across products. Mean per-fit time divided by 100 gives 0.104 seconds/product for active sales+price and 0.084 seconds/product for intermittent sales+price; these are amortized measures, not independent model training per product. Early stopping produced different epoch counts between variants, so times are not a controlled architecture speed benchmark.

GPU: NVIDIA GTX 1660 Ti; batch 64; sampled device-wide mean utilization across the overall run 22.6%, sampled peak utilization 42%, sampled peak device memory 160 MiB. PyTorch peak reserved memory was 48 MiB, with a 1 GiB PyTorch allocator cap and 1,900 MiB device watchdog. Device-wide samples include non-training phases and can miss short spikes; the PyTorch allocator cap does not include every CUDA/driver allocation, so it does not prove a hard VRAM bound. CPU peak RAM was not measured. Full per-run epochs, samples, training times, exclusions, forecast outputs and source hashes are in `reports/generated/lstm-feature-ablation/`.
