# Foundation model comparison

These runners live separately from the task-specific LightGBM and LSTM training
code. Each runner writes forecasts first and never joins test labels. The scoring
step runs only after all zero-shot and eligible fine-tuned forecasts are saved.

## Frozen comparison

- Forecast origin: 2011-10-04
- Test targets: 2011-10-05 through 2011-11-01
- Context/horizon: 90/28 days
- Same fixed cohort as the previous LightGBM/LSTM report: 2,693 products
- Fine-tuning fit targets end 2011-08-31; validation targets end 2011-09-28
- Zero-shot and adapted forecasts are scored against the same saved actuals

The user asked to reuse the previous test set. This is a paired experimental
comparison, not a newly untouched final benchmark. The test labels are not inputs
to fine-tuning, checkpoint selection, or inference.

## Current runner

`chronos2_zero_shot_and_lora.py` runs Amazon Chronos-2 zero-shot and LoRA
fine-tuning with a 256-series seeded validation sample. Its forecasts, weights,
and run metadata are stored under
`artifacts/foundation_models/test_2011-10-05_to_2011-11-01/` (ignored by Git).

`timesfm3_head_finetune.py` and `tirex2_head_finetune.py` perform local,
experimental output-head adaptation for TimesFM 3.0 and TiRex-2. They freeze the
pretrained backbone, train on all eligible series, choose the head using a
seeded validation sample with early stopping, then write forecasts on the
shared test cohort. These are explicitly not vendor-provided LoRA workflows.

## Evaluation snapshot

TimesFM 3.0 and TiRex-2 now have zero-shot and experimental output-head-tuned
results on the reused October test window:

| Model / method | MAE | RMSE | WAPE | Inference sec/product |
|---|---:|---:|---:|---:|
| TimesFM 3.0 zero-shot | 6.059 | 38.313 | 92.29% | 0.150 |
| TimesFM 3.0 output-head tuned | 6.009 | 38.456 | 91.52% | 0.204 |
| TiRex-2 zero-shot | 6.078 | 38.635 | 92.58% | 0.040 |
| TiRex-2 output-head tuned | 6.077 | 38.719 | 92.57% | 0.040 |

The adaptation improved TimesFM 3.0 MAE by 0.050 and WAPE by 0.76 percentage
points, while RMSE increased by 0.143. TiRex-2 changes were negligible: MAE
improved by 0.001 and WAPE by 0.015 points, while RMSE increased by 0.084.
These runs train only the final forecasting head; the pretrained backbones stay
frozen. Each used all 4,366 eligible training series per epoch, a seeded
256-series validation sample, a 96-day context, a 28-day horizon, batch size 1,
and early stopping with patience 2. The GPU cap was 1,800 MiB. TimesFM 3.0
trained for 42.9 minutes and used 1,494 MiB peak sampled device memory; TiRex-2
trained for 10.1 minutes and used 1,170 MiB. These are experimental local
adaptations, not official LoRA workflows.

Across all models, Timestral 209M LoRA still has the lowest MAE and WAPE
(5.781 and 88.05%); the LSTM still has the lowest RMSE (37.719). All results
reuse the previously evaluated test split and are not a fresh untouched
benchmark.

The interactive comparison and full scorecard are generated at
`artifacts/foundation_models/test_2011-10-05_to_2011-11-01/foundation_model_evaluation.html`
and `scorecard.json`. TimesFM 3.0 weights carry a non-commercial license, so
check the license terms before any commercial use or redistribution.

## Capability notes

| Model | Zero-shot | Fine-tuning path available here | Notes |
|---|---|---|---|
| Chronos-2 | Yes | Yes, LoRA | Official `Chronos2Pipeline.fit`; use a small batch under the GPU cap |
| TimesFM 3.0 | Yes | Experimental output-head tuning | Frozen backbone; non-commercial evaluation only; do not distribute tuned weights |
| Timestral 209M | Yes | Forge service | Local public repository documents inference/covariates; Forge access is external |
| Toto 2.0 22M | Yes | No | The official repository says 2.0 fine-tuning is not yet available; Toto 1.0 is a separate fine-tunable model |
| TiRex-2 | Yes | Experimental output-head tuning | Vendor fine-tuning is a TiRex-2 PRO feature; local run uses the open Apache-2.0 PyTorch model |

Do not label a different version (TimesFM 2.5 or Toto 1.0) as a fine-tuned
version of TimesFM 3.0 or Toto 2.0. Fine-tuning results are measured; improvement
is not assumed.

## Published evaluation reports

The tracked [common-cohort comparison HTML](https://bmaged23.github.io/retailmind-ai-forecasting/reports/comparison/validation-comparison.html) and [scorecard](../../reports/comparison/scorecard.json) summarize the July selection window. See the [evaluation README](../../docs/evaluation/README.md), [uncertainty dashboard](https://bmaged23.github.io/retailmind-ai-forecasting/reports/milestone2-uncertainty.html), [inventory simulation](https://bmaged23.github.io/retailmind-ai-forecasting/reports/milestone3-inventory.html), and [scenario analysis](https://bmaged23.github.io/retailmind-ai-forecasting/reports/milestone4-scenarios.html). The HTML guides open directly in your browser. The October evaluation described above reused a test period and is not an untouched benchmark.
