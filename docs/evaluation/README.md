# Model evaluation

Compare every candidate on the same product and date keys, then inspect accuracy and recorded inference time.

## Protocol

4,366 products, 28 days, and 122,248 product-day predictions per model. Training through April 2011; May checkpoint validation; July 2011 model-selection window. Forecasts were frozen before labels were scored.

## Results

Timestral 209M LoRA led pooled MAE at 2.331; Chronos-2 LoRA followed at 2.346. TimesFM 2.5 LoRA had the lowest RMSE at 18.971. July is a selection window, not an untouched final test.

## Reproduce

See scripts/foundation_models/prepare_validation_cohort.py, the model runners, and score_validation_comparison.py. Raw data and large forecast artifacts are excluded from Git; follow data/README.md and docs/validation-comparison.md.

## Limitations

Latency varies by batch size and execution path. The packaged adapter is from the earlier validation run; the exact October checkpoint was not retained. Historical recorded sales may understate demand during stockouts.

[Detailed reference](../validation-comparison.md) · [Visual guide](../../reports/evaluation.html) · [Project README](../../README.md)

## HTML dashboards and source

- [Interactive common-cohort model comparison](../../reports/comparison/validation-comparison.html) · [scorecard JSON](../../reports/comparison/scorecard.json).
- [Evaluation overview](../../reports/evaluation.html) · [comparison protocol](../validation-comparison.md).
- [Uncertainty and interval coverage](../../reports/milestone2-uncertainty.html) · [methodology](../milestone2-uncertainty.md).
- [Inventory simulation](../../reports/milestone3-inventory.html) · [methodology](../milestone3-inventory.md).
- [Scenario analysis](../../reports/milestone4-scenarios.html) · [methodology](../milestone4-scenarios.md).
- [Evaluation and model-runner scripts](../../scripts/foundation_models/) · [all reports](../../reports/README.md).

Download the repository and open HTML reports in a browser; GitHub displays them as source. Inventory outcomes are synthetic simulations, not observed business savings.
