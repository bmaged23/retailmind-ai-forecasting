# Model-specific feature and truncation experiments

Status: proposed follow-up experiments. The EDA and core model comparison are
complete; any feature or truncation selection must use validation data. The
October 2011 comparison split has already been reused and must not guide further
feature selection.

Each candidate model may use a different supported feature set. Compare experiments within a model using identical cohort, cutoffs, target masks, forecast horizons and data availability. Keep the univariate track for cross-model comparison; report enriched tracks separately so a model is not credited for information another model could not use. Record exact feature definitions, code/data/config hashes, hardware, latency, memory and failures. Track effects by product segment, not just pooled error. The full historical dashboard was inspected before the split, so the final periods are not completely blind.

Suggested feature ablations, selected on validation only:

1. Sales-only control.
2. Add known calendar features.
3. Add product description features that were available by each cutoff: audited literal tags or a text representation fitted on training data only. Missing description is distinct from a negative tag. Candidate pack size and code-family grouping need further independent review before use.
4. Add historical sales/price and buyer-country summaries, measured at each cutoff. Future realized price, customer identity and future transaction records are unavailable. Country is the buyer's country, not manufacture origin; description does not establish quality.
5. Compare combinations only after testing each family, so we can identify whether it helped. Keep unsupported features out of a model rather than silently changing the comparison.

Two distinct candidate meanings of truncation are awaiting user clarification:

- **History length:** compare 56, 90 and 180 days of input history, for models whose interfaces support those contexts. Use a fixed comparison mask that is valid for the longest context and report coverage. Record each model's separate training-window policy and inference latency.
- **Extreme values:** consider a model-input or training-target cap derived only from that fold's training history, and explicitly record the threshold and inversion policy. Never cap held-out actuals to improve reported metrics. Report uncapped evaluation metrics and sensitivity to cancellation handling; avoid erasing real bulk orders. A cap changes the modeling problem and requires careful interpretation.

Use chronological internal early stopping and validation only for selecting feature sets and truncation settings. Freeze those choices before the final test evaluation. The earlier test-period data was already explored descriptively, so consider an additional truly untouched dataset for a stronger final assessment. Enforce the user's 2 GB GPU-memory ceiling for both training and inference; use CPU or report infeasible when the ceiling cannot be guaranteed.
