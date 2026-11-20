# Cancellation sensitivity analysis

Run `uv run python scripts/analyze_cancellations.py` after preparing the UCI workbook. The workbook reader streams rows; matching retains candidate events in memory. The JSON report is written to `reports/generated/cancellation-sensitivity.json` and contains aggregate counts, not customer IDs. The raw data and generated outputs are excluded from Git.

A sale and cancellation are tentatively paired only when they share stock code, nonmissing customer ID, country, exact price and absolute quantity; the negative cancellation invoice must follow within 24 hours; and the candidate pairing must be unique in both directions. Ambiguous pairs are excluded. This is a conservative heuristic, not proof of invoice linkage. Partial returns, longer delays, different prices, missing customer identifiers and multiple identical candidates are not matched. Unit tests cover ambiguous and late cases.

The sensitivity calculation subtracts matched quantities from the original sale day solely to compare descriptive gross and retrospectively adjusted totals and peaks. It must **not** be used as a model-ready training or evaluation target: a cancellation after a forecast cutoff was not known at the cutoff, so retrospective subtraction could leak future information. A production-quality target needs an explicit as-of data policy and decisions about demand versus fulfilled sales. The current gross target is retained unchanged until that policy is defined.

## Initial full-workbook result

The run found 1,190 mutually unique candidate pairs totaling 205,068 units, or 1.80% of retained gross units; 1,096 pairs occurred on the same calendar day. Retrospectively subtracting them reduces the largest product-day total from 80,995 to 19,152 units. These are sensitivity figures under the stated heuristic, not verified cancellations or a revised target. Missing customer IDs and conservative matching mean the figures are not an estimate of all cancellations.
