# Feature inventory

Status: EDA and feature engineering in progress. No forecasting model training or evaluation should run until the user directs it after review.

| Feature | Meaning and timing | Current model use |
| --- | --- | --- |
| Daily sales and lags 1, 7, 28 | Gross recorded units observed before each forecast origin | LSTM uses historical quantities; LightGBM uses lags |
| Rolling averages and standard deviation | 7/28-day means and 28-day standard deviation of past daily units | LightGBM |
| Sale frequency and time since sale | Positive-day fraction over 7/28 days; days since most recent positive sales day, capped at 28 | LightGBM; recursive predictions greater than zero are treated as positive, a limitation |
| Calendar | Known target weekday and month as cyclic values; meteorological season | Calendar-enhanced LightGBM only |
| Bank holidays | England-and-Wales bank-holiday flag and days to next one (capped) | Calendar-enhanced LightGBM only; proxy for UK retailer, not every customer's holiday |
| Historical unit price | Quantity-weighted mean over 90 days ending at cutoff; unavailable if no qualifying sales | Extracted as-of; not yet used in a model |
| Invoice count | Distinct sales invoices involving product within trailing 90 days; invoice numbers not treated as numeric | Extracted as-of; not yet used in a model |
| Customer geography | Share of trailing sales units delivered to UK customers; not product origin | Extracted as-of; not yet used in a model |
| Product code | Categorical identifier and candidate formatting/family signals | Code identifies series; family and case normalization unverified and not model features |
| Description keywords and candidate pack size | Literal word flags and explicit counts, not verified categories or quality | Experimental only; not connected to a model |

`product_features.snapshot(..., require_observed=True)` uses qualifying transactions on or before its cutoff. A missing price or customer share remains missing. Missing description word flags remain missing. `scripts/materialize_metadata.py` generates the two validation-date snapshots for each initial cohort, without fitting a model. The retrospective full catalogue is for EDA only and must not define eligible products for a historical forecast.

Code case was investigated in the [code-family audit notebook](../notebooks/05_code_family_audit.ipynb) and [interactive audit](../reports/code-family-audit.html). Descriptive analysis finds 170 normalized groups with original codes differing only by case in the full history, 119 with sales for multiple variants on the same date. These observations do not establish identity or authorize automatic merging.
