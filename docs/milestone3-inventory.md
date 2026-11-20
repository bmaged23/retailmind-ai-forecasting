# Milestone 3: inventory policy simulation

## Goal and scope

This milestone turns the selected Timestral 209M LoRA forecasts into a
reproducible reorder-policy prototype. The comparison uses the same frozen
4,366-product cohort, November 12–December 9, 2011 recorded-sales window, and
seeded inventory scenarios for both policies. Forecasts originate on November
11, so the evaluated period is strictly after the forecast origin.

The source dataset has no on-hand inventory, supplier lead times, purchase
orders, procurement costs, or censored lost-demand records. Starting stock,
orders in transit, lead times, and costs are therefore explicit synthetic
assumptions. Recorded sales are only a demand proxy. Results must not be
presented as measured business savings or service levels.

## Policies

At the forecast origin, each SKU receives at most one order. Inventory position
is starting on-hand plus existing orders in transit. The protection period is
sampled supplier lead time plus a seven-day review period.

* **Timestral P50 + P90 buffer:** target stock is the sum of nonnegative P50
  demand over the protection period plus the sum of daily calibrated P90 minus
  P50 gaps. The gap multiplier is 1.0. This is a practical safety-stock
  heuristic; adding marginal daily P90 values does not produce a calibrated
  aggregate P90 forecast.
* **Seasonal-naive order-up-to:** repeat the last seven observed daily sales
  values over the protection period and order up to their sum, without a safety
  buffer.

The simulator replays each day's recorded sales, fulfills demand from available
stock, processes arrivals, and tracks unmet units, ending stock, fill rate,
stockout days, and assumed holding, order-placement, and shortage costs. Neither
policy reforecasts during the 28-day replay.

## Scenario and cost assumptions

The reproducible defaults are in [`configs/milestone3_inventory.json`](../configs/milestone3_inventory.json):

* 90 days of history to estimate average sales and demand segment.
* Initial stock uniformly sampled from 3–14 days of 90-day average sales.
* Supplier lead time uniformly sampled from 2–10 days.
* A 25% chance of an existing inbound order, sized at 0.5–1.5 weeks of
  average sales, arriving in 1–14 days.
* Holding cost £0.05 per unit per day, order placement £5 per new order, and
  shortage penalty £2 per unmet recorded unit. Purchase cost is excluded.
* Fixed scenario seed 42. Both policies see identical SKU scenarios and the
  exact same recorded-sales paths.

These are demonstration values, not estimates derived from retailer accounts.
Cost rankings can change when these assumptions change, especially the
shortage penalty and initial stock. A robust business decision needs real
inventory, replenishment, and cost data plus multiple periods and sensitivity
analysis.

Order quantities ignore forecast tails below 1e-6 units when rounding to whole
units, preventing floating-point noise in effectively zero forecasts from
creating spurious one-unit orders. The figures below include this correction.

## Result for this configured scenario

| Policy | Fill rate | Unmet recorded units | Products with stockout | Simulated cost |
|---|---:|---:|---:|---:|
| Timestral P50 + P90 buffer | 75.33% | 128,252 | 1,458 | £647,247.80 |
| Seasonal-naive order-up-to | 46.35% | 278,873 | 1,992 | £690,862.03 |

Timestral's policy placed more and larger orders, leaving 178,816 units on hand
at the end versus 36,700 for seasonal naive. Its assumed holding cost was
£379,713.80 versus £126,831.03, while its shortage penalty was lower. The
£43,614.23 simulated cost difference is consequently conditional on the chosen
cost weights and scenario; it is not realized savings. Existing order arrival
and policy order quantities are modeled at daily granularity.

Demand segments are defined using the 90-day context: active products have at
least 14 sale days; intermittent products have at most 13.

## Reproduce and inspect

From the repository root, run:

```sh
PYTHONPATH=src python3 scripts/simulate_inventory.py
```

The run writes a shareable dashboard at
[`reports/milestone3-inventory.html`](../reports/milestone3-inventory.html),
plus a JSON scorecard and product-level recommendations and daily ledger under
`artifacts/milestone3_inventory/timestral209m_lora/`. Artifacts and source data
are excluded from Git; the configuration and simulator are tracked.

## Limitations and next work

This is a single historical 28-day replay using a fixed cohort and one synthetic
scenario realization. Sales observed during stockouts can understate demand.
There are no measured availability or service-level labels. The result is
sensitive to stock, lead-time, and cost assumptions, and the summed daily
forecast interval is not a calibrated protection-period distribution. Before
using the recommendations operationally, obtain real stock and purchase-order
history, validate demand censoring and lead times, run sensitivity analysis
over multiple origins, and evaluate a rolling replenishment policy.
