# Milestone 4: stress-test model-driven inventory plans

## Models and fair comparison window

This comparison uses the three requested forecast candidates:

* Timestral 209M LoRA
* Trained LSTM
* Google TimesFM 3.0 head-tuned

All three saved forecasts contain exactly the same 75,404 product-date keys:
2,693 products over October 5–November 1, 2011, forecast from October 4. The
models were trained with targets through August 31 and validated through
September 28; the test target values were not loaded for their fit. This window
has been used in earlier exploratory/model-evaluation work, so this is a
descriptive comparison, not a pristine blind benchmark.

The October evaluation uses the same prepared recorded-sales targets for all
three plans. The forecast MAE values in this project output are 5.781 for
Timestral, 5.924 for LSTM, and 6.009 for TimesFM 3.0. This keeps the
inventory-policy comparison paired on the exact same cohort and dates.

## Turning forecasts into comparable order plans

Each model receives the same opening inventory, incoming orders, and supplier
lead time for a SKU. Its order-up-to target is calculated with one shared rule:

1. Sum the model's nonnegative forecast over the sampled lead time plus a
   seven-day review period.
2. Add a common 20% safety buffer.
3. Subtract on-hand stock and the full quantity of existing in-transit orders;
   order the positive remainder, rounded up to whole units.

The 20% buffer is a transparent shared assumption, not model-specific
probabilistic calibration. We use it because the LSTM and TimesFM3 artifacts
for this window contain point forecasts, while Timestral's P10/P50/P90
intervals were generated for a different origin. This compares point-forecast
plans under the same ordering rule without borrowing one model's interval for
the other models.

Each resulting plan places at most one new order at the origin. There is no
reforecast or replenishment during the 28-day simulation.

## Paired stress scenarios

The reproducible configuration is
[`configs/milestone4_scenarios.json`](../configs/milestone4_scenarios.json).
It defines 500 futures with seed `20261118`:

* **Demand:** the same observed daily sales path for all three plans, scaled by
  one horizon-wide multiplier: 1.0 (70%), 1.25 (20%), or 1.5 (10%).
* **Supplier delay:** add 0 (70%), 2 (20%), 5 (8%), or 10 (2%) days to every
  existing and newly recommended delivery.
* **Starting-stock loss:** remove 0% (70%), 10% (15%), 25% (10%), or 50% (5%)
  of starting on-hand inventory. In-transit stock is not damaged.

All three plans share each scenario's demand path, delay, stock loss, initial
stock, inbound orders, lead time, and costs. Deliveries arrive before sales on
their arrival day.

## Results

| Model-specific plan | Forecast MAE | Products ordered | Order units | Mean fill rate | 5th–95th fill range | Mean unmet recorded units | Mean modeled cost | 95th percentile cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Timestral 209M LoRA | 5.781 | 428 | 22,802 | 27.56% | 19.86%–31.06% | 397,807 | £845,108 | £1,213,175 |
| Trained LSTM | 5.924 | 1,097 | 28,788 | 28.60% | 21.01%–32.17% | 392,258 | £838,815 | £1,206,727 |
| Google TimesFM 3.0 head-tuned | 6.009 | 393 | 16,030 | 26.26% | 18.62%–29.65% | 404,760 | £857,559 | £1,226,017 |

LSTM had the highest fill rate and lowest modeled cost in this configured
simulation, while Timestral had the lowest forecast MAE. LSTM ordered 5,986
more units than Timestral, which helps explain its service advantage. TimesFM3
ordered the least and had the lowest mean fill. The differences are small
relative to the assumed costs and simulated conditions; they do not establish
that LSTM is the best operational model. Mean demand was 545,315 recorded units
across the simulated 28-day portfolio path.

The low fill rates reflect the one-order policy, the relatively small
model-based order quantities for this 2,693-product set, and the selected
historical period. The simulation does not place a second order when the first
order is consumed. That limitation is part of the current policy prototype.

## Cost assumptions and limitations

The inherited cost assumptions are £0.05 per unit per day held, £5 per order,
and £2 per unmet recorded unit. Purchase cost is excluded. The stock position,
supplier lead times, incoming orders, and cost rates are synthetic; none are
observed retailer data. The model cannot recover sales lost during real
stockouts, so recorded sales may understate demand.

The test window was reused earlier, the demand path comes from one historical
period, and the three stress factors are simple scenario distributions rather
than estimates from retailer operations. A common 20% safety buffer does not
calibrate each model's uncertainty. The 500 futures are suitable for a
prototype comparison, not precise tail-risk estimation or a business savings
claim. A stronger next pass needs untouched forecast origins, real stock and
purchase-order data, empirically estimated disruption rates, and rolling
replenishment.

## Reproduce and inspect

From the repository root:

```sh
PYTHONPATH=src .venv/bin/python scripts/simulate_scenarios.py
```

Open [`reports/milestone4-scenarios.html`](https://bmaged23.github.io/retailmind-ai-forecasting/reports/milestone4-scenarios.html).
Scorecard, per-SKU order plans, scenario outcomes, daily unmet units, and
stress-factor tables are generated under
`artifacts/milestone4_scenarios/model_plan_comparison/`.
