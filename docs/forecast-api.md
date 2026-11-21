# RetailMind Timestral forecasting API

The container packages one forecast model: the Timestral 209M LoRA adapter
available from the earlier validation run, with its pinned pretrained base
revision. It returns a 28-day maximum daily demand forecast with P10/P50/P90
quantiles and saved horizon-wise interval adjustments. The October benchmark
selected the same model family and adaptation method, but its exact adapter
checkpoint was not retained. Do not attribute the October test scores to the
adapter packaged here. See `models/timestral209m_lora/README.md`.

The API also provides batched forecasts, chronological holdout evaluation,
inventory reorder proposals, and inventory scenario simulations. These are
decision-support calculations only; they never submit purchase orders.

## Run

```sh
docker compose up --build -d
```

The service binds to `127.0.0.1:8000`. Interactive API documentation is at
`/docs`, and `GET /api/v1/capabilities` describes the model and operations. The
image includes the model base and adapter; it does not need a Hugging Face
download at container startup. Compose enables GPU access when the NVIDIA
Container Toolkit is configured. Inference is limited to 2,048 MiB of PyTorch
GPU allocator memory, and automatically uses CPU if CUDA is unavailable.

## Forecast

`POST /api/v1/forecast` accepts one ordered daily series. Include zero quantity
for days without recorded sales; missing dates are rejected.

```json
{
  "series": {
    "series_id": "SKU-1001",
    "history": [
      {"date": "2025-01-01", "quantity": 4},
      {"date": "2025-01-02", "quantity": 0},
      {"date": "2025-01-03", "quantity": 7},
      {"date": "2025-01-04", "quantity": 2},
      {"date": "2025-01-05", "quantity": 3},
      {"date": "2025-01-06", "quantity": 5},
      {"date": "2025-01-07", "quantity": 1}
    ]
  },
  "horizon_days": 28
}
```

Use at least seven consecutive daily observations and request no more than 28
forecast days. Each predicted day includes `demand_units` (P50), P10/P50/P90,
and `calibrated_lower`/`calibrated_upper`. The saved adjustments were estimated
on the project's 2011 historical calibration window. Summing daily bounds does
not produce a calibrated aggregate interval for the whole horizon.

## Other operations

* `POST /api/v1/forecast/batch`: forecast up to 25 series with the same
  Timestral model.
* `POST /api/v1/evaluate`: forecast a chronological holdout of up to 28 days and
  report MAE, RMSE, WAPE, MASE, and inference time.
* `POST /api/v1/inventory/proposal`: calculate demand over a lead plus review
  period, apply a safety-stock fraction, and return proposed order units.
  Pass the forecast's `p50` series as `forecast` and its
  `calibrated_upper` (or `p90`) series as `upper_forecast` to include the
  model interval buffer in the recommendation.
* `POST /api/v1/inventory/scenario`: simulate fulfilled and unmet units under a
  demand multiplier, inbound supply, and supplier delay.

Forecasts estimate recorded sales; stockouts can censor observed demand. The
adapter and Mistral base weights use the included Modified MIT license, which
has a monthly revenue condition. Review `models/timestral209m_lora/LICENSE`
before publishing or distributing the Docker image.

## Visual guide

Open the [API HTML guide](https://bmaged23.github.io/retailmind-ai-forecasting/reports/api.html) in a browser, and see the [report index](../reports/README.md) for evaluation, uncertainty, inventory and deployment dashboards.
