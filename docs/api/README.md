# Forecasting API

Turn daily sales history into forecasts, intervals, evaluation metrics, and inventory decision support.

## Forecast

POST /api/v1/forecast takes consecutive daily quantities and returns P10, P50, P90 and calibrated bounds for up to 28 days. Batch forecasting is available at /api/v1/forecast/batch.

## Evaluate

POST /api/v1/evaluate scores a chronological holdout using MAE, RMSE, WAPE and MASE where defined.

## Decide

POST /api/v1/inventory/proposal suggests an order quantity; /api/v1/inventory/scenario simulates demand and supply conditions. These are proposals and do not place orders.

## Validate

The local Docker smoke test passed a real seven-day forecast, interval checks, a proposal, a scenario, and holdout evaluation. Interactive OpenAPI documentation is served at /docs.

[Detailed reference](../forecast-api.md) · [Visual guide](../../reports/api.html) · [Project README](../../README.md)
