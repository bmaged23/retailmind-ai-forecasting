# RetailMind AI — Retail Demand Forecasting

**GitHub repository:** [github.com/bmaged23/retailmind-ai-forecasting](https://github.com/bmaged23/retailmind-ai-forecasting)

RetailMind turns a product's daily sales history into forecasts and inventory decision support. It is a research prototype for planners who want to explore how much stock might be needed, understand uncertainty, and assess potential supplier delays or demand changes.

This image packages the FastAPI service, the **Timestral 209M base model**, the project's fine-tuned **LoRA adapter**, and historical interval calibration. It supports Linux `amd64` and includes the model files needed for inference.

## What the service does

- Forecasts daily recorded sales quantities for **1–28 days** from consecutive daily product history.
- Returns P10, P50 and P90 quantiles and calibrated daily bounds.
- Supports a chronological holdout evaluation with MAE, RMSE, WAPE and MASE when defined.
- Suggests a replenishment quantity from forecasts, stock on hand, incoming stock, lead time, review period and safety assumptions.
- Simulates demand changes and supplier delays. Proposals are advisory: the service never places a purchase order.

## Pull and run

```bash
docker pull bmaged23/retailmind-ai-forecasting:latest
docker run --rm -p 8000:8000 bmaged23/retailmind-ai-forecasting:latest
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation, or request `GET http://localhost:8000/ready` and `GET http://localhost:8000/api/v1/capabilities`. Model loading occurs when inference is first requested.

For an NVIDIA GPU with the NVIDIA container runtime installed, add `--gpus all` to `docker run`. The application is configured with a **2,048 MiB PyTorch GPU allocator limit** and can use CPU when CUDA is unavailable. The image is large because it includes PyTorch, CUDA dependencies and model weights; allow additional space while pulling and unpacking it.

## Forecast example

The API requires at least seven consecutive daily observations. The following request forecasts the next seven days:

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H 'Content-Type: application/json' \
  -d '{
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
    "horizon_days": 7
  }'
```

Other endpoints include `POST /api/v1/forecast/batch`, `/api/v1/evaluate`, `/api/v1/inventory/proposal`, and `/api/v1/inventory/scenario`. See the [full API reference](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/docs/forecast-api.md) for request details.

## Research and evaluation

The project compared **15 candidate variants** on the same **4,366 products** and **28-day** July 2011 selection window. Timestral LoRA led pooled MAE at **2.331** units per product per day; Chronos-2 LoRA followed at **2.346**. The selection was close, and TimesFM 2.5 LoRA had lower RMSE.

July was used to select the model, so it is not an untouched final benchmark. The October window was reused, and the exact October adapter checkpoint was not retained. The adapter in this image comes from the earlier validation run; October scores should not be attributed to it. See the [evaluation methodology](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/docs/validation-comparison.md), [scorecard](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/reports/comparison/scorecard.json), and [report index](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/reports/README.md).

Historical transactions measure **recorded sales**, which may be lower than true demand during stockouts. Inventory and supplier inputs in the research simulations are synthetic; those results are not measured business savings. Validate performance and calibration on current operational data before use.

## Source, reports and licensing

- [Source code and project README](https://github.com/bmaged23/retailmind-ai-forecasting)
- [Docker and deployment guide](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/docs/docker/README.md)
- [CI/CD workflow](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/.github/workflows/docker-image-ci.yml)
- [Model license](https://github.com/bmaged23/retailmind-ai-forecasting/blob/main/models/timestral209m_lora/LICENSE): review its conditions, including the monthly revenue condition, before distribution or commercial use.

The existing image was built and smoke-tested locally and published to Docker Hub. GitHub Actions also builds and tests the image; automatic publication requires the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.
