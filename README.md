# RetailMind AI · Demand Forecasting & Inventory Decisions

**A practical way to turn retail sales history into demand forecasts and inventory decisions.**

Retailers need to decide what to replenish before the next delivery arrives. Ordering too little can leave shelves empty; ordering too much can tie up cash and warehouse space. RetailMind is a forecasting and decision-support prototype that helps a planner examine those trade-offs for each product.

The service takes **consecutive daily sales quantities for a product** and predicts the quantity expected to be sold on each of the next **1–28 days**. It returns a central forecast and a range of plausible outcomes. A planner can combine those forecasts with stock on hand, incoming stock, supplier lead time and review period to calculate a suggested replenishment quantity, then explore what could happen if demand rises or a delivery is late. The suggestion supports a human decision; it does not place an order.

![RetailMind animated production journey](docs/images/production-flow.gif)

[Download the shareable MP4](docs/images/retailmind-production.mp4) · [Download the GIF](docs/images/production-flow.gif)

### How the solution works

1. **Prepare the history.** Clean retail transactions and aggregate recorded sales into daily product series. The API accepts a caller's daily quantities; it does not require the original dataset at inference time.
2. **Forecast quantities.** A fine-tuned Timestral 209M model estimates future daily sales for the requested horizon.
3. **Express uncertainty.** P10, P50 and P90 forecasts and historically calibrated daily bounds show a range around the central estimate.
4. **Plan replenishment.** Inventory rules combine the forecast with on-hand and incoming stock, lead time, review period and a safety buffer to produce a proposed order.
5. **Explore scenarios.** Simulate demand changes and delivery delays to inspect projected fulfillment and unmet units before acting.

For technical readers, the project compares **15 model variants on 4,366 products**, exposes the selected model through FastAPI, packages it in Docker with a 2 GiB GPU allocator cap, and runs real inference and decision checks in GitHub Actions. The historical data measures **recorded sales**, not demand lost during stockouts; results and inventory simulations should be validated with current operational data before business use.

## Data & exploration

The research uses historical transaction data ending in **2011**. Transactions were cleaned and aggregated into daily product sales; cancellations, non-product codes, missing dates and price features were investigated. A frozen cohort of **4,366 products** was used for the common model comparison. A zero in the series represents zero *recorded sales*, which can include stockouts or unavailable products.

Explore the interactive [data dashboard](reports/data-exploration.html), [feature analysis](reports/feature-eda.html), and [product-code audit](reports/code-family-audit.html). See [data setup](data/README.md) and [feature inventory](docs/feature-inventory.md) for the inputs and transformations. Download the repository and open HTML files in a browser; GitHub displays HTML as source.

## Evaluation & model comparison

**15 model variants · 4,366 products · 28 forecast days · 122,248 scored product-days per candidate**

The common model-selection window is **1–28 July 2011**. Forecasts were frozen before scoring on the same product-date keys. Timestral 209M LoRA led on pooled MAE at **2.331**, narrowly ahead of Chronos-2 LoRA at **2.346**—roughly a **0.6%** difference. TimesFM 2.5 LoRA had the lowest RMSE, showing how model choice depends on the metric.

![Actual daily recorded sales compared with four model forecasts](docs/images/forecast-actual.svg)

*Daily portfolio totals on the common selection cohort. The model lines are saved predictions, and the white line is recorded sales.*

![Daily actual values, Timestral median forecast and uncertainty interval for product 21976](docs/images/forecast-interval.svg)

*Illustrative product 21976, 12 November–9 December 2011. Shading shows the saved calibrated daily bounds; it is not a calibrated interval for the sum of the 28 days.*

![Model comparison by pooled MAE](docs/images/model-comparison.svg)

| Model | MAE ↓ | RMSE ↓ | WAPE ↓ | Mean inference ms/product* |
|---|---:|---:|---:|---:|
| Timestral 209M LoRA | 2.331 | 19.488 | 91.6% | 11.44 |
| Chronos-2 LoRA | 2.346 | 19.551 | 92.2% | 6.90 |
| Chronos-2 zero-shot | 2.363 | 19.490 | 92.9% | 2.93 |
| Toto 2.0 custom LoRA | 2.363 | 19.662 | 92.9% | 2.35 |
| Timestral 209M zero-shot | 2.365 | 19.473 | 93.0% | 7.27 |
| TiRex-2 head-tuned | 2.403 | 19.723 | 94.5% | 26.39 |
| TiRex-2 zero-shot | 2.409 | 19.666 | 94.7% | 27.77 |
| TimesFM 2.5 zero-shot | 2.410 | 19.457 | 94.7% | 12.58 |
| Toto 2.0 zero-shot | 2.429 | 19.590 | 95.5% | 1.82 |
| TimesFM 3.0 head-tuned | 2.511 | 20.209 | 98.7% | 181.88 |
| TimesFM 3.0 zero-shot | 2.553 | 20.291 | 100.4% | 151.13 |
| LSTM | 2.562 | 19.018 | 100.7% | 19.14 |
| TimesFM 2.5 LoRA | 3.063 | 18.971 | 120.4% | 18.86 |
| Seasonal naive | 3.435 | 24.310 | 135.0% | 0.00 |
| LightGBM | 5.110 | 19.707 | 200.9% | 2.08 |


*Latency is the recorded batch-run time divided by products, not a controlled single-request comparison. Models used different batch sizes and execution paths.*

**Evaluation limits:** July selected the model; it is not an untouched final test. The October window was reused. The packaged LoRA adapter comes from the earlier validation run; the exact October checkpoint was not retained, so October scores cannot be attributed to the deployed checkpoint. See the [interactive comparison](reports/comparison/validation-comparison.html), [evaluation guide](reports/evaluation.html), [methodology](docs/validation-comparison.md), [scorecard](reports/comparison/scorecard.json), and [evaluation scripts](scripts/foundation_models/).

## Forecast API

FastAPI provides single and batch forecasts, P10/P50/P90 and calibrated bounds, chronological holdout evaluation, reorder proposals and supply scenarios. OpenAPI documentation is served at `/docs`. [API visual guide](reports/api.html) · [API reference](docs/forecast-api.md)

## Docker & deployment

The image packages the pinned Timestral base weights, fine-tuned adapter and API. [Docker Hub repository](https://hub.docker.com/r/bmaged23/retailmind-ai-forecasting). A local container passed real inference, interval, evaluation and inventory smoke checks. GPU inference is configured with a 2 GiB PyTorch allocator cap. [Docker guide](reports/docker.html) · [Deployment details](docs/docker/README.md)

```bash
# Pull the tested image, or build it from source.
docker pull bmaged23/retailmind-ai-forecasting:latest
docker build -t retailmind-ai-forecasting:local .
docker run --rm -p 8000:8000 retailmind-ai-forecasting:local
# API docs: http://localhost:8000/docs
```

Add `--gpus all` when the NVIDIA container runtime is available. Review the [model license](models/timestral209m_lora/LICENSE) before distributing the image.

## CI/CD

GitHub Actions builds the image, starts the service and runs a real-model API smoke test. A successful run on `main` publishes the tested image when the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets are configured; otherwise publication is skipped with a warning. [CI/CD guide](reports/cicd.html) · [Workflow](.github/workflows/docker-image-ci.yml)

## More analysis

[Uncertainty report](reports/milestone2-uncertainty.html) · [Inventory simulation](reports/milestone3-inventory.html) · [Scenario analysis](reports/milestone4-scenarios.html)
