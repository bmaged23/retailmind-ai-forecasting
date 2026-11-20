# Milestone 6: local deployment and MLOps foundation

## Scope

Milestone 6 packages a useful local forecasting service and adds repeatable
operational checks. The API accepts caller-supplied daily sales history for
demand forecasts, batch inference, holdout scoring, reorder proposals, and
inventory what-if simulations. The dashboard is an optional client of the API.
A separate monitor container checks local data freshness, forecast artifact
completeness, historical MAE, interval coverage, and per-product inference
latency. It fingerprints the data, forecast files, and active config for
reproducibility.

The source dataset stops in 2011 and there is no live sales or inventory feed.
The scheduled monitor therefore reports stale input and records historical
reference metrics. It deliberately does not generate a current forecast,
retrain a model, recalibrate intervals, or overwrite an accepted artifact.
Those actions require a validated current data pipeline first.

## Local Docker Compose

Prerequisites: Docker Engine/Desktop with Compose v2, and the prepared local
artifacts described in the README. From the repository root:

```sh
docker compose config
docker compose up --build -d
docker compose ps
```

Open <http://127.0.0.1:8000>. The API health check is
<http://127.0.0.1:8000/health>; readiness is
<http://127.0.0.1:8000/ready>; monitoring status is
<http://127.0.0.1:8000/api/ops/status>. API capabilities and interactive schema
are at <http://127.0.0.1:8000/api/v1/capabilities> and
<http://127.0.0.1:8000/docs>.

The request-driven API uses only the packaged Timestral 209M LoRA model. It
returns daily P10/P50/P90 forecasts and calibrated bounds, and exposes batch
forecasting, holdout evaluation, reorder proposals, and inventory scenarios.
The packaged adapter is from the earlier validation run, not the missing exact
October test-run checkpoint. The base weights are pinned and included in the
image; Compose enables GPU access when the host NVIDIA runtime is configured,
with a 2 GiB PyTorch allocator limit. See [Forecast API](forecast-api.md) for
the request contract and limits. When local historical artifacts are mounted,
the optional dashboard can still display the saved research comparisons.

To inspect the monitor once without Compose:

```sh
PYTHONPATH=src .venv/bin/python scripts/ops/monitor_daily.py --once
```

Compose keeps the source and historical inputs read-only in the containers.
Only the assistant audit database and monitoring snapshots use named writable
volumes. The API binds to `127.0.0.1`. The image includes the Timestral base
weights, adapter, license, and interval calibration metadata; source data,
historical forecasts, caches, and private artifacts remain outside the image.

Stop the local services with `docker compose down`. The named audit and
monitoring volumes remain available for the next run; `docker compose down -v`
removes them.

## Monitoring record

`scripts/ops/monitor_daily.py` writes `latest.json` and append-only `events.jsonl`
under `artifacts/milestone6/monitoring/`. Each snapshot records:

* latest source date, age, file size, and SHA-256 fingerprint;
* forecast row completeness and forecast file fingerprints for Timestral LoRA,
  the LSTM, and TimesFM 3.0 head-tuned;
* their historical MAE and mean inference seconds per product;
* the Timestral interval period, observed coverage, nominal coverage, and mean
  width;
* missing artifact warnings and the fact that no forecast/retraining action was
  triggered.

The monitor never treats old test metrics as current production measurements.
When a live feed arrives, replace the stale-data warning with rolling metrics on
newly observed labels, track failures and latency over time, and evaluate drift
before triggering candidate retraining.

## Reproducibility and release boundary

`Dockerfile`, `compose.yaml`, `.dockerignore`, the lockfile, and the monitor
establish a local reproducible runtime. `.github/workflows/docker-image-ci.yml`
builds the image and runs real API smoke tests on pull requests and pushes to
`main`; the workflow does not publish an image. Docker Hub publishing is
reserved for the final project release after Milestone 7. This workspace
currently has no Git repository or remote, so the workflow has not run in
GitHub Actions and no image has been pushed.

## Limitations and next work

This is a local operations prototype, not a cloud deployment. It has no
scheduled live forecasting because no current source exists. The service uses
SQLite for local audit persistence; production identity, managed storage,
backups, alerts, and access controls are not configured. Do not expose the
container beyond localhost until authentication and authorization are added.
See [Milestone 5](milestone5-dashboard.md) for assistant action boundaries.
Milestone 7 business evaluation remains the next project milestone.
