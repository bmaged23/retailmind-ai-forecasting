# Milestone 5: business dashboard and assistant MVP

## What it provides

The local dashboard combines the three-model forecast comparison, order
proposals, per-product inventory projections, the Timestral interval study, and
what-if simulation tools. It is served from a single FastAPI process with a
responsive browser interface; it does not require a separate frontend build.

For this MVP, the forecast and inventory data are the common October 5–November
1, 2011 comparison window (2,693 SKUs, 28 days). The Timestral uncertainty
panel comes from its separate November 12–December 9 interval evaluation and is
labelled as a different origin. It is not presented as uncertainty for the
October model forecasts.

The API exposes a product lookup, product detail, a constrained scenario tool,
and an order-recalculation tool. The assistant recognizes explanation,
scenario, and recommendation requests, obtains numbers from these structured
tools, and returns the computed values with the assumptions. Its local intent
router is deterministic; no generative LLM is connected in this MVP, and the
assistant does not invent missing metrics.

## Run locally

Install all project extras once, then start the API from the repository root:

```sh
uv sync --all-extras
uv run --no-sync uvicorn retailmind.dashboard_api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The API schema is at
<http://127.0.0.1:8000/docs>; the health endpoint is
<http://127.0.0.1:8000/health>.

`--host 127.0.0.1` keeps this development app on the local machine. Do not
expose it to a network without adding authentication and access controls.

## Safe actions and audit trail

* Scenario inputs are bounded by request validation. They recalculate one
  product's projection for a demand multiplier, supplier delay, and starting
  stock loss.
* Order recalculation changes the safety-stock fraction and returns a proposal.
  It cannot create a purchase order or contact a supplier.
* There is no external purchasing capability. Any future purchase integration
  must add authenticated roles, explicit confirmation, and an auditable order
  record before sending anything to a supplier.
* Assistant tool requests and their structured results are logged in the local
  SQLite database at `artifacts/milestone5/audit.sqlite3`. This captures the
  request, selected tool, timestamp, model version label, and result. Artifacts
  are ignored by Git; logs remain local.

SQLite is used for the local audit trail to avoid requiring a database server.
PostgreSQL and identity management belong in the production/MLOps milestone.

## Limitations

The dashboard reproduces historical experiments, not live forecasts. Sales
are not uncensored demand, inventory and supplier values are synthetic, and the
October model-evaluation window was previously reused. The local assistant uses
fixed intent patterns rather than a language model; connect an approved LLM
provider only after selecting a provider and credentials, while keeping the
same restricted tools and confirmation boundary. Timestral's interval plot is
from another historical period. No data is refreshed automatically.

## Project map

* UI: [`apps/dashboard/index.html`](../apps/dashboard/index.html)
* API and tool routing: [`src/retailmind/dashboard_api.py`](../src/retailmind/dashboard_api.py)
* Scenario and recommendation assumptions: [`configs/milestone4_scenarios.json`](../configs/milestone4_scenarios.json)
* Model-plan study: [`docs/milestone4-scenarios.md`](milestone4-scenarios.md)
