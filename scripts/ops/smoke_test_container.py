"""Exercise the built container's forecast, interval, evaluation, and action APIs."""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen


def call(base_url: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}", data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="GET" if body is None else "POST",
    )
    with urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return json.load(response)


def wait_until_ready(base_url: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = call(base_url, "/ready")
            if result.get("status") == "ready":
                return result
        except (URLError, TimeoutError, RuntimeError, ConnectionError) as exc:
            last_error = exc
        time.sleep(3)
    raise TimeoutError(f"API did not become ready within {timeout}s: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--startup-timeout", type=int, default=300)
    args = parser.parse_args()

    ready = wait_until_ready(args.base_url, args.startup_timeout)
    assert ready.get("model", {}).get("id") == "timestral209m_lora", ready
    capabilities = call(args.base_url, "/api/v1/capabilities")
    assert list(capabilities["forecast_methods"]) == ["timestral209m_lora"], capabilities

    start = date(2025, 1, 1)
    history = [
        {"date": (start + timedelta(days=index)).isoformat(),
         "quantity": float((index % 7) + (3 if index % 11 == 0 else 0))}
        for index in range(90)
    ]
    series = {"series_id": "ci-smoke-sku", "history": history}
    response = call(args.base_url, "/api/v1/forecast", {
        "series": series, "horizon_days": 7,
    })
    method = response["methods"]["timestral209m_lora"]
    predictions = method["predictions"]
    assert len(predictions) == 7, predictions
    for row in predictions:
        assert 0 <= row["p10"] <= row["p50"] <= row["p90"], row
        assert 0 <= row["calibrated_lower"] <= row["p50"] <= row["calibrated_upper"], row

    proposal = call(args.base_url, "/api/v1/inventory/proposal", {
        "forecast": [row["p50"] for row in predictions],
        "upper_forecast": [row["calibrated_upper"] for row in predictions],
        "on_hand_units": 5,
        "in_transit_units": 2,
        "lead_time_days": 3,
        "review_period_days": 4,
        "safety_stock_fraction": 0.1,
    })
    assert proposal["proposal_only"] is True
    assert proposal["proposed_order_units"] >= 0
    assert proposal["interval_safety_buffer_units"] >= 0

    scenario = call(args.base_url, "/api/v1/inventory/scenario", {
        "forecast": [row["p50"] for row in predictions],
        "starting_stock": 5,
        "inbound_units": 4,
        "inbound_arrival_day": 2,
        "demand_multiplier": 1.2,
        "extra_supply_delay_days": 1,
    })
    assert len(scenario["daily"]) == 7
    assert scenario["proposal_only"] is True

    evaluation = call(args.base_url, "/api/v1/evaluate", {
        "series": series, "holdout_days": 7,
    })
    metrics = evaluation["results"]["timestral209m_lora"]
    assert metrics["mae"] >= 0 and metrics["rmse"] >= 0

    print(json.dumps({
        "status": "passed",
        "model": ready["model"]["id"],
        "forecast_days": len(predictions),
        "mean_daily_p50": method["mean_daily_demand_units"],
        "inference_seconds": method["inference_seconds"],
        "proposed_order_units": proposal["proposed_order_units"],
        "scenario_fill_rate": scenario["fill_rate"],
        "holdout_mae": metrics["mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
