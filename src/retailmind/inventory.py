"""Transparent single-cycle inventory policy simulation for Milestone 3."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean


def load_inputs(root: Path, config: dict) -> tuple[list[str], dict, dict, dict]:
    """Load frozen-cohort forecasts and history available at the forecast origin."""
    cohort = json.loads((root / config["cohort_file"]).read_text(encoding="utf-8"))["codes"]
    codes = sorted(cohort)
    origin = date.fromisoformat(config["forecast_origin"])
    start = date.fromisoformat(config["evaluation_start"])
    end = date.fromisoformat(config["evaluation_end"])
    horizon = (end - start).days + 1
    if start != origin + timedelta(days=1) or horizon != 28:
        raise ValueError("Expected a contiguous 28-day horizon immediately after the forecast origin")

    forecast_path = root / config["forecast_file"]
    forecasts: dict[str, list[dict]] = defaultdict(list)
    forecast_keys = set()
    expected = {
        (code, (start + timedelta(days=step)).isoformat())
        for code in codes for step in range(horizon)
    }
    with forecast_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["stock_code"], row["target_date"])
            if key in forecast_keys:
                raise ValueError(f"Duplicate forecast row: {key}")
            forecast_keys.add(key)
            row["horizon"] = int(row["horizon"])
            for field in ("actual", "p50", "calibrated_lower", "calibrated_upper"):
                row[field] = float(row[field])
            row["context_sale_days"] = int(row["context_sale_days"])
            forecasts[row["stock_code"]].append(row)
    if forecast_keys != expected:
        raise ValueError(
            f"Forecast grid mismatch: {len(expected - forecast_keys)} missing, "
            f"{len(forecast_keys - expected)} extra rows"
        )
    for rows in forecasts.values():
        rows.sort(key=lambda row: row["horizon"])
        if [row["horizon"] for row in rows] != list(range(1, horizon + 1)):
            raise ValueError("Forecast horizon indices are incomplete or misordered")

    panel: dict[str, dict[date, float]] = defaultdict(dict)
    source_path = root / config["source_data"]
    with source_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            code = row["stock_code"]
            if code in forecasts and day <= origin:
                panel[code][day] = float(row["quantity"])
    context_days = int(config["context_days"])
    context_start = origin - timedelta(days=context_days - 1)
    histories = {}
    for code in codes:
        series = panel.get(code, {})
        values = [series.get(context_start + timedelta(days=i), 0.0) for i in range(context_days)]
        if any(value < 0 for value in values):
            raise ValueError(f"Negative historical sales in context for {code}")
        histories[code] = values
    return codes, forecasts, histories, {"origin": origin, "start": start, "end": end, "horizon": horizon}


def build_scenarios(codes: list[str], histories: dict[str, list[float]], config: dict) -> dict[str, dict]:
    """Create one repeatable stock/lead-time scenario per SKU, shared by policies."""
    rng = random.Random(int(config["scenario_seed"]))
    stock_range = config["initial_stock_coverage_days_range"]
    lead_range = config["supplier_lead_time_days_range"]
    in_transit_probability = float(config["incoming_order_probability"])
    transit_weeks = config["incoming_order_coverage_weeks_range"]
    transit_days = config["incoming_order_arrival_days_range"]
    scenarios = {}
    for code in codes:
        history = histories[code]
        daily_average = mean(history)
        starting_coverage = rng.uniform(float(stock_range[0]), float(stock_range[1]))
        on_hand = max(0, int(round(daily_average * starting_coverage)))
        lead = rng.randint(int(lead_range[0]), int(lead_range[1]))
        incoming = []
        if rng.random() < in_transit_probability:
            weeks = rng.uniform(float(transit_weeks[0]), float(transit_weeks[1]))
            quantity = max(0, int(round(daily_average * 7 * weeks)))
            if quantity > 0:
                incoming.append({
                    "arrival_day": rng.randint(int(transit_days[0]), int(transit_days[1])),
                    "quantity": quantity,
                })
        scenarios[code] = {
            "daily_average_90d": daily_average,
            "sale_days_90d": sum(value > 0 for value in history),
            "starting_on_hand": on_hand,
            "lead_time_days": lead,
            "incoming_orders": incoming,
        }
    return scenarios


def _policy_targets(history: list[float], forecasts: list[dict], protection_days: int,
                    safety_factor: float) -> dict[str, float]:
    model_rows = forecasts[:protection_days]
    p50_demand = sum(max(0.0, row["p50"]) for row in model_rows)
    p90_gap = sum(max(0.0, row["calibrated_upper"] - row["p50"]) for row in model_rows)
    seasonal_pattern = history[-7:]
    naive_demand = sum(seasonal_pattern[i % 7] for i in range(protection_days))
    return {
        "timestral_expected_demand": p50_demand,
        "timestral_safety_buffer": safety_factor * p90_gap,
        "timestral_target_stock": p50_demand + safety_factor * p90_gap,
        "seasonal_naive_expected_demand": naive_demand,
        "seasonal_naive_target_stock": naive_demand,
    }


def _replay(policy: str, code: str, product_rows: list[dict], scenario: dict,
            order_quantity: int, lead_days: int, costs: dict) -> tuple[dict, list[dict]]:
    on_hand = int(scenario["starting_on_hand"])
    arrivals = defaultdict(int)
    for incoming in scenario["incoming_orders"]:
        arrivals[int(incoming["arrival_day"]) - 1] += int(incoming["quantity"])
    if order_quantity > 0:
        arrivals[lead_days - 1] += order_quantity

    total_demand = total_fulfilled = total_unmet = stockout_days = 0.0
    holding_cost = shortage_cost = 0.0
    daily = []
    for day_index, row in enumerate(product_rows):
        incoming_today = arrivals[day_index]
        opening_stock = on_hand + incoming_today
        on_hand = opening_stock
        demand = max(0.0, row["actual"])
        fulfilled = min(on_hand, demand)
        unmet = demand - fulfilled
        on_hand -= fulfilled
        holding_cost_today = ((opening_stock + on_hand) / 2.0) * costs["holding_cost_per_unit_day"]
        shortage_cost_today = unmet * costs["shortage_penalty_per_unmet_unit"]
        holding_cost += holding_cost_today
        shortage_cost += shortage_cost_today
        total_demand += demand
        total_fulfilled += fulfilled
        total_unmet += unmet
        stockout_days += unmet > 0
        daily.append({
            "stock_code": code, "policy": policy, "target_date": row["target_date"],
            "demand_proxy": demand, "incoming_units": incoming_today,
            "opening_stock": opening_stock, "fulfilled_units": fulfilled,
            "unmet_units": unmet, "closing_stock": on_hand,
            "holding_cost_gbp": holding_cost_today,
            "shortage_cost_gbp": shortage_cost_today,
        })
    order_cost = costs["order_cost_per_order"] if order_quantity > 0 else 0.0
    summary = {
        "stock_code": code, "policy": policy, "demand_units": total_demand,
        "fulfilled_units": total_fulfilled, "unmet_units": total_unmet,
        "stockout_days": int(stockout_days), "ending_on_hand": on_hand,
        "new_order_units": order_quantity, "new_orders": int(order_quantity > 0),
        "holding_cost_gbp": holding_cost, "order_cost_gbp": order_cost,
        "shortage_cost_gbp": shortage_cost,
        "total_cost_gbp": holding_cost + order_cost + shortage_cost,
    }
    summary["fill_rate"] = (
        total_fulfilled / total_demand if total_demand > 0 else None
    )
    return summary, daily


def run_simulation(root: Path, config: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    codes, forecasts, histories, dates = load_inputs(root, config)
    scenarios = build_scenarios(codes, histories, config)
    cost_config = config["cost_assumptions_gbp"]
    costs = {
        "holding_cost_per_unit_day": float(cost_config["holding_cost_per_unit_day"]),
        "order_cost_per_order": float(cost_config["order_cost_per_order"]),
        "shortage_penalty_per_unmet_unit": float(cost_config["shortage_penalty_per_unmet_unit"]),
    }
    review_days = int(config["review_period_days"])
    safety_factor = float(config["forecast_safety_factor"])
    summaries, daily_rows, recommendations = [], [], []
    for code in codes:
        scenario = scenarios[code]
        rows = forecasts[code]
        lead = int(scenario["lead_time_days"])
        protection_days = min(lead + review_days, dates["horizon"])
        targets = _policy_targets(histories[code], rows, protection_days, safety_factor)
        inventory_position = scenario["starting_on_hand"] + sum(
            item["quantity"] for item in scenario["incoming_orders"]
        )
        # Prevent floating-point noise from zero forecasts creating one-unit orders.
        timestral_order = max(0, math.ceil(targets["timestral_target_stock"] - inventory_position - 1e-6))
        naive_order = max(0, math.ceil(targets["seasonal_naive_target_stock"] - inventory_position - 1e-6))
        incoming_units = sum(item["quantity"] for item in scenario["incoming_orders"])
        recommendations.append({
            "stock_code": code,
            "segment": "intermittent" if scenario["sale_days_90d"] <= 13 else "active",
            "sale_days_90d": scenario["sale_days_90d"],
            "average_daily_sales_90d": scenario["daily_average_90d"],
            "starting_on_hand": scenario["starting_on_hand"],
            "in_transit_units": incoming_units,
            "in_transit_count": len(scenario["incoming_orders"]),
            "in_transit_arrival_day": (
                scenario["incoming_orders"][0]["arrival_day"]
                if scenario["incoming_orders"] else ""
            ),
            "lead_time_days": lead,
            "review_period_days": review_days,
            "protection_period_days": protection_days,
            **targets,
            "inventory_position": inventory_position,
            "timestral_order_units": timestral_order,
            "seasonal_naive_order_units": naive_order,
            "timestral_arrival_date": (
                (dates["start"] + timedelta(days=lead - 1)).isoformat()
                if timestral_order > 0 else ""
            ),
            "seasonal_naive_arrival_date": (
                (dates["start"] + timedelta(days=lead - 1)).isoformat()
                if naive_order > 0 else ""
            ),
        })
        for policy, order in (("timestral_p50_plus_p90_buffer", timestral_order),
                              ("seasonal_naive_order_up_to", naive_order)):
            summary, ledger = _replay(policy, code, rows, scenario, order, lead, costs)
            summaries.append(summary)
            daily_rows.extend(ledger)
    result = {
        "model": "timestral209m_lora",
        "forecast_origin": dates["origin"].isoformat(),
        "evaluation_period": [dates["start"].isoformat(), dates["end"].isoformat()],
        "products": len(codes), "days": dates["horizon"],
        "scenario_seed": int(config["scenario_seed"]),
        "assumptions": config,
        "policies": {},
    }
    for policy in config["policies"]:
        subset = [row for row in summaries if row["policy"] == policy]
        demand = sum(row["demand_units"] for row in subset)
        fulfilled = sum(row["fulfilled_units"] for row in subset)
        result["policies"][policy] = {
            "products": len(subset),
            "demand_units": demand,
            "fulfilled_units": fulfilled,
            "unmet_units": sum(row["unmet_units"] for row in subset),
            "fill_rate": fulfilled / demand if demand else None,
            "products_with_stockout": sum(row["stockout_days"] > 0 for row in subset),
            "product_stockout_days": sum(row["stockout_days"] for row in subset),
            "total_new_order_units": sum(row["new_order_units"] for row in subset),
            "products_ordered": sum(row["new_orders"] for row in subset),
            "ending_inventory_units": sum(row["ending_on_hand"] for row in subset),
            "holding_cost_gbp": sum(row["holding_cost_gbp"] for row in subset),
            "order_cost_gbp": sum(row["order_cost_gbp"] for row in subset),
            "shortage_cost_gbp": sum(row["shortage_cost_gbp"] for row in subset),
            "total_cost_gbp": sum(row["total_cost_gbp"] for row in subset),
        }
    segment_results = {}
    for segment in ("intermittent", "active"):
        segment_results[segment] = {}
        segment_codes = {row["stock_code"] for row in recommendations if row["segment"] == segment}
        for policy in config["policies"]:
            subset = [row for row in summaries if row["policy"] == policy and row["stock_code"] in segment_codes]
            demand = sum(row["demand_units"] for row in subset)
            fulfilled = sum(row["fulfilled_units"] for row in subset)
            segment_results[segment][policy] = {
                "products": len(subset), "demand_units": demand,
                "unmet_units": sum(row["unmet_units"] for row in subset),
                "fill_rate": fulfilled / demand if demand else None,
                "products_with_stockout": sum(row["stockout_days"] > 0 for row in subset),
                "total_cost_gbp": sum(row["total_cost_gbp"] for row in subset),
            }
    result["segments"] = segment_results
    return result, summaries, daily_rows, recommendations
