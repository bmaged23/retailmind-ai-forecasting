"""Paired inventory-plan stress test for three frozen forecasting models."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np
from retailmind.model_catalog import LABELS, LSTM, MODELS, TIMESFM3, TIMESTRAL


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_inputs(root: Path, config: dict) -> tuple[list[str], dict, dict, dict, dict]:
    origin = date.fromisoformat(config["forecast_origin"])
    start = date.fromisoformat(config["evaluation_start"])
    end = date.fromisoformat(config["evaluation_end"])
    days = (end - start).days + 1
    if start != origin + timedelta(days=1) or days != int(config["horizon_days"]):
        raise ValueError("Configured origin and 28-day evaluation window are not contiguous")

    forecasts: dict[str, dict[str, list[dict]]] = {}
    keys_by_model = {}
    for model in MODELS:
        path = root / config["forecast_files"][model]
        grouped: dict[str, list[dict]] = defaultdict(list)
        keys = set()
        for row in _rows(path):
            if row.get("model") and row["model"] != model:
                continue
            key = (row["stock_code"], row["target_date"])
            if key in keys:
                raise ValueError(f"Duplicate {model} forecast key {key}")
            keys.add(key)
            row["prediction"] = max(0.0, float(row["prediction"]))
            row["horizon"] = int(row["horizon"])
            grouped[row["stock_code"]].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda item: item["horizon"])
            if len(rows) != days or [r["horizon"] for r in rows] != list(range(1, days + 1)):
                raise ValueError(f"Incomplete or invalid 28-day forecast for {model}")
        forecasts[model] = grouped
        keys_by_model[model] = keys
    first_keys = keys_by_model[MODELS[0]]
    if any(keys_by_model[model] != first_keys for model in MODELS[1:]):
        raise ValueError("Model forecasts do not share identical product-date keys")
    codes = sorted(forecasts[TIMESTRAL])
    if len(codes) != int(config["expected_products"]):
        raise ValueError(f"Expected {config['expected_products']} common products, got {len(codes)}")

    actual_panel: dict[str, dict[date, float]] = defaultdict(dict)
    history_path = root / config["source_data"]
    with history_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = row["stock_code"]
            if code in forecasts[TIMESTRAL]:
                actual_panel[code][date.fromisoformat(row["date"])] = float(row["quantity"])
    actuals, history = {}, {}
    context_days = int(config["context_days"])
    context_start = origin - timedelta(days=context_days - 1)
    target_dates = [start + timedelta(days=i) for i in range(days)]
    for code in codes:
        values = actual_panel.get(code, {})
        history[code] = [values.get(context_start + timedelta(days=i), 0.0) for i in range(context_days)]
        actuals[code] = [values.get(day, 0.0) for day in target_dates]
    if any(value < 0 for values in actuals.values() for value in values):
        raise ValueError("Negative target sales found; this simulator expects prepared nonnegative gross sales")
    return codes, forecasts, actuals, history, {
        "origin": origin, "start": start, "end": end, "days": days,
        "target_dates": target_dates, "forecast_rows": len(first_keys),
    }


def build_inventory_scenarios(codes: list[str], history: dict[str, list[float]], config: dict) -> dict[str, dict]:
    """Build one identical synthetic stock and supplier state per SKU/model."""
    rng = random.Random(int(config["inventory_scenario_seed"]))
    stock_range = config["initial_stock_coverage_days_range"]
    lead_range = config["supplier_lead_time_days_range"]
    result = {}
    for code in codes:
        values = history[code]
        average = mean(values)
        on_hand = max(0, int(round(average * rng.uniform(*map(float, stock_range)))))
        lead = rng.randint(*map(int, lead_range))
        inbound = []
        if rng.random() < float(config["incoming_order_probability"]):
            weeks = rng.uniform(*map(float, config["incoming_order_coverage_weeks_range"]))
            quantity = max(0, int(round(average * 7 * weeks)))
            if quantity:
                inbound.append({
                    "arrival_day": rng.randint(*map(int, config["incoming_order_arrival_days_range"])),
                    "quantity": quantity,
                })
        result[code] = {
            "daily_average_90d": average,
            "starting_on_hand": on_hand,
            "lead_time_days": lead,
            "incoming_order": inbound[0] if inbound else None,
        }
    return result


def build_plans(codes: list[str], forecasts: dict, stock: dict, config: dict, dates: dict) -> tuple[list[dict], dict]:
    protection_days = int(config["review_period_days"])
    plans = []
    model_arrays = {model: [] for model in MODELS}
    for code in codes:
        state = stock[code]
        protection = min(state["lead_time_days"] + int(config["review_period_days"]), dates["days"])
        inbound = state["incoming_order"]
        inventory_position = state["starting_on_hand"] + (inbound["quantity"] if inbound else 0)
        row = {
            "stock_code": code,
            "daily_average_90d": state["daily_average_90d"],
            "starting_on_hand": state["starting_on_hand"],
            "in_transit_units": inbound["quantity"] if inbound else 0,
            "in_transit_arrival_day": inbound["arrival_day"] if inbound else "",
            "lead_time_days": state["lead_time_days"],
            "review_period_days": int(config["review_period_days"]),
            "protection_period_days": protection,
            "inventory_position": inventory_position,
        }
        for model in MODELS:
            expected = sum(r["prediction"] for r in forecasts[model][code][:protection])
            buffer = expected * float(config["common_safety_stock_fraction"])
            target = expected + buffer
            order = max(0, math.ceil(target - inventory_position - 1e-6))
            row[f"{model}_forecast_demand"] = expected
            row[f"{model}_safety_buffer"] = buffer
            row[f"{model}_target_stock"] = target
            row[f"{model}_order_units"] = order
            model_arrays[model].append({
                "predictions": [r["prediction"] for r in forecasts[model][code]],
                "order": order,
            })
        plans.append(row)
    return plans, model_arrays


def _aggregate_paths(paths: list[dict], model: str) -> dict:
    subset = [row for row in paths if row["model"] == model]
    fill = np.asarray([r["fill_rate"] for r in subset], dtype=float)
    costs = np.asarray([r["total_cost_gbp"] for r in subset], dtype=float)
    unmet = np.asarray([r["unmet_units"] for r in subset], dtype=float)
    return {
        "simulation_count": len(subset),
        "mean_demand_units": float(np.mean([r["demand_units"] for r in subset])),
        "mean_fill_rate": float(fill.mean()),
        "p05_fill_rate": float(np.quantile(fill, .05)),
        "median_fill_rate": float(np.quantile(fill, .50)),
        "p95_fill_rate": float(np.quantile(fill, .95)),
        "mean_unmet_units": float(unmet.mean()),
        "p95_unmet_units": float(np.quantile(unmet, .95)),
        "mean_shortage_sku_fraction": float(np.mean([r["sku_shortage_fraction"] for r in subset])),
        "probability_any_sku_shortage": float(np.mean([r["sku_shortage_fraction"] > 0 for r in subset])),
        "mean_average_inventory_units": float(np.mean([r["average_inventory_units"] for r in subset])),
        "mean_holding_cost_gbp": float(np.mean([r["holding_cost_gbp"] for r in subset])),
        "mean_order_cost_gbp": float(np.mean([r["order_cost_gbp"] for r in subset])),
        "mean_shortage_cost_gbp": float(np.mean([r["shortage_cost_gbp"] for r in subset])),
        "mean_total_cost_gbp": float(costs.mean()),
        "p95_total_cost_gbp": float(np.quantile(costs, .95)),
    }


def run_monte_carlo(root: Path, config: dict) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    codes, forecasts, actuals, history, dates = load_inputs(root, config)
    inventory = build_inventory_scenarios(codes, history, config)
    plans, model_arrays = build_plans(codes, forecasts, inventory, config, dates)
    sims, days, products = int(config["simulation_count"]), dates["days"], len(codes)
    rng = np.random.default_rng(int(config["scenario_seed"]))
    demand_multipliers = rng.choice(config["demand_spike_multipliers"], sims, p=config["demand_spike_probabilities"])
    delays = rng.choice(config["supplier_delay_days"], sims, p=config["supplier_delay_probabilities"])
    losses = rng.choice(config["starting_stock_loss_fractions"], sims, p=config["starting_stock_loss_probabilities"])
    actual = np.asarray([actuals[code] for code in codes], dtype=float)
    starting_base = np.asarray([inventory[code]["starting_on_hand"] for code in codes], dtype=float)
    leads = np.asarray([inventory[code]["lead_time_days"] for code in codes], dtype=int)
    in_qty = np.asarray([inventory[code]["incoming_order"]["quantity"] if inventory[code]["incoming_order"] else 0 for code in codes], dtype=float)
    in_day = np.asarray([inventory[code]["incoming_order"]["arrival_day"] - 1 if inventory[code]["incoming_order"] else -1 for code in codes], dtype=int)
    hold_rate = float(config["cost_assumptions_gbp"]["holding_cost_per_unit_day"])
    place_cost = float(config["cost_assumptions_gbp"]["order_cost_per_order"])
    shortage_rate = float(config["cost_assumptions_gbp"]["shortage_penalty_per_unmet_unit"])
    paths, recommendations, daily_rows, factor_rows = [], [], [], []
    daily_unmet = {model: np.zeros((sims, days), dtype=float) for model in MODELS}
    for model in MODELS:
        orders = np.asarray([x["order"] for x in model_arrays[model]], dtype=float)
        summary_order = {
            "model": model, "model_label": LABELS[model],
            "products": products, "products_ordered": int(np.count_nonzero(orders)),
            "total_order_units": float(orders.sum()),
            "mean_order_units_per_product": float(orders.mean()),
            "mean_forecast_demand_28d": float(np.mean([sum(x["predictions"]) for x in model_arrays[model]])),
            "forecast_mae": float(np.mean([
                abs(model_arrays[model][i]["predictions"][j] - actual[i, j])
                for i in range(products) for j in range(days)
            ])),
        }
        recommendations.append(summary_order)
        for policy_i, plan in enumerate(plans):
            code = plan["stock_code"]
            plan[f"{model}_arrival_date"] = (
                (dates["start"] + timedelta(days=inventory[code]["lead_time_days"] - 1)).isoformat()
                if plan[f"{model}_order_units"] > 0 else ""
            )

        for sim in range(sims):
            on_hand = np.floor(starting_base * (1.0 - losses[sim]))
            demand = actual * demand_multipliers[sim]
            unmet_by_sku = np.zeros(products, dtype=float)
            total_demand = fulfilled_total = unmet_total = 0.0
            inventory_unit_days = holding_cost = shortage_cost = 0.0
            for day in range(days):
                arrival_existing = ((in_day + int(delays[sim])) == day) * in_qty
                arrival_new = ((leads - 1 + int(delays[sim])) == day) * orders
                opening = on_hand + arrival_existing + arrival_new
                todays = demand[:, day]
                fulfilled = np.minimum(opening, todays)
                unmet = todays - fulfilled
                on_hand = opening - fulfilled
                holding_cost += float((((opening + on_hand) / 2.0).sum()) * hold_rate)
                shortage_cost += float(unmet.sum() * shortage_rate)
                total_demand += float(todays.sum())
                fulfilled_total += float(fulfilled.sum())
                unmet_total += float(unmet.sum())
                unmet_by_sku += (unmet > 1e-9)
                inventory_unit_days += float(((opening + on_hand) / 2.0).sum())
                daily_unmet[model][sim, day] = float(unmet.sum())
            n_short = int(np.count_nonzero(unmet_by_sku))
            order_cost = float(np.count_nonzero(orders) * place_cost)
            paths.append({
                "scenario_id": sim + 1, "model": model,
                "demand_multiplier": float(demand_multipliers[sim]),
                "supplier_delay_days": int(delays[sim]),
                "starting_stock_loss_fraction": float(losses[sim]),
                "demand_units": total_demand, "fulfilled_units": fulfilled_total,
                "unmet_units": unmet_total,
                "fill_rate": fulfilled_total / total_demand if total_demand else 1.0,
                "sku_shortage_fraction": n_short / products,
                "average_inventory_units": inventory_unit_days / (products * days),
                "holding_cost_gbp": holding_cost, "order_cost_gbp": order_cost,
                "shortage_cost_gbp": shortage_cost,
                "total_cost_gbp": holding_cost + order_cost + shortage_cost,
                "new_order_units": float(orders.sum()),
            })
    policy_summary = {model: _aggregate_paths(paths, model) for model in MODELS}
    for model in MODELS:
        for day, value in enumerate(daily_unmet[model].mean(axis=0)):
            daily_rows.append({"day": day + 1, "date": dates["target_dates"][day],
                               "model": model, "mean_unmet_units": float(value)})
    for factor, values, column, formatter in (
        ("demand_spike", demand_multipliers, "demand_multiplier", float),
        ("supplier_delay", delays, "supplier_delay_days", int),
        ("starting_stock_loss", losses, "starting_stock_loss_fraction", float),
    ):
        for value in sorted(set(map(formatter, values))):
            for model in MODELS:
                subset = [r for r in paths if r["model"] == model and r[column] == value]
                if subset:
                    costs = [r["total_cost_gbp"] for r in subset]
                    factor_rows.append({
                        "factor": factor, "value": value, "model": model,
                        "scenario_count": len(subset),
                        "mean_fill_rate": float(np.mean([r["fill_rate"] for r in subset])),
                        "mean_unmet_units": float(np.mean([r["unmet_units"] for r in subset])),
                        "mean_total_cost_gbp": float(np.mean(costs)),
                        "p95_total_cost_gbp": float(np.quantile(costs, .95)),
                    })
    scorecard = {
        "name": "paired-model-inventory-plan-stress-test",
        "forecast_origin": dates["origin"].isoformat(),
        "evaluation_period": [dates["start"].isoformat(), dates["end"].isoformat()],
        "products": products, "forecast_rows_per_model": dates["forecast_rows"],
        "days": days, "simulation_count": sims,
        "scenario_seed": int(config["scenario_seed"]),
        "inventory_scenario_seed": int(config["inventory_scenario_seed"]),
        "models": {model: {"label": LABELS[model], **policy_summary[model]} for model in MODELS},
        "plan_summary": recommendations,
        "paired_scenarios": True,
        "assumptions": config["assumptions"],
        "scenario_probabilities": {
            "demand_spike_multipliers": dict(zip(map(str, config["demand_spike_multipliers"]), config["demand_spike_probabilities"])),
            "supplier_delay_days": dict(zip(map(str, config["supplier_delay_days"]), config["supplier_delay_probabilities"])),
            "starting_stock_loss_fractions": dict(zip(map(str, config["starting_stock_loss_fractions"]), config["starting_stock_loss_probabilities"])),
        },
        "forecast_period_previously_reused": True,
        "forecast_period_caveat": "This October window was previously used in exploratory/model evaluation work; results are descriptive, not a pristine blind benchmark.",
    }
    return scorecard, paths, daily_rows, factor_rows, plans
