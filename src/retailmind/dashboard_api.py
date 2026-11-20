"""Local FastAPI backend for the RetailMind decision dashboard and assistant tools."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retailmind.model_catalog import LABELS, MODELS
from retailmind.forecast_api import router as forecasting_router
from retailmind.timestral_runtime import model_files_missing, model_status

ROOT = Path(os.environ.get("RETAILMIND_ROOT", Path(__file__).resolve().parents[2])).resolve()
M4_OUT = ROOT / "artifacts/milestone4_scenarios/model_plan_comparison"
M2_OUT = ROOT / "artifacts/milestone2_uncertainty/timestral209m_lora"
UI_FILE = ROOT / "apps/dashboard/index.html"
AUDIT_DB = Path(os.environ.get("RETAILMIND_AUDIT_DB", ROOT / "artifacts/milestone5/audit.sqlite3"))
MODEL_VERSION = "retailmind-timestral-lstm-timesfm3-2011-10-v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit(event_type: str, request: dict, tool_name: str, result: dict) -> None:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUDIT_DB) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS assistant_events (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, event_type TEXT NOT NULL,
            tool_name TEXT NOT NULL, model_version TEXT NOT NULL,
            request_json TEXT NOT NULL, result_json TEXT NOT NULL
        )""")
        db.execute(
            "INSERT INTO assistant_events(created_at,event_type,tool_name,model_version,request_json,result_json) VALUES(?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), event_type, tool_name, MODEL_VERSION,
             json.dumps(request, sort_keys=True), json.dumps(result, sort_keys=True)),
        )


class ScenarioRequest(BaseModel):
    stock_code: str = Field(min_length=1, max_length=32)
    model: str
    demand_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)
    supplier_delay_days: int = Field(default=0, ge=0, le=90)
    starting_stock_loss_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class RecommendationRequest(BaseModel):
    stock_code: str = Field(min_length=1, max_length=32)
    model: str
    safety_stock_fraction: float = Field(default=0.2, ge=0.0, le=2.0)


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    stock_code: str = Field(min_length=1, max_length=32)
    model: str
    demand_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)
    supplier_delay_days: int = Field(default=0, ge=0, le=90)
    starting_stock_loss_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_stock_fraction: float = Field(default=0.2, ge=0.0, le=2.0)


class DashboardData:
    def __init__(self) -> None:
        self.scorecard = json.loads((M4_OUT / "scorecard.json").read_text(encoding="utf-8"))
        self.m2 = json.loads((M2_OUT / "scorecard.json").read_text(encoding="utf-8"))
        self.config = json.loads((ROOT / "configs/milestone4_scenarios.json").read_text(encoding="utf-8"))
        self.plan_rows = {row["stock_code"]: row for row in read_rows(M4_OUT / "product-plan-recommendations.csv")}
        self.forecasts: dict[str, dict[str, list[dict]]] = {model: {} for model in MODELS}
        forecast_files = self.config["forecast_files"]
        for model in MODELS:
            for row in read_rows(ROOT / forecast_files[model]):
                if row.get("model") and row["model"] != model:
                    continue
                self.forecasts[model].setdefault(row["stock_code"], []).append(row)
            for rows in self.forecasts[model].values():
                rows.sort(key=lambda row: int(row["horizon"]))
        self.intervals: dict[str, list[dict]] = {}
        for row in read_rows(M2_OUT / "evaluation-intervals.csv"):
            self.intervals.setdefault(row["stock_code"], []).append(row)
        self.products = sorted(self.plan_rows)
        self.actuals: dict[str, dict[str, float]] = {code: {} for code in self.products}
        period = set(self.scorecard["evaluation_period"])
        with (ROOT / "data/processed/daily_sales.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["stock_code"] in self.actuals and row["date"] in period:
                    self.actuals[row["stock_code"]][row["date"]] = float(row["quantity"])
        self.featured_product = max(self.products, key=lambda code: sum(self.actuals[code].values()))
        self.descriptions = {}
        feature_file = ROOT / "reports/generated/product-features-all.csv"
        if feature_file.exists():
            for row in read_rows(feature_file):
                if row.get("as_of") == "2011-04-30" and row.get("description"):
                    self.descriptions[row["stock_code"]] = row["description"]
        self.plan_summary = {row["model"]: row for row in self.scorecard["plan_summary"]}

    def validate(self, code: str, model: str) -> dict:
        if model not in MODELS:
            raise HTTPException(422, f"Unknown model. Choose one of: {', '.join(MODELS)}")
        if code not in self.plan_rows:
            raise HTTPException(404, "Product is outside the common model-comparison cohort")
        return self.plan_rows[code]

    def simulate(self, code: str, model: str, demand_multiplier: float = 1.0,
                 delay: int = 0, stock_loss: float = 0.0, order_override: int | None = None) -> dict:
        plan = self.validate(code, model)
        on_hand = int(float(plan["starting_on_hand"]) * (1.0 - stock_loss))
        rows = self.forecasts[model][code]
        dates = self.scorecard["evaluation_period"]
        inbound_qty = int(plan["in_transit_units"])
        inbound_day = int(plan["in_transit_arrival_day"]) - 1 if plan["in_transit_arrival_day"] else -1
        lead = int(plan["lead_time_days"])
        order_qty = int(plan[f"{model}_order_units"] if order_override is None else order_override)
        arrivals: dict[int, int] = {}
        if inbound_qty:
            arrivals[inbound_day + delay] = arrivals.get(inbound_day + delay, 0) + inbound_qty
        if order_qty:
            arrivals[lead - 1 + delay] = arrivals.get(lead - 1 + delay, 0) + order_qty
        holding = shortage_cost = demand_total = fulfilled_total = unmet_total = 0.0
        costs = self.config["cost_assumptions_gbp"]
        daily = []
        for i, forecast in enumerate(rows):
            opening = on_hand + arrivals.get(i, 0)
            demand = max(0.0, self.actuals[code].get(forecast["target_date"], 0.0) * demand_multiplier)
            fulfilled = min(opening, demand)
            unmet = demand - fulfilled
            on_hand = opening - fulfilled
            hold_today = ((opening + on_hand) / 2.0) * float(costs["holding_cost_per_unit_day"])
            shortage_today = unmet * float(costs["shortage_penalty_per_unmet_unit"])
            holding += hold_today
            shortage_cost += shortage_today
            demand_total += demand
            fulfilled_total += fulfilled
            unmet_total += unmet
            daily.append({
                "date": forecast["target_date"], "forecast": max(0.0, float(forecast["prediction"])),
                "actual_demand_proxy": demand, "opening_stock": opening,
                "fulfilled_units": fulfilled, "unmet_units": unmet, "closing_stock": on_hand,
                "holding_cost_gbp": hold_today, "shortage_cost_gbp": shortage_today,
            })
        order_cost = float(costs["order_cost_per_order"]) if order_qty > 0 else 0.0
        return {
            "stock_code": code, "model": model, "model_label": LABELS[model],
            "demand_multiplier": demand_multiplier, "supplier_delay_days": delay,
            "starting_stock_loss_fraction": stock_loss,
            "starting_on_hand": int(float(plan["starting_on_hand"]) * (1.0 - stock_loss)),
            "in_transit_units": inbound_qty, "lead_time_days": lead,
            "order_units": order_qty, "total_demand_units": demand_total,
            "fulfilled_units": fulfilled_total, "unmet_units": unmet_total,
            "fill_rate": fulfilled_total / demand_total if demand_total else 1.0,
            "ending_on_hand": on_hand, "holding_cost_gbp": holding,
            "order_cost_gbp": order_cost, "shortage_cost_gbp": shortage_cost,
            "total_cost_gbp": holding + order_cost + shortage_cost,
            "daily": daily,
        }

    def recommendation(self, code: str, model: str, safety_fraction: float) -> dict:
        plan = self.validate(code, model)
        rows = self.forecasts[model][code]
        n = int(plan["protection_period_days"])
        expected = sum(max(0.0, float(r["prediction"])) for r in rows[:n])
        buffer = expected * safety_fraction
        position = int(plan["starting_on_hand"]) + int(plan["in_transit_units"])
        target = expected + buffer
        order = max(0, int(math.ceil(target - position - 1e-6)))
        return {
            "stock_code": code, "model": model, "model_label": LABELS[model],
            "protection_period_days": n, "expected_demand_units": expected,
            "safety_stock_fraction": safety_fraction, "safety_buffer_units": buffer,
            "target_stock_units": target, "inventory_position_units": position,
            "proposed_order_units": order,
            "is_proposal_only": True,
        }


try:
    DATA: DashboardData | None = DashboardData()
except FileNotFoundError:
    # Input-driven forecasting endpoints work without the optional historical
    # report bundle. Dashboard routes return 503 until that bundle is mounted.
    DATA = None


def require_dashboard_data() -> DashboardData:
    if DATA is None:
        raise HTTPException(503, "Historical dashboard artifacts are unavailable; mount the project artifacts bundle")
    return DATA


app = FastAPI(
    title="RetailMind Demand Forecasting API",
    version="0.6.0",
    description=("Input-driven daily demand forecasting, holdout evaluation, batch inference, "
                 "inventory proposals, and scenarios. Historical dashboard routes are optional."),
)
app.include_router(forecasting_router)


@app.get("/", include_in_schema=False)
def index():
    if DATA is None:
        return {
            "service": "RetailMind Demand Forecasting API",
            "openapi": "/openapi.json",
            "interactive_docs": "/docs",
            "capabilities": "/api/v1/capabilities",
            "forecast_endpoint": "POST /api/v1/forecast",
        }
    if not UI_FILE.exists():
        raise HTTPException(404, "Dashboard UI file is missing")
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "service": "demand-forecasting", "model_version": MODEL_VERSION,
            "historical_dashboard_loaded": DATA is not None,
            "forecast_api": "/api/v1/capabilities"}


@app.get("/ready")
def ready():
    missing_model = model_files_missing()
    if missing_model:
        raise HTTPException(503, {"status": "model_not_ready", "missing_artifacts": missing_model})
    if DATA is None:
        return {"status": "ready", "service": "demand-forecasting",
                "historical_dashboard_loaded": False,
                "model": model_status(),
                "detail": "Input-driven forecast API is ready; historical dashboard artifacts are not mounted."}
    required = [
        M4_OUT / "scorecard.json", M4_OUT / "product-plan-recommendations.csv",
        M2_OUT / "evaluation-intervals.csv", ROOT / "data/processed/daily_sales.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise HTTPException(503, {"status": "not_ready", "missing_artifacts": missing})
    return {"status": "ready", "model_version": MODEL_VERSION,
            "historical_dashboard_loaded": True, "products": len(DATA.products),
            "model": model_status()}


@app.get("/api/ops/status")
def ops_status():
    snapshot_path = ROOT / "artifacts/milestone6/monitoring/latest.json"
    if not snapshot_path.is_file():
        return {"status": "not_initialized", "detail": "Run scripts/ops/monitor_daily.py --once"}
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return {
        "status": snapshot["status"], "checked_at_utc": snapshot["checked_at_utc"],
        "latest_record_date": snapshot["data"]["latest_record_date"],
        "age_days": snapshot["data"]["age_days"],
        "action": snapshot["action"],
    }


@app.get("/api/portfolio")
def portfolio():
    data = require_dashboard_data()
    return {
        "products": data.scorecard["products"], "days": data.scorecard["days"],
        "featured_product": data.featured_product,
        "period": data.scorecard["evaluation_period"],
        "models": data.scorecard["models"], "plan_summary": data.scorecard["plan_summary"],
        "uncertainty": {
            "model": data.m2["model"], "period": data.m2["evaluation_period"],
            "nominal_coverage": data.m2["nominal_coverage"],
            "observed_coverage": data.m2["overall"]["calibrated_coverage"],
            "mean_interval_width": data.m2["overall"]["calibrated_mean_width"],
        },
        "caveat": data.scorecard["forecast_period_caveat"],
    }


@app.get("/api/products")
def products(search: str = Query(default="", max_length=100), limit: int = Query(default=100, ge=1, le=500)):
    data = require_dashboard_data()
    term = search.casefold()
    matched = [code for code in data.products if term in code.casefold() or term in data.descriptions.get(code, "").casefold()]
    return [{"stock_code": code, "description": data.descriptions.get(code, "")}
            for code in matched[:limit]]


@app.get("/api/products/{stock_code}")
def product_detail(stock_code: str):
    data = require_dashboard_data()
    if stock_code not in data.plan_rows:
        raise HTTPException(404, "Product is outside the comparison cohort")
    plan = data.plan_rows[stock_code]
    models = {}
    for model in MODELS:
        rows = data.forecasts[model][stock_code]
        sim = data.simulate(stock_code, model)
        models[model] = {
            "label": LABELS[model],
            "forecast_mae": data.plan_summary[model]["forecast_mae"],
            "forecast_demand_28d": float(plan[f"{model}_forecast_demand"]),
            "safety_buffer_units": float(plan[f"{model}_safety_buffer"]),
            "order_units": int(plan[f"{model}_order_units"]),
            "arrival_date": plan[f"{model}_arrival_date"],
            "forecast": [
                {"date": row["target_date"], "prediction": max(0.0, float(row["prediction"])),
                 "actual": data.actuals[stock_code].get(row["target_date"], 0.0),
                 "closing_stock": sim["daily"][i]["closing_stock"],
                 "unmet_units": sim["daily"][i]["unmet_units"]}
                for i, row in enumerate(rows)
            ],
            "baseline_projection": sim,
        }
    intervals = [
        {k: row[k] for k in ("target_date", "actual", "p10", "p50", "p90", "calibrated_lower", "calibrated_upper")}
        for row in data.intervals.get(stock_code, [])
    ]
    return {
        "stock_code": stock_code,
        "description": data.descriptions.get(stock_code, "Description unavailable"),
        "period": data.scorecard["evaluation_period"],
        "inventory": {
            "opening_on_hand": int(plan["starting_on_hand"]),
            "in_transit_units": int(plan["in_transit_units"]),
            "in_transit_arrival_day": plan["in_transit_arrival_day"],
            "lead_time_days": int(plan["lead_time_days"]),
            "review_period_days": int(plan["review_period_days"]),
            "protection_period_days": int(plan["protection_period_days"]),
            "inventory_position": int(plan["inventory_position"]),
        },
        "models": models,
        "timestral_uncertainty": {
            "period": data.m2["evaluation_period"], "rows": intervals,
        },
        "uncertainty_note": "Timestral interval chart is a separate November–December evaluation origin; it is not aligned to the October model-plan comparison.",
    }


@app.post("/api/tools/scenario")
def scenario(request: ScenarioRequest):
    data = require_dashboard_data()
    result = data.simulate(request.stock_code, request.model, request.demand_multiplier,
                           request.supplier_delay_days, request.starting_stock_loss_fraction)
    result["proposal_only"] = True
    audit("scenario", request.model_dump(), "simulate_product_inventory", result)
    return result


@app.post("/api/tools/recommendation")
def recommendation(request: RecommendationRequest):
    data = require_dashboard_data()
    result = data.recommendation(request.stock_code, request.model, request.safety_stock_fraction)
    audit("recommendation", request.model_dump(), "recalculate_proposed_order", result)
    return result


@app.post("/api/assistant")
def assistant(request: AssistantRequest):
    data = require_dashboard_data()
    text = request.message.casefold()
    if any(word in text for word in ("delay", "late", "what if", "spike", "loss", "shortage", "simulate")):
        delay = request.supplier_delay_days
        mult = request.demand_multiplier
        loss = request.starting_stock_loss_fraction
        delay_match = re.search(r"(\d+)\s*(?:day|days|d)\b", text)
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if delay_match and any(word in text for word in ("delay", "late")):
            delay = int(delay_match.group(1))
        if pct_match:
            fraction = float(pct_match.group(1)) / 100.0
            if "loss" in text or "missing" in text or "damage" in text:
                loss = fraction
            elif "demand" in text or "spike" in text or "higher" in text:
                mult = 1.0 + fraction
        result = data.simulate(request.stock_code, request.model, mult, delay, loss)
        tool = "simulate_product_inventory"
        answer = (
            f"Under this proposed scenario for {request.stock_code}, {LABELS[request.model]} "
            f"would fill {result['fill_rate']:.1%} of the recorded-sales demand proxy, "
            f"with {result['unmet_units']:.1f} units unmet and estimated cost £{result['total_cost_gbp']:.2f}. "
            "This is a simulation only; no order was placed."
        )
        result["answer"] = answer
    elif (any(word in text for word in ("order", "recommend", "buffer", "buy"))
          and not any(word in text for word in ("why", "explain"))):
        fraction = request.safety_stock_fraction
        if any(word in text for word in ("more", "increase", "higher")) and not re.search(r"\d+(?:\.\d+)?\s*%", text):
            fraction = min(2.0, fraction + 0.1)
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if pct_match:
            fraction = float(pct_match.group(1)) / 100.0
        result = data.recommendation(request.stock_code, request.model, fraction)
        tool = "recalculate_proposed_order"
        result["answer"] = (
            f"For {request.stock_code}, {LABELS[request.model]} forecasts "
            f"{result['expected_demand_units']:.1f} units over {result['protection_period_days']} protection days. "
            f"With a {fraction:.0%} safety buffer, the proposed order is {result['proposed_order_units']} units. "
            "This is a proposal only; no purchase was placed."
        )
    elif any(word in text for word in ("why", "explain", "forecast", "stock")):
        plan = data.plan_rows.get(request.stock_code)
        if plan is None:
            raise HTTPException(404, "Product is outside the comparison cohort")
        sim = data.simulate(request.stock_code, request.model)
        order = int(plan[f"{request.model}_order_units"])
        result = {
            "stock_code": request.stock_code, "model": request.model,
            "forecast_demand_protection_period": float(plan[f"{request.model}_forecast_demand"]),
            "safety_buffer_units": float(plan[f"{request.model}_safety_buffer"]),
            "opening_on_hand": int(plan["starting_on_hand"]),
            "in_transit_units": int(plan["in_transit_units"]),
            "lead_time_days": int(plan["lead_time_days"]),
            "proposed_order_units": order, "baseline_fill_rate": sim["fill_rate"],
            "baseline_unmet_units": sim["unmet_units"], "proposal_only": True,
            "answer": (
                f"{LABELS[request.model]} projects {float(plan[f'{request.model}_forecast_demand']):.1f} units "
                f"during the {int(plan['protection_period_days'])}-day protection period. "
                f"The shared 20% buffer adds {float(plan[f'{request.model}_safety_buffer']):.1f} units. "
                f"After {int(plan['starting_on_hand'])} on hand and {int(plan['in_transit_units'])} in transit, "
                f"the model-specific proposal is {order} units. It is not a committed purchase."
            ),
        }
        tool = "explain_product_plan"
    else:
        result = {
            "answer": "I can explain a product plan, simulate a supplier delay/demand spike/stock loss, or recalculate a proposed order with a different safety buffer. Try: ‘Why should I order more?’ or ‘What if the supplier is delayed 5 days?’",
            "available_tools": ["explain_product_plan", "simulate_product_inventory", "recalculate_proposed_order"],
            "proposal_only": True,
        }
        tool = "no_tool_matched"
    result["model_version"] = MODEL_VERSION
    audit("assistant", request.model_dump(), tool, result)
    return result
