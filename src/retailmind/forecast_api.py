"""Input-driven demand forecasting and evaluation endpoints.

These routes forecast from caller-supplied daily sales history. They are kept
separate from the dashboard's frozen historical comparison artifacts so the
API never presents a saved test forecast as a live prediction.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import isfinite, sqrt
from statistics import mean
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from retailmind.timestral_runtime import forecast_quantiles, model_status

router = APIRouter(prefix="/api/v1", tags=["Demand forecasting"])
ForecastMethod = Literal["timestral209m_lora"]


class DailyObservation(BaseModel):
    date: date
    quantity: float = Field(ge=0, le=1_000_000_000, allow_inf_nan=False)

    @model_validator(mode="after")
    def quantity_is_finite(self):
        if not isfinite(self.quantity):
            raise ValueError("quantity must be a finite number")
        return self


class DemandSeries(BaseModel):
    series_id: str = Field(min_length=1, max_length=100)
    history: list[DailyObservation] = Field(min_length=7, max_length=3650)

    @model_validator(mode="after")
    def dates_must_be_daily_and_unique(self):
        days = [row.date for row in self.history]
        if days != sorted(days) or len(days) != len(set(days)):
            raise ValueError("history dates must be unique and ordered oldest to newest")
        if any((right - left).days != 1 for left, right in zip(days, days[1:])):
            raise ValueError("history must contain every calendar day; send zero for days with no recorded sales")
        if any(not isfinite(row.quantity) for row in self.history):
            raise ValueError("quantities must be finite numbers")
        return self


class ForecastRequest(BaseModel):
    series: DemandSeries
    horizon_days: int = Field(default=28, ge=1, le=28)
    methods: list[ForecastMethod] = Field(
        default_factory=lambda: ["timestral209m_lora"], min_length=1, max_length=1,
    )

    @model_validator(mode="after")
    def unique_methods(self):
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("methods must not contain duplicates")
        return self


class BatchForecastRequest(BaseModel):
    series: list[DemandSeries] = Field(min_length=1, max_length=25)
    horizon_days: int = Field(default=28, ge=1, le=28)
    methods: list[ForecastMethod] = Field(
        default_factory=lambda: ["timestral209m_lora"], min_length=1, max_length=1,
    )

    @model_validator(mode="after")
    def ids_are_unique(self):
        ids = [item.series_id for item in self.series]
        if len(ids) != len(set(ids)):
            raise ValueError("series_id values must be unique within a batch")
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("methods must not contain duplicates")
        return self


class EvaluationRequest(BaseModel):
    series: DemandSeries
    holdout_days: int = Field(default=28, ge=1, le=28)
    methods: list[ForecastMethod] = Field(
        default_factory=lambda: ["timestral209m_lora"], min_length=1, max_length=1,
    )


class InventoryProposalRequest(BaseModel):
    forecast: list[float] = Field(min_length=1, max_length=90)
    upper_forecast: list[float] | None = Field(default=None, min_length=1, max_length=90)
    on_hand_units: float = Field(ge=0, allow_inf_nan=False)
    in_transit_units: float = Field(default=0, ge=0, allow_inf_nan=False)
    lead_time_days: int = Field(default=0, ge=0, le=365)
    review_period_days: int = Field(default=0, ge=0, le=365)
    safety_stock_fraction: float = Field(default=0.2, ge=0, le=5, allow_inf_nan=False)

    @model_validator(mode="after")
    def forecast_is_finite_nonnegative(self):
        if any(not isfinite(value) or value < 0 for value in self.forecast):
            raise ValueError("forecast values must be finite and nonnegative")
        if self.upper_forecast is not None:
            if len(self.upper_forecast) != len(self.forecast):
                raise ValueError("upper_forecast must have the same length as forecast")
            if any(not isfinite(value) or value < self.forecast[index]
                   for index, value in enumerate(self.upper_forecast)):
                raise ValueError("upper_forecast values must be finite and at least the P50 forecast")
        return self


class ScenarioRequest(BaseModel):
    forecast: list[float] = Field(min_length=1, max_length=90)
    starting_stock: float = Field(ge=0, allow_inf_nan=False)
    inbound_units: float = Field(default=0, ge=0, allow_inf_nan=False)
    inbound_arrival_day: int = Field(default=0, ge=0, le=90)
    demand_multiplier: float = Field(default=1, ge=0, le=10, allow_inf_nan=False)
    extra_supply_delay_days: int = Field(default=0, ge=0, le=90)

    @model_validator(mode="after")
    def forecast_is_finite_nonnegative(self):
        if any(not isfinite(value) or value < 0 for value in self.forecast):
            raise ValueError("forecast values must be finite and nonnegative")
        return self


def _history(series: DemandSeries) -> list[float]:
    return [float(row.quantity) for row in series.history]


def _forecast_one(series: DemandSeries, horizon: int, methods: list[ForecastMethod]) -> dict:
    values, dates = _history(series), [row.date for row in series.history]
    origin = dates[-1]
    if methods != ["timestral209m_lora"]:
        raise HTTPException(422, "This image serves only the selected Timestral 209M LoRA model")
    started = perf_counter()
    try:
        quantiles, device = forecast_quantiles(values, horizon)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(503, f"Timestral model is unavailable: {exc}") from exc
    elapsed = perf_counter() - started
    predictions = [row[1] for row in quantiles]
    outputs = {
        "timestral209m_lora": {
            "predictions": [
                {"date": (origin + timedelta(days=step)).isoformat(),
                 "demand_units": row[1], "p10": row[0], "p50": row[1], "p90": row[2],
                 "calibrated_lower": row[3], "calibrated_upper": row[4]}
                for step, row in enumerate(quantiles, start=1)
            ],
            "total_demand_units": sum(predictions),
            "mean_daily_demand_units": mean(predictions),
            "interval_total_lower_units": sum(row[3] for row in quantiles),
            "interval_total_upper_units": sum(row[4] for row in quantiles),
            "inference_seconds": elapsed,
            "device": device,
            "method_type": "fine_tuned_foundation_model",
        }
    }
    return {
        "series_id": series.series_id,
        "forecast_origin": origin.isoformat(),
        "horizon_days": horizon,
        "observations": len(values),
        "zero_sales_day_fraction": sum(value == 0 for value in values) / len(values),
        "methods": outputs,
        "model": model_status(),
        "note": "Daily lower and upper bounds use the saved horizon-wise conformal adjustments. Summed daily bounds are not calibrated aggregate portfolio intervals. Forecasts estimate recorded sales and cannot recover lost sales during stockouts.",
    }


def _metrics(actual: list[float], predicted: list[float], history: list[float]) -> dict:
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    scale_values = [abs(history[i] - history[i - 7]) for i in range(7, len(history))]
    scale = mean(scale_values) if scale_values else 0.0
    total = sum(actual)
    return {
        "mae": mean(errors),
        "rmse": sqrt(mean(error * error for error in errors)),
        "wape": sum(errors) / total if total else None,
        "mase": mean(errors) / scale if scale else None,
    }


@router.get("/capabilities")
def capabilities():
    return {
        "service": "RetailMind demand forecasting API",
        "selected_model": model_status(),
        "forecast_methods": {"timestral209m_lora": {"kind": "fine_tuned_foundation_model", "quantiles": [0.1, 0.5, 0.9], "maximum_horizon_days": 28}},
        "inventory_proposal": {"supports_interval_buffer": True, "input_quantiles": ["p50", "calibrated_upper"]},
        "operations": ["forecast", "batch_forecast", "holdout_evaluation", "inventory_proposal", "inventory_scenario"],
        "limits": {"max_horizon_days": 28, "max_batch_series": 25, "max_history_days": 3650},
        "data_contract": "Daily observations, ordered and contiguous; include zero quantity for dates without recorded sales.",
    }


@router.post("/forecast", summary="Forecast future recorded demand from daily sales history")
def forecast(request: ForecastRequest):
    return _forecast_one(request.series, request.horizon_days, request.methods)


@router.post("/forecast/batch")
def forecast_batch(request: BatchForecastRequest):
    started = perf_counter()
    forecasts = [_forecast_one(item, request.horizon_days, request.methods) for item in request.series]
    return {"count": len(forecasts), "horizon_days": request.horizon_days,
            "elapsed_seconds": perf_counter() - started, "forecasts": forecasts}


@router.post("/evaluate")
def evaluate(request: EvaluationRequest):
    observations = request.series.history
    holdout = request.holdout_days
    if len(observations) <= holdout + 6:
        raise HTTPException(422, "Provide more history than holdout_days plus 6 weekly baseline context days")
    training, target = observations[:-holdout], observations[-holdout:]
    train_series = DemandSeries(series_id=request.series.series_id, history=training)
    forecast_result = _forecast_one(train_series, holdout, request.methods)
    actual = [float(row.quantity) for row in target]
    result = {}
    for method, output in forecast_result["methods"].items():
        predicted = [row["demand_units"] for row in output["predictions"]]
        result[method] = {**_metrics(actual, predicted, _history(train_series)),
                          "inference_seconds": output["inference_seconds"]}
    return {
        "series_id": request.series.series_id,
        "train_end": training[-1].date.isoformat(), "evaluation_start": target[0].date.isoformat(),
        "evaluation_end": target[-1].date.isoformat(), "holdout_days": holdout,
        "actual_total_units": sum(actual), "results": result,
        "protocol": "Chronological holdout; methods are fit or forecast using the training prefix only.",
    }


@router.post("/inventory/proposal")
def inventory_proposal(request: InventoryProposalRequest):
    protection = request.lead_time_days + request.review_period_days
    if protection < 1:
        raise HTTPException(422, "lead_time_days + review_period_days must be at least 1")
    if protection > len(request.forecast):
        raise HTTPException(422, "forecast horizon must cover lead_time_days + review_period_days")
    expected = sum(request.forecast[:protection])
    upper_expected = (
        sum(request.upper_forecast[:protection])
        if request.upper_forecast is not None else expected
    )
    interval_buffer = upper_expected - expected
    fractional_buffer = expected * request.safety_stock_fraction
    safety = interval_buffer + fractional_buffer
    position = request.on_hand_units + request.in_transit_units
    target = expected + safety
    order = max(0, int(__import__("math").ceil(target - position - 1e-9)))
    return {"protection_period_days": protection, "expected_demand_units": expected,
            "upper_interval_demand_units": upper_expected,
            "interval_safety_buffer_units": interval_buffer,
            "fractional_safety_buffer_units": fractional_buffer,
            "safety_stock_units": safety, "target_stock_units": target,
            "inventory_position_units": position, "proposed_order_units": order,
            "proposal_only": True, "note": "No purchase order is created or sent."}


@router.post("/inventory/scenario")
def inventory_scenario(request: ScenarioRequest):
    stock = request.starting_stock
    arrivals = {}
    if request.inbound_units:
        day = request.inbound_arrival_day + request.extra_supply_delay_days
        if day < len(request.forecast):
            arrivals[day] = arrivals.get(day, 0.0) + request.inbound_units
    daily, fulfilled_total, unmet_total = [], 0.0, 0.0
    for index, base_demand in enumerate(request.forecast):
        opening = stock + arrivals.get(index, 0.0)
        demand = base_demand * request.demand_multiplier
        fulfilled = min(opening, demand)
        unmet = demand - fulfilled
        stock = opening - fulfilled
        fulfilled_total += fulfilled
        unmet_total += unmet
        daily.append({"day": index + 1, "forecast_demand_units": base_demand,
                      "scenario_demand_units": demand, "opening_stock_units": opening,
                      "fulfilled_units": fulfilled, "unmet_units": unmet,
                      "closing_stock_units": stock})
    total = fulfilled_total + unmet_total
    return {"daily": daily, "fulfilled_units": fulfilled_total, "unmet_units": unmet_total,
            "fill_rate": fulfilled_total / total if total else 1.0,
            "ending_stock_units": stock, "proposal_only": True}
