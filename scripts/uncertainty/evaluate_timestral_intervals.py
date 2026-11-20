"""Evaluate frozen Timestral interval calibration on the later study window."""
from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/milestone2_uncertainty.json").read_text())
OUT = ROOT / "artifacts/milestone2_uncertainty/timestral209m_lora"


def load_actuals(end: date) -> dict[tuple[str, str], float]:
    actuals = {}
    with (ROOT / CONFIG["source_data"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if date.fromisoformat(row["date"]) <= end:
                actuals[(row["stock_code"], row["date"])] = float(row["quantity"])
    return actuals


def read_forecasts(path: Path) -> list[dict]:
    period = CONFIG["evaluation"]
    start, end = date.fromisoformat(period["target_start"]), date.fromisoformat(period["target_end"])
    cohort = json.loads((ROOT / CONFIG["cohort_file"]).read_text())["codes"]
    expected = {
        (code, (start.fromordinal(start.toordinal() + step)).isoformat())
        for code in cohort for step in range((end - start).days + 1)
    }
    rows, keys = [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["stock_code"], row["target_date"])
            if key in keys:
                raise ValueError(f"Duplicate evaluation forecast: {key}")
            keys.add(key)
            if row["forecast_origin"] != period["forecast_origin"]:
                raise ValueError(f"Unexpected origin for {key}")
            for name in ("p10", "p50", "p90"):
                row[name] = float(row[name])
            row["horizon"] = int(row["horizon"])
            row["context_sale_days"] = int(row["context_sale_days"])
            rows.append(row)
    if keys != expected:
        raise ValueError(f"Evaluation forecast keys differ: {len(expected-keys)} missing, {len(keys-expected)} extra")
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    actual = [row["actual"] for row in rows]
    median = [row["p50"] for row in rows]
    lower = [row["calibrated_lower"] for row in rows]
    upper = [row["calibrated_upper"] for row in rows]
    alpha = 1.0 - float(CONFIG["interval_coverage_target"])
    interval_scores = [
        (hi - lo) + (2 / alpha) * max(lo - y, 0.0) + (2 / alpha) * max(y - hi, 0.0)
        for lo, hi, y in zip(lower, upper, actual)
    ]
    raw_covered = [row["p10"] <= row["actual"] <= row["p90"] for row in rows]
    calibrated_covered = [lo <= y <= hi for lo, y, hi in zip(lower, actual, upper)]
    pinball_losses = []
    for q, field in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        for row in rows:
            err = row["actual"] - row[field]
            pinball_losses.append((q * err) if err >= 0 else ((q - 1) * err))
    return {
        "rows": len(rows),
        "products": len({row["stock_code"] for row in rows}),
        "mae_p50": mean(abs(y - p) for y, p in zip(actual, median)),
        "rmse_p50": math.sqrt(mean((y - p) ** 2 for y, p in zip(actual, median))),
        "raw_p10_p90_coverage": mean(raw_covered),
        "calibrated_coverage": mean(calibrated_covered),
        "calibrated_mean_width": mean(hi - lo for lo, hi in zip(lower, upper)),
        "calibrated_interval_score": mean(interval_scores),
        "raw_quantile_pinball_mean": mean(pinball_losses),
    }


def write_report(result: dict, horizon_rows: list[dict], date_rows: list[dict]) -> None:
    labels = [row["target_date"][5:] for row in date_rows]
    actual = [row["actual"] for row in date_rows]
    p50 = [row["p50"] for row in date_rows]
    low = [row["lower"] for row in date_rows]
    high = [row["upper"] for row in date_rows]
    width, height, left, top, plot_w, plot_h = 1000, 360, 70, 20, 900, 270
    maximum = max(1.0, max(high + actual) * 1.05)
    x = lambda i: left + i * plot_w / max(1, len(labels) - 1)
    y = lambda v: top + plot_h * (1 - v / maximum)
    polygon = " ".join([f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(high)] +
                       [f"{x(i):.1f},{y(low[i]):.1f}" for i in reversed(range(len(low)))])
    def polyline(values):
        return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    chart = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Daily portfolio actual sales and Timestral median forecast">'
        f'<polygon points="{polygon}" fill="#4f7cff" opacity=".18"/>'
        f'<polyline points="{polyline(actual)}" fill="none" stroke="#101828" stroke-width="3"/>'
        f'<polyline points="{polyline(p50)}" fill="none" stroke="#155eef" stroke-width="2.5"/>'
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#cad3df"/>'
        f'<text x="{left}" y="{height-18}" font-size="12" fill="#667085">{html.escape(labels[0])}</text>'
        f'<text x="{left+plot_w-42}" y="{height-18}" font-size="12" fill="#667085">{html.escape(labels[-1])}</text>'
        '</svg>'
    )
    horizon_table = "".join(
        f"<tr><td>{row['horizon']}</td><td>{row['rows']:,}</td>"
        f"<td>{row['raw_p10_p90_coverage']:.1%}</td><td>{row['calibrated_coverage']:.1%}</td>"
        f"<td>{row['calibrated_mean_width']:.2f}</td></tr>"
        for row in horizon_rows
    )
    segment_table = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{row['rows']:,}</td>"
        f"<td>{row['calibrated_coverage']:.1%}</td><td>{row['calibrated_mean_width']:.2f}</td>"
        f"<td>{row['mae_p50']:.2f}</td></tr>"
        for name, row in result["segments"].items()
    )
    m = result["overall"]
    path = OUT / "milestone2-uncertainty-report.html"
    path.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailMind | Probabilistic Forecasting</title>
<style>body{{margin:0;background:#f3f6fb;color:#17253c;font:15px/1.5 system-ui,Segoe UI,sans-serif}}main{{max-width:1200px;margin:auto;padding:32px 22px}}h1{{font-size:36px;margin:4px 0}}h2{{font-size:19px;margin:0 0 14px}}.muted{{color:#667085}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}}.card,.panel{{background:white;border:1px solid #dce4ef;border-radius:15px;padding:18px;box-shadow:0 4px 16px #12233d0a}}.card small{{display:block;color:#667085}}.card strong{{font-size:25px}}.panel{{margin:14px 0}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #e7edf5;text-align:right}}td:first-child,th:first-child{{text-align:left}}.legend span{{margin-right:16px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}.note{{background:#fff8e7;border-left:4px solid #e6a700;padding:12px 14px;color:#594500}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:20px 12px}}}}</style></head><body><main>
<p class="muted">MILESTONE 2 · PROBABILISTIC FORECASTING</p><h1>Timestral 209M LoRA uncertainty</h1><p class="muted">Nominal central 80% interval · calibration {result['calibration_period'][0]}–{result['calibration_period'][1]} · evaluation {result['evaluation_period'][0]}–{result['evaluation_period'][1]}</p>
<section class="cards"><div class="card"><small>Products × days</small><strong>{m['products']:,} × 28</strong></div><div class="card"><small>Evaluation coverage</small><strong>{m['calibrated_coverage']:.1%}</strong></div><div class="card"><small>Mean interval width</small><strong>{m['calibrated_mean_width']:.2f} units</strong></div><div class="card"><small>Median forecast MAE</small><strong>{m['mae_p50']:.2f} units</strong></div></section>
<section class="panel"><h2>Daily portfolio totals</h2><p class="legend"><span><i class="dot" style="background:#101828"></i>Actual</span><span><i class="dot" style="background:#155eef"></i>P50</span><span><i class="dot" style="background:#4f7cff"></i>Sum of product P10–P90 bounds</span></p>{chart}<p class="muted">The shaded band sums product intervals. It is a visual aggregate, not a separately calibrated portfolio interval.</p></section>
<section class="panel"><h2>Coverage and width by forecast horizon</h2><div style="overflow:auto"><table><thead><tr><th>Day ahead</th><th>Rows</th><th>Raw coverage</th><th>Calibrated coverage</th><th>Mean width (units)</th></tr></thead><tbody>{horizon_table}</tbody></table></div></section>
<section class="panel"><h2>By demand segment</h2><div style="overflow:auto"><table><thead><tr><th>Segment</th><th>Rows</th><th>Coverage</th><th>Mean width</th><th>P50 MAE</th></tr></thead><tbody>{segment_table}</tbody></table></div></section>
<section class="panel"><h2>Method and limits</h2><p>Base P10/P50/P90 quantiles come from the saved Timestral LoRA adapter. Horizon-specific conformal quantile adjustments are estimated from August residuals; positive adjustments widen intervals and negative adjustments narrow them. Corrections are frozen before later evaluation labels are scored.</p><p class="note">This historical evaluation is diagnostic, not a pristine blind benchmark. Earlier project work examined some dates and product subsets in these periods. SKU and date dependence also weakens formal conformal guarantees. Forecasts model recorded sales; stockouts and product discontinuations are not observed.</p></section>
<p class="muted">Machine-readable results: scorecard.json · per-product forecasts: evaluation-intervals.csv · calibration corrections: calibration.json</p></main></body></html>''', encoding="utf-8")
    (ROOT / "reports/milestone2-uncertainty.html").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8"
    )


def main() -> None:
    calibration = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    if calibration["model"] != CONFIG["model"]:
        raise ValueError("Calibration was produced for a different model")
    eval_rows = read_forecasts(OUT / "evaluation-forecasts.csv")
    end = date.fromisoformat(CONFIG["evaluation"]["target_end"])
    # Do not touch the later evaluation labels until all forecast keys are verified.
    actuals = load_actuals(end)
    corrections = calibration["horizon_adjustments"]
    scored = []
    for row in eval_rows:
        key = (row["stock_code"], row["target_date"])
        actual = actuals.get(key, 0.0)
        qhat = float(corrections[str(row["horizon"])]["adjustment_units"])
        row.update({
            "actual": actual,
            "calibrated_lower": min(row["p50"], max(0.0, row["p10"] - qhat)),
            "calibrated_upper": max(row["p50"], row["p90"] + qhat),
        })
        scored.append(row)

    overall = summarize(scored)
    by_horizon = []
    for horizon in range(1, int(CONFIG["horizon_days"]) + 1):
        by_horizon.append({"horizon": horizon, **summarize([r for r in scored if r["horizon"] == horizon])})
    segments = {
        "Intermittent (≤13 sale days/90d)": summarize([r for r in scored if r["context_sale_days"] <= 13]),
        "Active (≥14 sale days/90d)": summarize([r for r in scored if r["context_sale_days"] >= 14]),
    }
    by_date_map = defaultdict(lambda: {"actual": 0.0, "p50": 0.0, "lower": 0.0, "upper": 0.0})
    for row in scored:
        total = by_date_map[row["target_date"]]
        for name, source in (("actual", "actual"), ("p50", "p50"),
                             ("lower", "calibrated_lower"), ("upper", "calibrated_upper")):
            total[name] += row[source]
    date_rows = [{"target_date": day, **values} for day, values in sorted(by_date_map.items())]
    result = {
        "model": CONFIG["model"], "nominal_coverage": CONFIG["interval_coverage_target"],
        "calibration_period": [CONFIG["calibration"]["target_start"], CONFIG["calibration"]["target_end"]],
        "evaluation_period": [CONFIG["evaluation"]["target_start"], CONFIG["evaluation"]["target_end"]],
        "forecast_origin": CONFIG["evaluation"]["forecast_origin"],
        "context_days": CONFIG["context_days"], "horizon_days": CONFIG["horizon_days"],
        "overall": overall, "by_horizon": by_horizon, "segments": segments,
        "limitations": CONFIG["limitations"],
    }
    (OUT / "scorecard.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = ["model", "stock_code", "forecast_origin", "target_date", "horizon",
              "context_sale_days", "actual", "p10", "p50", "p90",
              "calibrated_lower", "calibrated_upper"]
    with (OUT / "evaluation-intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in scored)
    write_report(result, by_horizon, date_rows)
    print(json.dumps(result, indent=2))
    print(f"HTML report: {OUT / 'milestone2-uncertainty-report.html'}")
    print(f"Shareable report: {ROOT / 'reports/milestone2-uncertainty.html'}")


if __name__ == "__main__":
    main()
