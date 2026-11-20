"""Compare Timestral, LSTM, and TimesFM 3.0 reorder plans under shared scenarios."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from retailmind.scenarios import LABELS, MODELS, run_monte_carlo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/milestone4_scenarios.json"
OUT = ROOT / "artifacts/milestone4_scenarios/model_plan_comparison"
COLORS = {MODELS[0]: "#155eef", MODELS[1]: "#12b76a", MODELS[2]: "#f79009"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value:.1%}"


def gbp(value: float) -> str:
    return f"£{value:,.0f}"


def build_report(result: dict, paths: list[dict], daily: list[dict], factors: list[dict], plan_summary: list[dict]) -> str:
    models = list(MODELS)
    summary = result["models"]
    dates = [r["date"] for r in daily if r["model"] == models[0]]
    left, top, pw, ph, width, height = 70, 16, 900, 230, 1010, 292
    max_y = max(1.0, max(r["mean_unmet_units"] for r in daily))
    svg = []
    for i in range(5):
        val = max_y * (1 - i/4)
        y = top + ph*i/4
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e5ebf3"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" fill="#667085" font-size="11">{val:.0f}</text>')
    for model in models:
        vals = [r["mean_unmet_units"] for r in daily if r["model"] == model]
        points = " ".join(f"{left+j*pw/max(1,len(vals)-1):.1f},{top+ph*(1-v/max_y):.1f}" for j,v in enumerate(vals))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[model]}" stroke-width="2.8"/>')
    for i in range(0, len(dates), 4):
        x = left+i*pw/max(1,len(dates)-1)
        svg.append(f'<text x="{x:.1f}" y="{top+ph+22}" text-anchor="middle" fill="#667085" font-size="10">{html.escape(str(dates[i])[5:])}</text>')
    x = left+pw
    svg.append(f'<text x="{x:.1f}" y="{top+ph+22}" text-anchor="middle" fill="#667085" font-size="10">{html.escape(str(dates[-1])[5:])}</text>')
    chart = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Daily average unmet sales under model-specific reorder plans">{"".join(svg)}</svg>'
    legend = "".join(f'<span><i class="dot" style="background:{COLORS[m]}"></i>{LABELS[m]}</span>' for m in models)

    forecasts_by_model = {r["model"]: r for r in plan_summary}
    plan_rows = []
    for model in models:
        p = forecasts_by_model[model]
        s = summary[model]
        plan_rows.append(
            f"<tr><td>{LABELS[model]}</td><td>{p['forecast_mae']:.3f}</td>"
            f"<td>{p['mean_forecast_demand_28d']:,.1f}</td><td>{p['products_ordered']:,}</td>"
            f"<td>{p['total_order_units']:,.0f}</td><td>{pct(s['mean_fill_rate'])}</td>"
            f"<td>{s['mean_unmet_units']:,.0f}</td><td>{gbp(s['mean_total_cost_gbp'])}</td></tr>"
        )
    outcome_rows = []
    for model in models:
        s = summary[model]
        outcome_rows.append(
            f"<tr><td>{LABELS[model]}</td><td>{s['mean_demand_units']:,.0f}</td>"
            f"<td>{pct(s['mean_fill_rate'])}</td><td>{pct(s['p05_fill_rate'])}–{pct(s['p95_fill_rate'])}</td>"
            f"<td>{s['mean_unmet_units']:,.0f}</td><td>{pct(s['mean_shortage_sku_fraction'])}</td>"
            f"<td>{s['mean_average_inventory_units']:,.1f}</td><td>{gbp(s['mean_holding_cost_gbp'])}</td>"
            f"<td>{gbp(s['mean_order_cost_gbp'])}</td><td>{gbp(s['mean_shortage_cost_gbp'])}</td>"
            f"<td>{gbp(s['mean_total_cost_gbp'])} ({gbp(s['p95_total_cost_gbp'])})</td></tr>"
        )
    factor_rows = []
    labeler = {
        "demand_spike": lambda x: f"Demand ×{float(x):.2f}",
        "supplier_delay": lambda x: f"Supplier delay +{int(float(x))} days",
        "starting_stock_loss": lambda x: f"Starting stock loss {float(x):.0%}",
    }
    for r in factors:
        factor_rows.append(
            f"<tr><td>{labeler[r['factor']](r['value'])}</td><td>{LABELS[r['model']]}</td>"
            f"<td>{r['scenario_count']:,}</td><td>{pct(r['mean_fill_rate'])}</td>"
            f"<td>{r['mean_unmet_units']:,.0f}</td><td>{gbp(r['mean_total_cost_gbp'])}</td>"
            f"<td>{gbp(r['p95_total_cost_gbp'])}</td></tr>"
        )
    probabilities = result["scenario_probabilities"]
    dist = lambda d, suffix: ", ".join(f"{k}{suffix}: {v:.0%}" for k, v in d.items())
    cards = []
    for m in models:
        s = summary[m]
        cards.append(f'<div class="card"><small>{LABELS[m]} mean fill</small><strong>{pct(s["mean_fill_rate"])}</strong><small>5th–95th: {pct(s["p05_fill_rate"])}–{pct(s["p95_fill_rate"])}</small></div>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailMind | Three-model inventory stress test</title>
<style>body{{margin:0;background:#f3f6fb;color:#17253c;font:15px/1.5 system-ui,Segoe UI,sans-serif}}main{{max-width:1400px;margin:auto;padding:30px 20px}}h1{{font-size:35px;margin:2px 0}}h2{{font-size:19px;margin:0 0 12px}}.muted{{color:#667085}}.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:20px 0}}.card,.panel{{background:white;border:1px solid #dce4ef;border-radius:15px;padding:17px;box-shadow:0 4px 16px #12233d0a}}.card small{{display:block;color:#667085}}.card strong{{font-size:25px}}.panel{{margin:14px 0}}svg{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:970px}}td,th{{padding:9px;border-bottom:1px solid #e7edf5;text-align:right;font-variant-numeric:tabular-nums}}td:first-child,th:first-child{{text-align:left}}.legend span{{margin-right:16px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}.note{{background:#fff8e7;border-left:4px solid #e6a700;padding:12px 14px;color:#594500}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.grid div{{background:#f7f9fc;border-radius:9px;padding:11px}}@media(max-width:760px){{.cards{{grid-template-columns:1fr}}.grid{{grid-template-columns:1fr}}main{{padding:18px 12px}}}}</style></head><body><main>
<p class="muted">MILESTONE 4 · PAIRED MODEL-PLAN STRESS TEST</p><h1>Timestral vs LSTM vs Google TimesFM 3.0</h1><p class="muted">{result['products']:,} common products · {result['days']} days · {result['simulation_count']:,} paired futures · origin {result['forecast_origin']} · {result['evaluation_period'][0]}–{result['evaluation_period'][1]}</p>
<section class="cards">{''.join(cards)}</section>
<section class="panel"><h2>Forecast accuracy and resulting order plan</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Forecast MAE</th><th>Mean 28-day forecast / SKU</th><th>Products ordered</th><th>Total order units</th><th>Stress-test fill</th><th>Mean unmet units</th><th>Mean modeled cost</th></tr></thead><tbody>{''.join(plan_rows)}</tbody></table></div><p class="muted">All three plans use the same order-up-to rule and a 20% safety buffer over forecast demand during lead time plus the seven-day review period. Each receives identical starting inventory and pipeline orders.</p></section>
<section class="panel"><h2>Mean unmet recorded sales by day</h2><p class="legend">{legend}</p>{chart}<p class="muted">Daily average over the 500 shared scenarios; all models face the same sales path and disruptions.</p></section>
<section class="panel"><h2>Portfolio outcomes under shared scenarios</h2><div class="scroll"><table><thead><tr><th>Model-specific plan</th><th>Mean demand</th><th>Mean fill</th><th>5th–95th fill</th><th>Mean unmet units</th><th>Mean SKU shortage rate</th><th>Mean on-hand units</th><th>Holding cost</th><th>Order cost</th><th>Shortage cost</th><th>Mean total (95th percentile)</th></tr></thead><tbody>{''.join(outcome_rows)}</tbody></table></div><p class="muted">SKU shortage rate is the share of products with at least one unmet-demand day per future. Unit purchase costs are excluded.</p></section>
<section class="panel"><h2>Stress-factor breakdown</h2><div class="scroll"><table><thead><tr><th>Condition</th><th>Model-specific plan</th><th>Futures</th><th>Mean fill</th><th>Mean unmet units</th><th>Mean total cost</th><th>95th percentile cost</th></tr></thead><tbody>{''.join(factor_rows)}</tbody></table></div><p class="muted">Rows condition on one factor while the other two continue to vary. Small tail groups have high Monte Carlo uncertainty.</p></section>
<section class="panel"><h2>Shared simulation assumptions</h2><div class="grid"><div><b>Demand multiplier</b><br>{dist(probabilities['demand_spike_multipliers'],'×')}</div><div><b>Supplier delay added to all deliveries</b><br>{dist(probabilities['supplier_delay_days'],' days')}</div><div><b>Starting on-hand stock loss</b><br>{dist(probabilities['starting_stock_loss_fractions'],'%')}</div></div><p>Seed {result['scenario_seed']}. Scenario seed {result['inventory_scenario_seed']} creates common synthetic opening stock, inbound orders, and lead times.</p><p class="note">The October comparison window has been reused in earlier model evaluation. These stress-test results are descriptive, not a pristine blind benchmark. Stock, supplier, and cost inputs are synthetic. Demand is the recorded sales path scaled by shared scenario multipliers; stockout-censored demand is unknown. Cost rankings depend on illustrative cost weights.</p></section>
<p class="muted">Detailed outputs: <code>model-plan-summary.csv</code> · <code>product-plan-recommendations.csv</code> · <code>scenario-results.csv</code> · <code>daily-expected-unmet.csv</code> · <code>stress-factor-results.csv</code></p></main></body></html>'''


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scorecard, paths, daily, factors, recommendations = run_monte_carlo(ROOT, config)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "scenario-results.csv", paths)
    write_csv(OUT / "daily-expected-unmet.csv", daily)
    write_csv(OUT / "stress-factor-results.csv", factors)
    write_csv(OUT / "product-plan-recommendations.csv", recommendations)
    plan_summary = [{k: v for k, v in row.items() if k in {
        "model", "model_label", "products", "products_ordered", "total_order_units",
        "mean_order_units_per_product", "mean_forecast_demand_28d", "forecast_mae",
    }} for row in scorecard["plan_summary"]]
    write_csv(OUT / "model-plan-summary.csv", plan_summary)
    report = build_report(scorecard, paths, daily, factors, plan_summary)
    (OUT / "milestone4-scenarios.html").write_text(report, encoding="utf-8")
    (ROOT / "reports/milestone4-scenarios.html").write_text(report, encoding="utf-8")
    for model, metrics in scorecard["models"].items():
        print(f"{metrics['label']}: fill={metrics['mean_fill_rate']:.3%}, "
              f"unmet={metrics['mean_unmet_units']:.0f}, "
              f"mean_cost=£{metrics['mean_total_cost_gbp']:,.2f}, "
              f"p95_cost=£{metrics['p95_total_cost_gbp']:,.2f}")
    print(f"Report: {ROOT / 'reports/milestone4-scenarios.html'}")


if __name__ == "__main__":
    main()
