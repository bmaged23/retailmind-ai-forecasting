"""Run and report the Milestone 3 inventory policy comparison."""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

from retailmind.inventory import run_simulation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/milestone3_inventory.json"
OUT = ROOT / "artifacts/milestone3_inventory/timestral209m_lora"
POLICY_LABELS = {
    "timestral_p50_plus_p90_buffer": "Timestral P50 + P90 buffer",
    "seasonal_naive_order_up_to": "Seasonal naive order-up-to",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def money(value):
    return f"£{value:,.2f}"


def build_html(result: dict, summaries: list[dict], daily_rows: list[dict], recommendations: list[dict]) -> str:
    policy_results = result["policies"]
    policies = list(policy_results)
    palette = {policies[0]: "#155eef", policies[1]: "#f79009"}
    daily_unmet = defaultdict(lambda: {policy: 0.0 for policy in policies})
    for row in daily_rows:
        daily_unmet[row["target_date"]][row["policy"]] += row["unmet_units"]
    dates = sorted(daily_unmet)
    max_y = max(1.0, max(v for item in daily_unmet.values() for v in item.values()))
    left, top, plot_w, plot_h, width, height = 68, 16, 910, 250, 1020, 320
    x = lambda i: left + i * plot_w / max(1, len(dates) - 1)
    y = lambda value: top + plot_h * (1 - value / max_y)
    chart_lines = []
    for policy in policies:
        points = " ".join(
            f"{x(i):.1f},{y(daily_unmet[day][policy]):.1f}"
            for i, day in enumerate(dates)
        )
        chart_lines.append(
            f'<polyline points="{points}" fill="none" stroke="{palette[policy]}" stroke-width="2.5"/>'
        )
    tick_lines = []
    for step in range(5):
        value = max_y * (1 - step / 4)
        yy = top + plot_h * step / 4
        tick_lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5ebf3"/>'
            f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" fill="#667085" font-size="11">{value:.0f}</text>'
        )
    x_labels = "".join(
        f'<text x="{x(i):.1f}" y="{top+plot_h+24}" text-anchor="middle" fill="#667085" font-size="11">{html.escape(day[5:])}</text>'
        for i, day in enumerate(dates) if i % 4 == 0 or i == len(dates) - 1
    )
    chart = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Daily unmet recorded sales by inventory policy">'
        + "".join(tick_lines) + "".join(chart_lines) + x_labels + "</svg>"
    )

    overall_rows = []
    for policy in policies:
        row = policy_results[policy]
        label = POLICY_LABELS[policy]
        overall_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{pct(row['fill_rate'])}</td>"
            f"<td>{row['unmet_units']:,.0f}</td><td>{row['products_with_stockout']:,}</td>"
            f"<td>{row['product_stockout_days']:,}</td><td>{row['total_new_order_units']:,}</td>"
            f"<td>{money(row['total_cost_gbp'])}</td></tr>"
        )
    segment_rows = []
    for segment, label in (("active", "Active (≥14 sale days / 90d)"),
                           ("intermittent", "Intermittent (≤13 sale days / 90d)")):
        for policy in policies:
            row = result["segments"][segment][policy]
            segment_rows.append(
                f"<tr><td>{label}</td><td>{html.escape(POLICY_LABELS[policy])}</td>"
                f"<td>{row['products']:,}</td><td>{pct(row['fill_rate'])}</td>"
                f"<td>{row['unmet_units']:,.0f}</td><td>{row['products_with_stockout']:,}</td>"
                f"<td>{money(row['total_cost_gbp'])}</td></tr>"
            )
    top_recs = sorted(recommendations, key=lambda row: row["timestral_order_units"], reverse=True)[:12]
    rec_rows = "".join(
        f"<tr><td>{html.escape(row['stock_code'])}</td><td>{row['starting_on_hand']:,}</td>"
        f"<td>{row['in_transit_units']:,}</td><td>{row['lead_time_days']}</td>"
        f"<td>{row['protection_period_days']}</td><td>{row['timestral_order_units']:,}</td>"
        f"<td>{row['seasonal_naive_order_units']:,}</td></tr>"
        for row in top_recs
    )
    forecast = policy_results[policies[0]]
    naive = policy_results[policies[1]]
    assumptions = result["assumptions"]
    cost = assumptions["cost_assumptions_gbp"]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailMind | Inventory Optimization</title>
<style>body{{margin:0;background:#f3f6fb;color:#17253c;font:15px/1.5 system-ui,Segoe UI,sans-serif}}main{{max-width:1320px;margin:auto;padding:32px 22px}}h1{{font-size:36px;margin:4px 0}}h2{{font-size:19px;margin:0 0 14px}}.muted{{color:#667085}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}}.card,.panel{{background:white;border:1px solid #dce4ef;border-radius:15px;padding:18px;box-shadow:0 4px 16px #12233d0a}}.card small{{display:block;color:#667085}}.card strong{{font-size:24px}}.panel{{margin:14px 0}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;min-width:750px}}td,th{{padding:9px;border-bottom:1px solid #e7edf5;text-align:right;font-variant-numeric:tabular-nums}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}.legend span{{margin-right:18px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}.note{{background:#fff8e7;border-left:4px solid #e6a700;padding:12px 14px;color:#594500}}.assumptions{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}.assumptions div{{background:#f7f9fc;border-radius:9px;padding:10px}}@media(max-width:720px){{.cards{{grid-template-columns:repeat(2,1fr)}}.assumptions{{grid-template-columns:1fr}}main{{padding:20px 12px}}}}</style></head><body><main>
<p class="muted">MILESTONE 3 · INVENTORY OPTIMIZATION</p><h1>One-cycle inventory policy simulation</h1><p class="muted">Timestral 209M LoRA · {result['products']:,} products · {result['evaluation_period'][0]}–{result['evaluation_period'][1]} · forecast origin {result['forecast_origin']}</p>
<section class="cards"><div class="card"><small>Timestral policy fill rate</small><strong>{pct(forecast['fill_rate'])}</strong></div><div class="card"><small>Seasonal naive fill rate</small><strong>{pct(naive['fill_rate'])}</strong></div><div class="card"><small>Timestral simulated cost</small><strong>{money(forecast['total_cost_gbp'])}</strong></div><div class="card"><small>Seasonal naive simulated cost</small><strong>{money(naive['total_cost_gbp'])}</strong></div></section>
<section class="panel"><h2>Daily unmet recorded sales</h2><p class="legend"><span><i class="dot" style="background:{palette[policies[0]]}"></i>{POLICY_LABELS[policies[0]]}</span><span><i class="dot" style="background:{palette[policies[1]]}"></i>{POLICY_LABELS[policies[1]]}</span></p>{chart}<p class="muted">The gap represents recorded sales not served by the simulated stock policy; actual lost demand is unknown.</p></section>
<section class="panel"><h2>Portfolio comparison</h2><div class="scroll"><table><thead><tr><th>Policy</th><th>Fill rate</th><th>Unmet units</th><th>Products with stockout</th><th>Product stockout days</th><th>New order units</th><th>Simulated total cost</th></tr></thead><tbody>{''.join(overall_rows)}</tbody></table></div><p class="muted">Cost = holding + order placement + shortage penalty. Purchase cost is excluded.</p></section>
<section class="panel"><h2>By demand segment</h2><div class="scroll"><table><thead><tr><th>Segment</th><th>Policy</th><th>Products</th><th>Fill rate</th><th>Unmet units</th><th>Products with stockout</th><th>Simulated cost</th></tr></thead><tbody>{''.join(segment_rows)}</tbody></table></div></section>
<section class="panel"><h2>Largest Timestral order recommendations</h2><div class="scroll"><table><thead><tr><th>Product</th><th>On hand</th><th>In transit</th><th>Lead days</th><th>Protection days</th><th>Timestral order</th><th>Naive order</th></tr></thead><tbody>{rec_rows}</tbody></table></div><p class="muted">Full product-level recommendations are in recommendations.csv.</p></section>
<section class="panel"><h2>Scenario assumptions</h2><div class="assumptions"><div>Initial stock: uniform {assumptions['initial_stock_coverage_days_range'][0]}–{assumptions['initial_stock_coverage_days_range'][1]} days of 90-day average sales</div><div>Supplier lead time: uniform {assumptions['supplier_lead_time_days_range'][0]}–{assumptions['supplier_lead_time_days_range'][1]} days</div><div>Existing in-transit order: {assumptions['incoming_order_probability']:.0%} chance; {assumptions['incoming_order_coverage_weeks_range'][0]}–{assumptions['incoming_order_coverage_weeks_range'][1]} weeks of average sales</div><div>Review period: {assumptions['review_period_days']} days; one new order at the forecast origin</div><div>Holding: £{cost['holding_cost_per_unit_day']:.2f}/unit/day; order: £{cost['order_cost_per_order']:.2f}</div><div>Shortage penalty: £{cost['shortage_penalty_per_unmet_unit']:.2f}/unmet unit; seed {assumptions['scenario_seed']}</div></div></section>
<section class="panel"><h2>Interpretation and limits</h2><p>The Timestral policy sets target stock to forecast P50 demand over lead time plus the seven-day review period, with the sum of daily P90–P50 gaps as a safety-stock heuristic. The comparator repeats the last seven observed daily sales and orders up to that expected demand without a safety buffer. Both share the same starting stock, outstanding order, lead time, recorded-sales path, and cost assumptions.</p><p class="note">All stock, supplier, and cost inputs are simulated. The replay uses historical recorded sales as a demand proxy; it cannot reveal lost sales during stockouts. Summed daily P90 bounds are not calibrated aggregate quantiles. The policy places one order at the origin and does not reforecast during the 28-day replay. Treat the result as a policy prototype, not measured business savings.</p></section>
<p class="muted">Detailed outputs: scorecard.json · policy-comparison.csv · recommendations.csv · daily-ledger.csv</p></main></body></html>'''


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result, summaries, daily_rows, recommendations = run_simulation(ROOT, config)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorecard.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "policy-comparison.csv", summaries)
    write_csv(OUT / "recommendations.csv", recommendations)
    write_csv(OUT / "daily-ledger.csv", daily_rows)
    report = build_html(result, summaries, daily_rows, recommendations)
    (OUT / "milestone3-inventory.html").write_text(report, encoding="utf-8")
    (ROOT / "reports/milestone3-inventory.html").write_text(report, encoding="utf-8")
    print(json.dumps(result["policies"], indent=2))
    print(f"Scorecard: {OUT / 'scorecard.json'}")
    print(f"Recommendations: {OUT / 'recommendations.csv'}")
    print(f"HTML report: {ROOT / 'reports/milestone3-inventory.html'}")


if __name__ == "__main__":
    main()
