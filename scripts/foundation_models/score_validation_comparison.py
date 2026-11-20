"""Score validation forecasts after every configured candidate is frozen."""
from collections import defaultdict
from datetime import date, timedelta
import csv
import json
import math
from pathlib import Path
import statistics
from html import escape

from retailmind.forecast import load_panel

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/validation_comparison.json"
OUT = ROOT / "artifacts/validation_comparison_2011-07-01_to_2011-07-28"
FORECASTS = OUT / "forecasts"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cohort_path = OUT / "cohort.json"
    if not cohort_path.exists():
        raise FileNotFoundError("Run prepare_validation_cohort.py first")
    cohort_info = json.loads(cohort_path.read_text(encoding="utf-8"))
    codes = cohort_info["codes"]
    origin = date.fromisoformat(config["selection"]["forecast_origin"])
    horizon = int(config["horizon_days"])
    keys = {
        (code, (origin + timedelta(days=step)).isoformat())
        for code in codes
        for step in range(1, horizon + 1)
    }

    paths = {name: FORECASTS / f"{name}.csv" for name in config["candidates"]}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Refusing to score a partial comparison; missing forecasts:\n"
            + "\n".join(missing)
        )

    predictions = {}
    latencies = {}
    for name, path in paths.items():
        by_key = {}
        per_product = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["stock_code"], row["target_date"])
                if key in by_key:
                    raise ValueError(f"Duplicate forecast row in {path}: {key}")
                by_key[key] = max(0.0, float(row["prediction"]))
                raw_latency = row.get("inference_seconds_per_series")
                if raw_latency:
                    per_product[key[0]].append(float(raw_latency))
        if set(by_key) != keys:
            missing_keys = len(keys - set(by_key))
            extra_keys = len(set(by_key) - keys)
            raise ValueError(
                f"{name} has mismatched forecast keys: "
                f"{missing_keys} missing, {extra_keys} extra"
            )
        predictions[name] = by_key
        latencies[name] = (
            statistics.mean(statistics.mean(values) for values in per_product.values())
            if per_product
            else None
        )

    # Labels are loaded only after all model forecast files and key sets pass.
    panel = load_panel(ROOT / config["data"])
    actuals = {
        key: float(panel.get(key[0], {}).get(date.fromisoformat(key[1]), 0.0))
        for key in keys
    }
    total_actual = sum(actuals.values())
    dates = sorted({day for _, day in keys})
    actual_by_date = {
        day: sum(value for (code, target_day), value in actuals.items() if target_day == day)
        for day in dates
    }
    forecast_by_date = {
        name: {
            day: sum(value for (code, target_day), value in by_key.items() if target_day == day)
            for day in dates
        }
        for name, by_key in predictions.items()
    }
    scores = []
    for name, by_key in predictions.items():
        errors = [actuals[key] - by_key[key] for key in keys]
        absolute = [abs(error) for error in errors]
        scores.append(
            {
                "model": name,
                "products": len(codes),
                "forecast_rows": len(keys),
                "mae": statistics.mean(absolute),
                "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
                "wape": sum(absolute) / total_actual if total_actual else None,
                "mean_inference_seconds_per_product": latencies[name],
            }
        )
    scores.sort(key=lambda row: row["mae"])
    result = {
        "config": "configs/validation_comparison.json",
        "cohort_file": "artifacts/validation_comparison_2011-07-01_to_2011-07-28/cohort.json",
        "forecast_origin": origin.isoformat(),
        "selection_period": [
            config["selection"]["target_start"],
            config["selection"]["target_end"],
        ],
        "products": len(codes),
        "rows_per_model": len(keys),
        "scored_only_after_all_forecasts_frozen": True,
        "models": scores,
        "selection_policy": config["selection_policy"],
        "target_assumption": config["selection"]["missing_transaction_dates"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scorecard.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = OUT / "validation-comparison.html"
    write_html(report, result, dates, actual_by_date, forecast_by_date)
    print(json.dumps(result, indent=2))
    print(f"Scorecard: {path}")
    print(f"HTML report: {report}")


def write_html(path: Path, result: dict, dates: list[str], actual: dict, forecasts: dict) -> None:
    import json

    payload = {
        "dates": dates,
        "actual": [actual[day] for day in dates],
        "forecasts": {name: [values[day] for day in dates] for name, values in forecasts.items()},
        "models": result["models"],
    }
    embedded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    rows = "\n".join(
        "<tr>"
        f"<td>{rank}</td><td>{escape(model['model'])}</td>"
        f"<td>{model['mae']:.4f}</td><td>{model['rmse']:.4f}</td>"
        f"<td>{model['wape']:.2%}</td>"
        f"<td>{model['mean_inference_seconds_per_product']:.6f}</td>"
        "</tr>"
        for rank, model in enumerate(result["models"], 1)
    )
    path.write_text(
        f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RetailMind | Validation Model Selection</title>
<style>
:root{{--bg:#f3f6fb;--ink:#152238;--muted:#64748b;--line:#dce4ef;--panel:#fff;--accent:#155eef}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1440px;margin:0 auto;padding:34px 24px 60px}}header{{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:24px}}
h1{{font-size:clamp(28px,4vw,42px);line-height:1.08;margin:0 0 8px;letter-spacing:-.035em}}h2{{font-size:18px;margin:0 0 16px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:800;color:var(--accent)}}.sub{{color:var(--muted);margin:0}}
.pill{{background:#e8f0ff;color:#1246a0;padding:8px 12px;border-radius:999px;font-weight:700;white-space:nowrap}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 5px 18px #182c4a0a}}
.card{{padding:17px 18px}}.card small{{display:block;color:var(--muted);font-weight:700}}.card strong{{display:block;font-size:25px;margin-top:3px;letter-spacing:-.03em}}
.panel{{padding:22px;margin:16px 0}}.chart-wrap{{height:410px;overflow:hidden}}svg{{width:100%;height:100%;overflow:visible}}.axis{{stroke:#d5deea;stroke-width:1}}.grid{{stroke:#e9eef5;stroke-width:1;stroke-dasharray:3 5}}.tick{{fill:#718096;font-size:11px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 16px}}.toggle{{font-size:12px;border:1px solid var(--line);padding:5px 9px;border-radius:999px;background:#fff;cursor:pointer;user-select:none}}.toggle input{{vertical-align:middle;margin:0 5px 0 0;accent-color:var(--accent)}}
.legend-note{{font-size:12px;color:var(--muted);margin:0 0 10px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:850px}}th,td{{padding:11px 12px;text-align:right;border-bottom:1px solid #edf1f6;font-variant-numeric:tabular-nums}}th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}td:nth-child(2),th:nth-child(2){{text-align:left}}tbody tr:first-child{{background:#f1f6ff;font-weight:750}}
.protocol{{color:var(--muted);font-size:13px}}.protocol strong{{color:var(--ink)}}footer{{color:var(--muted);font-size:12px;padding:12px 2px}}
@media(max-width:760px){{main{{padding:22px 14px 40px}}header{{align-items:flex-start;flex-direction:column}}.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.panel{{padding:16px}}.chart-wrap{{height:330px}}}}
</style></head><body><main>
<header><div><div class="eyebrow">RetailMind • Model selection</div><h1>Daily portfolio forecast</h1><p class="sub">A leakage-safe comparison on one frozen cohort and one July horizon.</p></div><div class="pill">{escape(result['selection_period'][0])} — {escape(result['selection_period'][1])}</div></header>
<section class="cards"><div class="card"><small>Products scored</small><strong>{result['products']:,}</strong></div><div class="card"><small>Forecast rows per model</small><strong>{result['rows_per_model']:,}</strong></div><div class="card"><small>Models compared</small><strong>{len(result['models'])}</strong></div><div class="card"><small>Best pooled MAE</small><strong>{result['models'][0]['mae']:.3f}</strong></div></section>
<section class="panel"><h2>Portfolio totals by day</h2><p class="legend-note">Sum of daily units across the complete cohort. Select model lines to compare them with recorded July sales.</p><div class="controls" id="controls"></div><div class="chart-wrap"><svg id="chart" role="img" aria-label="Daily actual portfolio sales and model forecasts"></svg></div></section>
<section class="panel"><h2>Ranked scorecard</h2><p class="legend-note">Primary rank: pooled MAE. RMSE and WAPE are guardrails. Latency is average inference seconds per product.</p><div class="scroll"><table><thead><tr><th>Rank</th><th>Model</th><th>MAE ↓</th><th>RMSE</th><th>WAPE</th><th>Inference sec/product</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section class="panel protocol"><h2>Protocol and interpretation</h2><p>Fit targets end <strong>2011-04-30</strong>. Checkpoint validation uses <strong>2011-05-01 through 2011-05-28</strong>. All models forecast from <strong>2011-06-30</strong> over the same {result['products']:,} products and 28 target dates. July labels were loaded only after every forecast passed exact key checks.</p><p>Missing transaction dates are scored as zero recorded sales; this does not distinguish no demand from unavailable or discontinued products. The October split is excluded from model selection because it was previously reused.</p></section>
<footer>Generated by scripts/foundation_models/score_validation_comparison.py • Pooled metrics over {result['rows_per_model']:,} product-day rows per model</footer>
</main><script>const DATA={embedded};
const palette=['#155eef','#f04438','#12b76a','#7a5af8','#f79009','#06aed4','#d444f1','#344054','#c11574','#039855','#b54708','#026aa2','#6938ef','#1570ef','#667085'];
const controls=document.getElementById('controls');
DATA.models.forEach((m,i)=>{{const label=document.createElement('label');label.className='toggle';label.innerHTML='<input type="checkbox" checked data-model="'+m.model+'"><span style="color:'+palette[i%palette.length]+'">'+m.model+'</span>';controls.appendChild(label)}});
const svg=document.getElementById('chart'),NS='http://www.w3.org/2000/svg';
function el(tag,attrs,text){{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{{}}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}}
function render(){{const W=Math.max(700,svg.clientWidth),H=svg.clientHeight||410,L=66,R=20,T=18,B=42,iw=W-L-R,ih=H-T-B;svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);svg.replaceChildren();
const checked=[...controls.querySelectorAll('input:checked')].map(x=>x.dataset.model);let vals=DATA.actual.slice();checked.forEach(n=>vals=vals.concat(DATA.forecasts[n]||[]));let max=Math.max(1,...vals)*1.08;
for(let i=0;i<=4;i++){{let y=T+ih*i/4,v=max*(1-i/4);svg.append(el('line',{{x1:L,y1:y,x2:W-R,y2:y,class:'grid'}}));svg.append(el('text',{{x:L-9,y:y+4,'text-anchor':'end',class:'tick'}},Math.round(v).toLocaleString()))}}
const x=i=>L+(DATA.dates.length<=1?0:i*iw/(DATA.dates.length-1)),y=v=>T+ih*(1-v/max);
svg.append(el('line',{{x1:L,y1:T+ih,x2:W-R,y2:T+ih,class:'axis'}}));
DATA.dates.forEach((d,i)=>{{if(i%4===0||i===DATA.dates.length-1)svg.append(el('text',{{x:x(i),y:H-13,'text-anchor':'middle',class:'tick'}},d.slice(5)))}});
function line(series,color,width,dash){{const points=series.map((v,i)=>`${{x(i)}},${{y(v)}}`).join(' ');svg.append(el('polyline',{{points,fill:'none',stroke:color,'stroke-width':width,'stroke-linejoin':'round','stroke-linecap':'round','stroke-dasharray':dash||''}}))}}
line(DATA.actual,'#101828',3.5,'');checked.forEach(n=>{{let i=DATA.models.findIndex(m=>m.model===n);line(DATA.forecasts[n],palette[i%palette.length],2,'')}});
}}
controls.addEventListener('change',render);new ResizeObserver(render).observe(svg);render();</script></body></html>''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
