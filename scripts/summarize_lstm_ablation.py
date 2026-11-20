"""Summarize validation ablations; never read or evaluate the test period."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/generated/lstm-feature-ablation'
    forecasts=pd.read_csv(folder/'forecasts.csv'); metrics=pd.read_csv(folder/'metrics.csv')
    meta=json.loads((folder/'metadata.json').read_text()); summary=[]
    for (cohort,variant),data in forecasts.groupby(['cohort','variant']):
        errors=(data.actual-data.prediction).abs();actual=data.actual.abs().sum()
        runs=[r for r in meta['runs'] if r['cohort']==cohort and r['variant']==variant]
        samples=sum(r['gpu_training_samples'].get('samples',0) for r in runs)
        weighted=sum(r['gpu_training_samples'].get('mean_gpu_utilization_percent',0)*r['gpu_training_samples'].get('samples',0) for r in runs)
        summary.append({'cohort':cohort,'variant':variant,'mae':float(errors.mean()),'wape_pct':float(100*errors.sum()/actual) if actual else None,
                        'product_folds':int(len(data)/28),'mean_training_seconds':sum(r['training_seconds'] for r in runs)/len(runs),
                        'mean_training_seconds_per_product':sum(r['seconds_per_training_product'] for r in runs)/len(runs),
                        'mean_gpu_util_pct_sampled':weighted/samples if samples else None,
                        'median_forecast_ms':float(metrics[(metrics.cohort==cohort)&(metrics.variant==variant)].inference_seconds.median()*1000)})
    path=folder/'summary.json';path.write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print('GPU:',meta['device_sampling']);print('PyTorch peak reserved MiB:',meta['torch_peak_reserved_mib']);print('Selected bases:',meta['selected_base_per_cohort'])

if __name__=='__main__':main()
