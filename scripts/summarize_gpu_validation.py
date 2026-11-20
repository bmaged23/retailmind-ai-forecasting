"""Summarize GPU validation without conflating shared training and per-SKU runs."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'reports/generated/validation-enriched-gpu'
    result={}
    for group in ('active','intermittent'):
        folder=base/group
        f=pd.read_csv(folder/'forecasts.csv');m=pd.read_csv(folder/'metrics.csv')
        metadata=json.loads((folder/'metadata.json').read_text())
        models={}
        for model, rows in f.groupby('model'):
            error=(rows.actual-rows.prediction).abs(); actual=rows.actual.abs().sum()
            models[model]={'mae':float(error.mean()),'wape_pct':float(100*error.sum()/actual) if actual else None,
                           'median_forecast_ms':float(m[m.model==model].inference_seconds.median()*1000)}
        folds=[]
        for fold in metadata['folds']:
            record={'cutoff':fold['cutoff'],'scored_products':fold['scored_products'],'cohort_size':fold['cohort_size'],'excluded':fold['excluded']}
            for model in ('lstm','lstm_enriched'):
                secs=fold[f'{model}_train_seconds'];stats=fold[f'{model}_gpu_training_samples']
                record[model]={'training_seconds':secs,'seconds_per_cohort_product':secs/fold['cohort_size'],
                               'best_epoch':fold[f'{model}_best_epoch'], 'epochs_run':fold[f'{model}_epochs_run'], 'gpu_samples':stats}
            folds.append(record)
        result[group]={'models':models,'folds':folds,'gpu_device_sampling':metadata['gpu_stats'],
                       'batch_size':metadata['batch_size_override']}
    out=base/'summary.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
