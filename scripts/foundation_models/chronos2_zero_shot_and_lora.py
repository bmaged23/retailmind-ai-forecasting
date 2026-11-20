"""Chronos-2 zero-shot vs LoRA comparison on the frozen LGBM/LSTM test cohort.

This script writes forecasts only. Test targets are intentionally not loaded here;
scripts/score_foundation_comparison.py joins labels after every model is frozen.
"""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from math import ceil
import csv, json, hashlib
import numpy as np
import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from transformers import EarlyStoppingCallback
from common import RUN_DIR, RUN_VALIDATION, ORIGIN, CONTEXT as SHARED_CONTEXT, HORIZON as SHARED_HORIZON
from retailmind.gpu_monitor import GPUMonitor

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/train-validation-test-2011-11-01'
OUT=RUN_DIR
FORECAST_OUT=OUT/'forecasts/chronos2'
MODEL_OUT=OUT/'models/chronos2/lora_full_series_epoch'
METADATA_OUT=OUT/'metadata'
TRAIN_END=date(2011,4,30) if RUN_VALIDATION else date(2011,8,31)
VALID_END=date(2011,5,28) if RUN_VALIDATION else date(2011,9,28)
TEST_ORIGIN=ORIGIN
CONTEXT=SHARED_CONTEXT
HORIZON=SHARED_HORIZON

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dense_history(series,end):
    available={d:q for d,q in series.items() if d<=end}
    if not available:return None
    first=min(available)
    return np.asarray([available.get(first+timedelta(days=i),0.)
                       for i in range((end-first).days+1)],dtype=np.float32)

def main():
    FORECAST_OUT.mkdir(parents=True,exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True,exist_ok=True)
    METADATA_OUT.mkdir(parents=True,exist_ok=True)
    from common import load_contexts, write_forecasts as write_common_forecasts
    panel,cohort,contexts=load_contexts()
    panel_path=ROOT/'data/processed/daily_sales.csv'
    eligible=cohort
    # These contain no test targets: histories end at the explicit pre-test cutoffs.
    fit_inputs=[];validation_inputs=[]
    for series in panel.values():
        train=dense_history(series,TRAIN_END)
        valid=dense_history(series,VALID_END)
        if train is not None and len(train)>=CONTEXT+HORIZON:fit_inputs.append(train)
        if valid is not None and len(valid)>=CONTEXT+HORIZON:validation_inputs.append(valid)
    # Validation checkpoint selection is sampled deterministically; test scoring
    # still uses the complete frozen 2,693-product cohort.
    rng=np.random.default_rng(42)
    if len(validation_inputs)>256:
        chosen=np.sort(rng.choice(len(validation_inputs),size=256,replace=False))
        validation_inputs=[validation_inputs[int(i)] for i in chosen]
    print(f'Frozen test cohort={len(cohort)}; fit series={len(fit_inputs)}; validation checkpoint sample={len(validation_inputs)}',flush=True)
    monitor=GPUMonitor(max_used_mib=1900).start()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; aborting rather than training on CPU')
    torch.backends.cudnn.benchmark=False
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,1536*1024**2/total),0)
    torch.cuda.reset_peak_memory_stats();monitor.check()
    cache=ROOT/'.cache/huggingface'
    import os
    os.environ['HF_HOME']=str(cache)
    os.environ['HF_HUB_DISABLE_XET']='1'
    print('Loading Chronos-2 on CUDA (zero-shot phase)...',flush=True)
    started=perf_counter()
    base=Chronos2Pipeline.from_pretrained('amazon/chronos-2',device_map='cuda')
    load_seconds=perf_counter()-started
    batch_size=4
    probe=Chronos2Dataset(inputs=fit_inputs,context_length=CONTEXT,
      prediction_length=HORIZON,batch_size=batch_size,
      output_patch_size=base.model_output_patch_size,min_past=CONTEXT,
      mode=DatasetMode.TRAIN)
    training_series_count=len(probe.inputs)
    steps_per_epoch=ceil(training_series_count/batch_size)
    epoch_counter={'value':0}
    def full_series_epoch(dataset):
        rng=np.random.default_rng(42+epoch_counter['value'])
        epoch_counter['value']+=1
        indices=rng.permutation(len(dataset.inputs))
        for offset in range(0,len(indices),dataset.batch_size):
            yield dataset._build_batch(indices[offset:offset+dataset.batch_size].tolist())
    # The pinned Chronos version samples histories with replacement. For this
    # full-training pass, visit every eligible product once per epoch.
    Chronos2Dataset._generate_train_batches=full_series_epoch
    Chronos2Dataset.__len__=lambda dataset: ceil(len(dataset.inputs)/dataset.batch_size)
    print(f'Chronos-2 full-series epochs: {training_series_count} histories, '
          f'{steps_per_epoch} optimizer steps per epoch, batch={batch_size}.',flush=True)
    zero=forecast(base,contexts,monitor)
    if not RUN_VALIDATION:
        write_predictions(FORECAST_OUT/'chronos2_zero_shot.csv',eligible,zero)
    print('Zero-shot forecasts saved. Starting LoRA fine-tuning...',flush=True)
    MODEL_OUT.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    fine=base.fit(inputs=fit_inputs,validation_inputs=validation_inputs,
       prediction_length=HORIZON,context_length=CONTEXT,finetune_mode='lora',
       learning_rate=1e-5,num_steps=steps_per_epoch*8,batch_size=batch_size,min_past=CONTEXT,
       output_dir=MODEL_OUT,finetuned_ckpt_name='adapter',
       # Chronos fit() defaults validation to every 100 steps when validation
       # data is present. Override those defaults here so one patience unit is
       # exactly one complete product-series epoch.
       eval_strategy='epoch',save_strategy='epoch',
       save_total_limit=1,load_best_model_at_end=True,metric_for_best_model='eval_loss',
       callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
       disable_data_parallel=True,remove_printer_callback=True)
    fit_seconds=perf_counter()-started
    trainer_states=[]
    for state_path in MODEL_OUT.glob('checkpoint-*/trainer_state.json'):
        state=json.loads(state_path.read_text())
        trainer_states.append({'checkpoint':state_path.parent.name,
          'best_metric':state.get('best_metric'),'best_model_checkpoint':state.get('best_model_checkpoint'),
          'log_history':state.get('log_history',[])})
    del base
    torch.cuda.empty_cache();monitor.check()
    adapted=forecast(fine,contexts,monitor)
    write_common_forecasts('chronos2_lora',eligible,adapted['values'],adapted['elapsed_seconds'])
    gpu=monitor.finish()
    metadata={'model':'amazon/chronos-2','phase':'zero-shot-and-lora','test_forecasts_only':True,
      'test_targets_loaded':False,'cohort_source':'frozen validation cohort' if RUN_VALIDATION else 'frozen scoreable stock-code set from prior October 5-November 1 LGBM/LSTM evaluation',
      'cohort_products':len(cohort),'train_end':TRAIN_END.isoformat(),'validation_end':VALID_END.isoformat(),
      'test_origin':TEST_ORIGIN.isoformat(),'test_period':['2011-10-05','2011-11-01'],
      'context_days':CONTEXT,'horizon_days':HORIZON,'fit_series':len(fit_inputs),
      'validation_checkpoint_sample_size':len(validation_inputs),'validation_sample_seed':42,
      'finetune_mode':'LoRA','max_epochs':8,'early_stopping_patience_epochs':2,
      'training_series_per_epoch':training_series_count,'optimizer_steps_per_epoch':steps_per_epoch,
      'training_sampling':'Each eligible training product is visited once per epoch; one training window is sampled from that product.',
      'learning_rate':1e-5,'batch_size':batch_size,'best_checkpoint_selected_on_validation':True,
      'validation_early_stopping_metric':'eval_loss','validation_patience_epochs':2,
      'trainer_state':trainer_states,
      'model_load_seconds':load_seconds,'fine_tune_seconds':fit_seconds,
      'inference_seconds':{'zero_shot':zero['elapsed_seconds'],'lora':adapted['elapsed_seconds']},
      'gpu':gpu,'peak_pytorch_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
      'hashes':{'sales_panel':sha(panel_path)}}
    (METADATA_OUT/'chronos2_lora_full_series_epoch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)

def forecast(model,contexts,monitor):
    start=perf_counter();out=[]
    for offset in range(0,len(contexts),16):
        monitor.check()
        q,_=model.predict_quantiles(contexts[offset:offset+16],prediction_length=HORIZON,
          quantile_levels=[.5],batch_size=16,context_length=CONTEXT)
        for value in q:
            arr=value.detach().float().cpu().numpy()
            # Result is (1, horizon, 1) for one univariate input.
            out.append(np.maximum(0.,arr.reshape(-1,HORIZON,1)[0,:,0]).tolist())
    return {'values':out,'elapsed_seconds':perf_counter()-start}

def write_predictions(path,codes,result):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','stock_code','target_date','horizon','prediction'])
        w.writeheader()
        for code,pred in zip(codes,result['values']):
            for h,value in enumerate(pred,1):
                w.writerow({'model':path.stem,'stock_code':code,
                  'target_date':(TEST_ORIGIN+timedelta(days=h)).isoformat(),
                  'horizon':h,'prediction':value})

if __name__=='__main__':main()
