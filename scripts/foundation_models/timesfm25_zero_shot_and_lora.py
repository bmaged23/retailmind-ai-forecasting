"""TimesFM 2.5 zero-shot and LoRA forecasts on the frozen test cohort.

Training uses one random 96-day/28-day window per eligible product each epoch,
validation-only early stopping, and never loads test targets.
"""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import json
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TimesFm2_5ModelForPrediction

from common import ROOT, RUN_DIR, RUN_VALIDATION, ORIGIN, CONTEXT, HORIZON, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata
from retailmind.forecast import load_panel

TRAIN_END=date(2011,4,30) if RUN_VALIDATION else date(2011,8,31)
VALID_END=date(2011,5,28) if RUN_VALIDATION else date(2011,9,28)
BATCH_SIZE=4
MAX_EPOCHS=8
PATIENCE=2
LR=5e-5
MODEL_CONTEXT=96  # prepend zeros to the shared 90-day context to align 32-day patches
MODEL_ID='google/timesfm-2.5-200m-transformers'
PORTABLE_CACHE=ROOT/'.cache/huggingface'
RETAINED_CACHE=Path('/mnt/d/retailmind-model-cache/huggingface')
DEFAULT_CACHE=RETAINED_CACHE if RETAINED_CACHE.exists() else PORTABLE_CACHE
MODEL_CACHE=Path(os.getenv('RETAILMIND_HF_CACHE',str(DEFAULT_CACHE)))
OUT=RUN_DIR
ADAPTER=OUT/'models/timesfm25/lora'
ZERO_FORECAST=(OUT/'forecasts/timesfm25_zero_shot.csv' if RUN_VALIDATION else OUT/'forecasts/timesfm25_zero_shot/timesfm25_zero_shot.csv')

def dense(series,end):
    vals={d:q for d,q in series.items() if d<=end}
    if not vals:return None
    first=min(vals)
    return np.asarray([vals.get(first+timedelta(days=i),0.)
                       for i in range((end-first).days+1)],dtype=np.float32)

def forward(model,contexts,targets=None,device='cuda'):
    # Preserve the shared origin while aligning contexts to 32-day patches.
    past=[]
    for x in contexts:
        values=torch.as_tensor(x,device=device,dtype=torch.float32)
        if values.numel()>MODEL_CONTEXT:values=values[-MODEL_CONTEXT:]
        if values.numel()<MODEL_CONTEXT:
            values=torch.nn.functional.pad(values,(MODEL_CONTEXT-values.numel(),0))
        past.append(values)
    kwargs={'past_values':past,'forecast_context_len':MODEL_CONTEXT,
            'force_flip_invariance':False,'truncate_negative':False}
    if targets is not None:
        kwargs['future_values']=torch.as_tensor(np.stack(targets),device=device,dtype=torch.float32)
    return model(**kwargs)

def evaluate(model,rows,device):
    losses=[]
    model.eval()
    with torch.inference_mode():
        for offset in range(0,len(rows),BATCH_SIZE):
            batch=rows[offset:offset+BATCH_SIZE]
            contexts=[r[-HORIZON-CONTEXT:-HORIZON] for r in batch]
            targets=[r[-HORIZON:] for r in batch]
            losses.append(float(forward(model,contexts,targets,device).loss.detach().cpu()))
    return float(np.mean(losses))

def main():
    os.environ.setdefault('HF_HOME',str(MODEL_CACHE))
    os.environ['HF_HUB_DISABLE_XET']='1'
    panel,codes,test_contexts=load_contexts()
    train_rows=[];valid_rows=[]
    for series in panel.values():
        tr=dense(series,TRAIN_END);va=dense(series,VALID_END)
        if tr is not None and len(tr)>=CONTEXT+HORIZON:train_rows.append(tr)
        if va is not None and len(va)>=CONTEXT+HORIZON:valid_rows.append(va)
    rng=np.random.default_rng(42)
    if len(valid_rows)>256:
        valid_rows=[valid_rows[i] for i in np.sort(rng.choice(len(valid_rows),256,replace=False))]

    monitor=start_gpu_monitor()
    print('Loading TimesFM 2.5 on CUDA...',flush=True)
    load_start=perf_counter()
    base=TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,cache_dir=MODEL_CACHE,torch_dtype=torch.float32)
    base.to('cuda');base.eval()
    load_seconds=perf_counter()-load_start
    print(f'Loaded {MODEL_ID}; train_series={len(train_rows)}; validation_sample={len(valid_rows)}',flush=True)

    # Reuse the saved zero-shot cohort on recovery runs; it was produced before
    # any adaptation and has already been frozen to disk.
    if ZERO_FORECAST.exists():
        print(f'Reusing saved zero-shot forecasts: {ZERO_FORECAST}',flush=True)
        zero_seconds=0.0
    else:
        infer_start=perf_counter();zero=[]
        with torch.inference_mode():
            for offset in range(0,len(test_contexts),BATCH_SIZE):
                monitor.check()
                batch=test_contexts[offset:offset+BATCH_SIZE]
                out=forward(base,batch,device='cuda')
                zero.extend(np.maximum(0.,out.mean_predictions[:,:HORIZON].float().cpu().numpy()))
                if (offset+BATCH_SIZE)%512<BATCH_SIZE:
                    print(f'Zero-shot forecasts {min(offset+BATCH_SIZE,len(codes))}/{len(codes)}',flush=True)
        zero_seconds=perf_counter()-infer_start
        write_forecasts('timesfm25_zero_shot',codes,zero,zero_seconds)
    print('Saved TimesFM 2.5 zero-shot forecasts; beginning LoRA training.',flush=True)

    wrapped=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.05,
                                           target_modules='all-linear',bias='none'))
    trainable=[p for p in wrapped.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=LR,weight_decay=0.01)
    steps_per_epoch=(len(train_rows)+BATCH_SIZE-1)//BATCH_SIZE
    best=float('inf');best_state=None;stale=0;history=[];started=perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        wrapped.train();order=rng.permutation(len(train_rows));train_losses=[]
        for offset in range(0,len(order),BATCH_SIZE):
            ids=order[offset:offset+BATCH_SIZE]
            contexts=[];targets=[]
            for i in ids:
                row=train_rows[int(i)];max_start=len(row)-CONTEXT-HORIZON
                start=int(rng.integers(max_start+1)) if max_start>0 else 0
                contexts.append(row[start:start+CONTEXT])
                targets.append(row[start+CONTEXT:start+CONTEXT+HORIZON])
            optimizer.zero_grad(set_to_none=True)
            out=forward(wrapped,contexts,targets,device='cuda')
            loss=out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if (offset//BATCH_SIZE+1)%250==0:monitor.check()
            if (offset//BATCH_SIZE+1)%500==0:
                print(f'epoch={epoch} batches={offset//BATCH_SIZE+1}/{steps_per_epoch}',flush=True)
        val_loss=evaluate(wrapped,valid_rows,'cuda')
        train_loss=float(np.mean(train_losses))
        history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss})
        print(f'epoch={epoch} train_loss={train_loss:.6f} validation_loss={val_loss:.6f}',flush=True)
        if val_loss<best:
            best=val_loss;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(wrapped).items()}
            ADAPTER.mkdir(parents=True,exist_ok=True)
            wrapped.save_pretrained(ADAPTER)
            (OUT/'metadata/timesfm25_best_validation.json').write_text(json.dumps({
                'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss,
                'checkpoint_path':str(ADAPTER)},indent=2)+'\n')
        else:stale+=1
        if stale>=PATIENCE:
            print(f'Early stopping after {epoch} epochs; best validation loss={best:.6f}',flush=True)
            break
    set_peft_model_state_dict(wrapped,best_state)
    ADAPTER.mkdir(parents=True,exist_ok=True);wrapped.save_pretrained(ADAPTER)
    del optimizer,trainable
    torch.cuda.empty_cache();monitor.check()

    infer_start=perf_counter();tuned=[];wrapped.eval()
    with torch.inference_mode():
        for offset in range(0,len(test_contexts),BATCH_SIZE):
            monitor.check()
            batch=test_contexts[offset:offset+BATCH_SIZE]
            out=forward(wrapped,batch,device='cuda')
            tuned.extend(np.maximum(0.,out.mean_predictions[:,:HORIZON].float().cpu().numpy()))
            if (offset+BATCH_SIZE)%512<BATCH_SIZE:
                print(f'Fine-tuned forecasts {min(offset+BATCH_SIZE,len(codes))}/{len(codes)}',flush=True)
    tuned_seconds=perf_counter()-infer_start
    write_forecasts('timesfm25_lora',codes,tuned,tuned_seconds)
    gpu=monitor.finish()
    metadata={'model':MODEL_ID,'method':'LoRA fine-tuning','train_end':TRAIN_END.isoformat(),
      'validation_end':VALID_END.isoformat(),'forecast_origin':ORIGIN.isoformat(),
      'forecast_targets_loaded':False,
      'products':len(codes),'train_series':len(train_rows),
      'validation_sample_size':len(valid_rows),'validation_seed':42,
      'context_days':CONTEXT,'horizon_days':HORIZON,'batch_size':BATCH_SIZE,
      'lora_rank':4,'lora_alpha':8,'learning_rate':LR,'weight_decay':0.01,
      'max_epochs':MAX_EPOCHS,'early_stopping_patience':PATIENCE,
      'best_validation_loss':best,'epochs':history,'training_seconds':perf_counter()-started,
      'zero_shot_load_seconds':load_seconds,'inference_seconds':{
        'zero_shot':zero_seconds,'lora':tuned_seconds},'gpu':gpu}
    save_run_metadata('timesfm25_lora', **metadata)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
