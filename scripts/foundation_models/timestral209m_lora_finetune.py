"""Custom LoRA adaptation for Timestral-209M with chronological early stopping.

Timestral's upstream package exposes an inference API, not a trainer. This
experiment uses its differentiable PyTorch forecast block with a quantile loss,
and PEFT LoRA adapters. Test targets are never used here.
"""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import csv, json

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from timestral import TimestralForecaster

from common import ROOT, RUN_DIR, RUN_VALIDATION, ORIGIN, HORIZON, CONTEXT, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata
from retailmind.forecast import load_panel

TRAIN_END=date(2011,4,30) if RUN_VALIDATION else date(2011,8,31)
VALID_END=date(2011,5,28) if RUN_VALIDATION else date(2011,9,28)
BATCH_SIZE=4
MAX_EPOCHS=8
PATIENCE=2
LR=1e-4
MODEL_DIR=RUN_DIR/'models/timestral209m/lora'

def dense(series,end):
    vals={d:q for d,q in series.items() if d<=end}
    if not vals:return None
    first=min(vals)
    return np.asarray([vals.get(first+timedelta(days=i),0.)
                       for i in range((end-first).days+1)],dtype=np.float32)

def batch_loss(model,contexts,targets,patch_size,device,train=False,optimizer=None):
    x=torch.as_tensor(np.stack(contexts),device=device,dtype=torch.float32)
    y=torch.as_tensor(np.stack(targets),device=device,dtype=torch.float32)
    actual_length=x.shape[-1]
    padded_length=((actual_length+patch_size-1)//patch_size)*patch_size
    if padded_length>actual_length:
        x=torch.nn.functional.pad(x,(padded_length-actual_length,0))
    obs=torch.isfinite(x)
    if padded_length>actual_length:obs[...,:padded_length-actual_length]=False
    x=torch.nan_to_num(x,nan=0.0)
    groups=torch.arange(len(contexts),device=device,dtype=torch.long)
    steps=(HORIZON+patch_size-1)//patch_size
    if train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred=model.predict_block(x,obs,groups,steps)[:,:HORIZON]
    else:
        model.eval()
        with torch.inference_mode():
            pred=model.predict_block(x,obs,groups,steps)[:,:HORIZON]
    loc_seq,scale_seq=model.scale(x,obs)[1:]
    loc=loc_seq[:,-1].view(-1,1)
    scale=scale_seq[:,-1].clamp_min(1e-6).view(-1,1)
    y_norm=(y-loc)/scale
    p_norm=(pred-loc.unsqueeze(-1))/scale.unsqueeze(-1)
    errors=y_norm.unsqueeze(-1)-p_norm
    levels=torch.as_tensor(model.config.quantile_knots,device=device,dtype=errors.dtype)
    qloss=torch.maximum(levels*(errors), (levels-1)*errors).mean()
    if train:
        qloss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        optimizer.step()
    return float(qloss.detach().cpu())

def evaluate(model,rows,context_len,patch_size,device):
    losses=[]
    for start in range(0,len(rows),BATCH_SIZE):
        group=rows[start:start+BATCH_SIZE]
        contexts=[r[-HORIZON-context_len:-HORIZON] for r in group]
        targets=[r[-HORIZON:] for r in group]
        losses.append(batch_loss(model,contexts,targets,patch_size,device))
    return float(np.mean(losses))

def main():
    import os
    os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
    os.environ['HF_HUB_DISABLE_XET']='1'
    panel,codes,test_contexts=load_contexts()
    monitor=start_gpu_monitor()
    sample=TimestralForecaster.from_pretrained('mistralai/Timestral-209M',device='cuda')
    base=sample.model
    patch=base.config.patch_size
    # Keep at least the shared 90-day model context, rounded up to a complete
    # Timestral patch so its training and inference windows are comparable.
    context_len=CONTEXT
    train_rows=[];valid_rows=[]
    for series in panel.values():
        tr=dense(series,TRAIN_END)
        va=dense(series,VALID_END)
        if tr is not None and len(tr)>=context_len+HORIZON:train_rows.append(tr)
        if va is not None and len(va)>=context_len+HORIZON:valid_rows.append(va)
    rng=np.random.default_rng(42)
    if len(valid_rows)>256:
        valid_rows=[valid_rows[i] for i in np.sort(rng.choice(len(valid_rows),256,replace=False))]
    del sample

    device=torch.device('cuda')
    lora=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.05,
                    target_modules='all-linear',bias='none')
    wrapped=get_peft_model(base,lora)
    model=wrapped.get_base_model()
    params=[p for p in wrapped.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=LR,weight_decay=0.01)
    print(f'Timestral LoRA: train_series={len(train_rows)}, validation_sample={len(valid_rows)}, '
          f'context={context_len}, batch={BATCH_SIZE}, steps/epoch={(len(train_rows)+BATCH_SIZE-1)//BATCH_SIZE}',flush=True)

    best=float('inf');best_state=None;stale=0;history=[];started=perf_counter()
    steps_per_epoch=(len(train_rows)+BATCH_SIZE-1)//BATCH_SIZE
    for epoch in range(1,MAX_EPOCHS+1):
        order=rng.permutation(len(train_rows));train_losses=[]
        for offset in range(0,len(order),BATCH_SIZE):
            ids=order[offset:offset+BATCH_SIZE]
            contexts=[];targets=[]
            for i in ids:
                row=train_rows[int(i)]
                max_start=len(row)-context_len-HORIZON
                start=int(rng.integers(max_start+1)) if max_start>0 else 0
                contexts.append(row[start:start+context_len])
                targets.append(row[start+context_len:start+context_len+HORIZON])
            train_losses.append(batch_loss(model,contexts,targets,patch,device,True,optimizer))
            if (offset//BATCH_SIZE+1)%250==0:
                monitor.check()
        val_loss=evaluate(model,valid_rows,context_len,patch,device)
        history.append({'epoch':epoch,'train_loss':float(np.mean(train_losses)),'validation_loss':val_loss})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} validation_loss={val_loss:.6f}",flush=True)
        if val_loss<best:
            best=val_loss;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(wrapped).items()}
        else:
            stale+=1
        if stale>=PATIENCE:
            print(f'Early stopping after {epoch} epochs; best validation loss={best:.6f}',flush=True)
            break
    set_peft_model_state_dict(wrapped,best_state)
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    wrapped.save_pretrained(MODEL_DIR)

    del optimizer,params
    torch.cuda.empty_cache();monitor.check()
    tuned=TimestralForecaster(model,max_context=90)
    monitor.check();start=perf_counter();predictions=[]
    for offset in range(0,len(test_contexts),16):
        monitor.check()
        out=tuned.forecast(np.asarray(test_contexts[offset:offset+16])[:,None,:],
                           prediction_length=HORIZON)
        med=out.median.detach().float().cpu().numpy()
        predictions.extend(np.maximum(0,med[:,0,:]))
    inference=perf_counter()-start
    path=write_forecasts('timestral209m_lora',codes,predictions,inference)
    gpu=monitor.finish()
    metadata={'model':'mistralai/Timestral-209M','method':'custom LoRA fine-tuning',
      'base_model_api':'Timestral PyTorch predict_block with quantile loss; upstream package has no trainer',
      'train_end':TRAIN_END.isoformat(),'validation_end':VALID_END.isoformat(),
      'forecast_origin':ORIGIN.isoformat(),
      'forecast_targets_loaded':False,'products':len(codes),'train_series':len(train_rows),
      'validation_sample_size':len(valid_rows),'validation_seed':42,'context_days':context_len,
      'model_padded_context_days':((context_len+patch-1)//patch)*patch,
      'horizon_days':HORIZON,'batch_size':BATCH_SIZE,'lora_rank':4,'lora_alpha':8,
      'learning_rate':LR,'weight_decay':0.01,'max_epochs':MAX_EPOCHS,
      'early_stopping_patience':PATIENCE,'best_validation_loss':best,'epochs':history,
      'fine_tune_seconds':perf_counter()-started,'inference_seconds':inference,
      'gpu':gpu,'forecast_file':str(path)}
    save_run_metadata('timestral209m_lora', **metadata)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
