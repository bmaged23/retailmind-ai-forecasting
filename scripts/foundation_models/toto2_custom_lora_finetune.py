"""Custom experimental LoRA adaptation for Toto 2.0 on retail sales.

Datadog does not currently publish a Toto 2.0 fine-tuning workflow. This
runner uses the differentiable model forward path with a masked quantile loss;
it is not an official Toto fine-tuning recipe.
"""
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
import json
import os

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from toto2 import Toto2Model

from common import ROOT, RUN_DIR, RUN_VALIDATION, ORIGIN, CONTEXT, HORIZON, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata
from retailmind.forecast import load_panel

TRAIN_END=date(2011,4,30) if RUN_VALIDATION else date(2011,8,31)
VALID_END=date(2011,5,28) if RUN_VALIDATION else date(2011,9,28)
PATCH=32
PADDED_CONTEXT=((CONTEXT+PATCH-1)//PATCH)*PATCH
BATCH_SIZE=4
MAX_EPOCHS=8
PATIENCE=2
LR=5e-5
MODEL_ID='Datadog/Toto-2.0-22m'
CACHE=Path(os.getenv('HF_HOME',str(ROOT/'.cache/huggingface')))
OUT=RUN_DIR
ADAPTER=OUT/'models/toto2_22m/custom_lora'
ZERO_FORECAST=(OUT/'forecasts/toto2_22m_zero_shot.csv' if RUN_VALIDATION else OUT/'forecasts/toto2_22m_zero_shot/toto2_22m_zero_shot.csv')


def dense(series,end):
    values={d:q for d,q in series.items() if d<=end}
    if not values:return None
    first=min(values)
    return np.asarray([values.get(first+timedelta(days=i),0.)
                       for i in range((end-first).days+1)],dtype=np.float32)


def batch_loss(model,contexts,targets,device,levels,target_mask_future=None):
    padded=[padded_inference_context(x) for x in contexts]
    context=torch.as_tensor(np.stack(padded),device=device,dtype=torch.float32).unsqueeze(1)
    observed=torch.ones_like(context,dtype=torch.bool)
    observed[...,:PADDED_CONTEXT-len(contexts[0])]=False
    # Retain explicit mask support for later padded/short training contexts.
    if target_mask_future is None:
        future_mask=torch.ones((len(targets),1,HORIZON),device=device,dtype=torch.bool)
    else:
        future_mask=target_mask_future
    target_values=np.stack(targets)
    target=torch.as_tensor(target_values,device=device,dtype=torch.float32)
    target=torch.nn.functional.pad(target,(0,PATCH-HORIZON)).unsqueeze(1)
    mask_future=torch.nn.functional.pad(future_mask,(0,PATCH-HORIZON),value=False)

    out=model(target=context,target_mask=observed,cpm_mask=observed,
              series_ids=torch.zeros((len(contexts),1),device=device,dtype=torch.long),
              num_return_steps=1)
    pred=out.quantiles[:,:,:, -1, :]
    loc=out.loc[..., -1].unsqueeze(-1)
    scale=out.scale[..., -1].unsqueeze(-1).clamp_min(1e-6)
    normalized_target=torch.asinh((target-loc)/scale)
    error=normalized_target.unsqueeze(0)-pred
    levels=levels.view(-1,1,1,1)
    pinball=torch.maximum(levels*error,(levels-1)*error)
    valid=mask_future.unsqueeze(0)
    return (pinball*valid).sum()/(valid.sum().clamp_min(1)*len(levels))


def evaluate(model,rows,device,levels):
    losses=[];model.eval()
    with torch.no_grad():
        for offset in range(0,len(rows),BATCH_SIZE):
            batch=rows[offset:offset+BATCH_SIZE]
            contexts=[r[-HORIZON-CONTEXT:-HORIZON] for r in batch]
            targets=[r[-HORIZON:] for r in batch]
            losses.append(float(batch_loss(model,contexts,targets,device,levels).cpu()))
    return float(np.mean(losses))


def padded_inference_context(values):
    values=np.asarray(values,dtype=np.float32)
    if len(values)>PADDED_CONTEXT:values=values[-PADDED_CONTEXT:]
    return np.pad(values,(PADDED_CONTEXT-len(values),0),constant_values=0.)


def main():
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
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
    print('Loading Toto 2.0 22M on CUDA...',flush=True)
    load_start=perf_counter()
    base=Toto2Model.from_pretrained(MODEL_ID,cache_dir=CACHE/'hub',local_files_only=True)
    base.to('cuda').eval()
    load_seconds=perf_counter()-load_start
    print(f'Loaded {MODEL_ID}; train_series={len(train_rows)}; validation_sample={len(valid_rows)}',flush=True)

    # Reuse the scored zero-shot forecasts produced before any adaptation.
    if ZERO_FORECAST.exists():
        print(f'Reusing saved zero-shot forecasts: {ZERO_FORECAST}',flush=True)
        zero_seconds=0.0
    else:
        infer_start=perf_counter();zero=[]
        with torch.inference_mode():
            for offset in range(0,len(test_contexts),32):
                monitor.check();batch=test_contexts[offset:offset+32]
                values=np.stack([padded_inference_context(x) for x in batch])
                target=torch.as_tensor(values,device='cuda').unsqueeze(1)
                target_mask=torch.ones_like(target,dtype=torch.bool);target_mask[...,:CONTEXT-len(batch[0])]=False
                quantiles=base.forecast({'target':target,'target_mask':target_mask,
                    'series_ids':torch.zeros((len(batch),1),dtype=torch.long,device='cuda')},
                    horizon=HORIZON,decode_block_size=None,has_missing_values=True)
                zero.extend(np.maximum(0.,quantiles[4,:,0,:].float().cpu().numpy()))
        zero_seconds=perf_counter()-infer_start
        write_forecasts('toto2_22m_zero_shot',codes,zero,zero_seconds)

    wrapped=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.05,
                         target_modules='all-linear',bias='none'))
    trainable=[p for p in wrapped.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=LR,weight_decay=0.01)
    levels=torch.as_tensor(base.output_head.knots,device='cuda',dtype=torch.float32)
    steps_per_epoch=(len(train_rows)+BATCH_SIZE-1)//BATCH_SIZE
    best=float('inf');best_state=None;stale=0;history=[];started=perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        wrapped.train();order=rng.permutation(len(train_rows));losses=[]
        for offset in range(0,len(order),BATCH_SIZE):
            monitor.check();ids=order[offset:offset+BATCH_SIZE]
            contexts=[];targets=[]
            for i in ids:
                row=train_rows[int(i)];max_start=len(row)-CONTEXT-HORIZON
                start=int(rng.integers(max_start+1)) if max_start>0 else 0
                contexts.append(row[start:start+CONTEXT])
                targets.append(row[start+CONTEXT:start+CONTEXT+HORIZON])
            optimizer.zero_grad(set_to_none=True)
            loss=batch_loss(wrapped,contexts,targets,'cuda',levels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0)
            optimizer.step();losses.append(float(loss.detach().cpu()))
            if (offset//BATCH_SIZE+1)%250==0:monitor.check()
            if (offset//BATCH_SIZE+1)%500==0:
                print(f'epoch={epoch} batches={offset//BATCH_SIZE+1}/{steps_per_epoch}',flush=True)
        val=evaluate(wrapped,valid_rows,'cuda',levels);train_loss=float(np.mean(losses))
        history.append({'epoch':epoch,'train_pinball_loss':train_loss,'validation_pinball_loss':val})
        print(f'epoch={epoch} train_pinball_loss={train_loss:.6f} validation_pinball_loss={val:.6f}',flush=True)
        if val<best:
            best=val;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(wrapped).items()}
            ADAPTER.mkdir(parents=True,exist_ok=True);wrapped.save_pretrained(ADAPTER)
            (OUT/'metadata/toto2_custom_lora_best.json').write_text(json.dumps({
                'epoch':epoch,'train_pinball_loss':train_loss,'validation_pinball_loss':val,
                'checkpoint_path':str(ADAPTER),'method':'custom experimental LoRA'},indent=2)+'\n')
        else:stale+=1
        if stale>=PATIENCE:
            print(f'Early stopping after {epoch} epochs; best validation loss={best:.6f}',flush=True)
            break
    set_peft_model_state_dict(wrapped,best_state)
    adapter_model=wrapped.get_base_model().eval()
    del optimizer,trainable
    torch.cuda.empty_cache();monitor.check()

    predictions=[];infer_start=perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(test_contexts),32):
            monitor.check();batch=test_contexts[offset:offset+32]
            values=np.stack([padded_inference_context(x) for x in batch])
            target=torch.as_tensor(values,device='cuda').unsqueeze(1)
            target_mask=torch.ones_like(target,dtype=torch.bool);target_mask[...,:CONTEXT-len(batch[0])]=False
            quantiles=adapter_model.forecast({'target':target,'target_mask':target_mask,
                'series_ids':torch.zeros((len(batch),1),dtype=torch.long,device='cuda')},
                horizon=HORIZON,decode_block_size=None,has_missing_values=True)
            predictions.extend(np.maximum(0.,quantiles[4,:,0,:].float().cpu().numpy()))
            if (offset+32)%512<32:
                print(f'Fine-tuned forecasts {min(offset+32,len(codes))}/{len(codes)}',flush=True)
    inference=perf_counter()-infer_start
    write_forecasts('toto2_22m_custom_lora',codes,predictions,inference)
    gpu=monitor.finish()
    metadata={'model':MODEL_ID,'method':'custom experimental LoRA (not an official Toto 2.0 recipe)',
      'train_end':TRAIN_END.isoformat(),'validation_end':VALID_END.isoformat(),
      'forecast_origin':ORIGIN.isoformat(),
      'forecast_targets_loaded':False,'products':len(codes),'train_series':len(train_rows),
      'validation_sample_size':len(valid_rows),'validation_seed':42,
      'context_days':CONTEXT,'forecast_horizon_days':HORIZON,'patch_size':PATCH,
      'batch_size':BATCH_SIZE,'lora_rank':4,'lora_alpha':8,'learning_rate':LR,
      'weight_decay':0.01,'max_epochs':MAX_EPOCHS,'early_stopping_patience':PATIENCE,
      'best_validation_pinball_loss':best,'epochs':history,
      'training_seconds':perf_counter()-started,'load_seconds':load_seconds,
      'zero_shot_inference_seconds':zero_seconds,'fine_tuned_inference_seconds':inference,
      'gpu':gpu,'fine_tuning_support_note':'Custom research adaptation because official Toto 2.0 fine-tuning is not yet available.'}
    save_run_metadata('toto2_22m_custom_lora', **metadata)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
