"""Zero-shot Timestral-209M forecast on the frozen October test cohort."""
from time import perf_counter
import json, os
import numpy as np
from timestral import TimestralForecaster
from common import ROOT, RUN_VALIDATION, ORIGIN, HORIZON, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata

def main():
    os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
    os.environ['HF_HUB_DISABLE_XET']='1'
    _,codes,contexts=load_contexts()
    monitor=start_gpu_monitor()
    started=perf_counter();model=TimestralForecaster.from_pretrained('mistralai/Timestral-209M',device='cuda')
    load_seconds=perf_counter()-started
    all_predictions=[];infer_started=perf_counter()
    for offset in range(0,len(contexts),16):
        monitor.check()
        batch=np.asarray(contexts[offset:offset+16],dtype=np.float32)[:,None,:]
        result=model.forecast(batch,prediction_length=HORIZON)
        median=result.median
        if hasattr(median,'detach'):
            median=median.detach().float().cpu().numpy()
        values=np.asarray(median,dtype=np.float32)
        all_predictions.extend(values[:,0,:])
        if (offset+16)%512<16:print(f'Forecasted {min(offset+16,len(contexts))}/{len(contexts)} products',flush=True)
    infer_seconds=perf_counter()-infer_started
    path=write_forecasts('timestral209m_zero_shot',codes,all_predictions,infer_seconds)
    gpu=monitor.finish()
    import torch
    save_run_metadata('timestral209m_zero_shot',model_id='mistralai/Timestral-209M',
      method='zero-shot',test_targets_loaded=False,products=len(codes),
      forecast_origin=ORIGIN.isoformat(),selection_targets_loaded=False,
      test_period=None if RUN_VALIDATION else ['2011-10-05','2011-11-01'],
      selection_period=['2011-07-01','2011-07-28'] if RUN_VALIDATION else None,
      context_days=90,horizon_days=HORIZON,
      device='cuda',batch_size=16,load_seconds=load_seconds,inference_seconds=infer_seconds,
      gpu=gpu,peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
      weight_license='Modified MIT; see model card')
    print(json.dumps({'forecast_file':str(path),'load_seconds':load_seconds,
                      'inference_seconds':infer_seconds,'gpu':gpu},indent=2),flush=True)

if __name__=='__main__':main()
