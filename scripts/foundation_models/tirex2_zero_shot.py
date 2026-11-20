"""Zero-shot TiRex-2 forecast on the frozen October test cohort."""
from time import perf_counter
import json, os
import numpy as np
import torch
from tirex2 import TimeseriesType, load_model
from common import ROOT, RUN_VALIDATION, ORIGIN, HORIZON, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata

def main():
    os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
    os.environ['HF_HUB_DISABLE_XET']='1'
    _,codes,contexts=load_contexts();monitor=start_gpu_monitor()
    # CUDA graphs trigger Triton JIT compilation; use eager inference so this
    # runner works in minimal WSL installs without a system C compiler.
    started=perf_counter();model=load_model('NX-AI/TiRex-2',device='cuda',cuda_graphs=False)
    load_seconds=perf_counter()-started;predictions=[];infer_started=perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(contexts),32):
            monitor.check();batch=[]
            for values in contexts[offset:offset+32]:
                target=torch.as_tensor(values,dtype=torch.float32,device='cuda').unsqueeze(0)
                batch.append(TimeseriesType(target=target,past_covariates=None,future_covariates=None))
            forecasts=model.forecast(batch,prediction_length=HORIZON,output_type='numpy')
            predictions.extend([np.asarray(f,dtype=np.float32)[0,4,:] for f in forecasts])
            if (offset+32)%512<32:print(f'Forecasted {min(offset+32,len(contexts))}/{len(contexts)} products',flush=True)
    infer_seconds=perf_counter()-infer_started
    path=write_forecasts('tirex2_zero_shot',codes,predictions,infer_seconds)
    gpu=monitor.finish()
    save_run_metadata('tirex2_zero_shot',model_id='NX-AI/TiRex-2',method='zero-shot',
      test_targets_loaded=False,selection_targets_loaded=False,products=len(codes),
      forecast_origin=ORIGIN.isoformat(),test_period=None if RUN_VALIDATION else ['2011-10-05','2011-11-01'],
      selection_period=['2011-07-01','2011-07-28'] if RUN_VALIDATION else None,
      context_days=90,horizon_days=HORIZON,device='cuda',batch_size=32,
      load_seconds=load_seconds,inference_seconds=infer_seconds,gpu=gpu,
      peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
      fine_tuning='Open release is zero-shot; fine-tuning is a TiRex-2 PRO feature')
    print(json.dumps({'forecast_file':str(path),'load_seconds':load_seconds,
                      'inference_seconds':infer_seconds,'gpu':gpu},indent=2),flush=True)

if __name__=='__main__':main()
