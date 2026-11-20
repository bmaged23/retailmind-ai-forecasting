"""Zero-shot Toto 2.0 22M forecast on the frozen October test cohort."""
from time import perf_counter
import json, os
import numpy as np
import torch
from toto2 import Toto2Model
from common import ROOT, RUN_VALIDATION, ORIGIN, HORIZON, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata

def main():
    os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
    os.environ['HF_HUB_DISABLE_XET']='1'
    _,codes,contexts=load_contexts();monitor=start_gpu_monitor()
    start=perf_counter();model=Toto2Model.from_pretrained('Datadog/Toto-2.0-22m').to('cuda').eval()
    load_seconds=perf_counter()-start
    all_predictions=[];infer_start=perf_counter();batch_size=32
    with torch.inference_mode():
        for offset in range(0,len(contexts),batch_size):
            monitor.check();values=np.asarray(contexts[offset:offset+batch_size],dtype=np.float32)
            # Toto 2.0 patchifies context in 32-step patches. Pad the 90-day
            # frozen context to 96 and mask those leading six positions.
            target=torch.nn.functional.pad(
                torch.as_tensor(values,device='cuda').unsqueeze(1),(6,0))
            target_mask=torch.ones_like(target,dtype=torch.bool)
            target_mask[...,:6]=False
            series_ids=torch.zeros((len(values),1),dtype=torch.long,device='cuda')
            quantiles=model.forecast({'target':target,'target_mask':target_mask,'series_ids':series_ids},
              horizon=HORIZON,decode_block_size=None,has_missing_values=True)
            all_predictions.extend(quantiles[4,:,0,:].float().cpu().numpy())
            if (offset+batch_size)%512<batch_size:print(f'Forecasted {min(offset+batch_size,len(contexts))}/{len(contexts)} products',flush=True)
    infer_seconds=perf_counter()-infer_start
    path=write_forecasts('toto2_22m_zero_shot',codes,all_predictions,infer_seconds)
    gpu=monitor.finish()
    save_run_metadata('toto2_22m_zero_shot',model_id='Datadog/Toto-2.0-22m',
      method='zero-shot',test_targets_loaded=False,products=len(codes),
      forecast_origin=ORIGIN.isoformat(),selection_targets_loaded=False,
      test_period=None if RUN_VALIDATION else ['2011-10-05','2011-11-01'],
      selection_period=['2011-07-01','2011-07-28'] if RUN_VALIDATION else None,
      context_days=90,horizon_days=HORIZON,
      device='cuda',batch_size=batch_size,load_seconds=load_seconds,inference_seconds=infer_seconds,
      gpu=gpu,peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
      fine_tuning='Not available in Toto 2.0; official repository directs fine-tuning users to Toto 1.0')
    print(json.dumps({'forecast_file':str(path),'load_seconds':load_seconds,
                      'inference_seconds':infer_seconds,'gpu':gpu},indent=2),flush=True)

if __name__=='__main__':main()
