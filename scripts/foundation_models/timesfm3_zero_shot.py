"""Zero-shot TimesFM 3.0 forecast on the frozen October test cohort."""
from time import perf_counter
import json, os
from pathlib import Path
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig
from common import ROOT, TEST_DIR, RUN_VALIDATION, ORIGIN, HORIZON, load_contexts, start_gpu_monitor, write_forecasts, save_run_metadata

def main():
    os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
    os.environ['HF_HUB_DISABLE_XET']='1'
    _,codes,contexts=load_contexts()
    monitor=start_gpu_monitor();device='cuda'
    print('Loading TimesFM 3.0 on CUDA with batch size 1...',flush=True)
    load_start=perf_counter()
    model=TimesFM3Evaluator(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
                                        per_core_batch_size=1,device=device))
    load_seconds=perf_counter()-load_start
    start=perf_counter();predictions=[]
    for i,context in enumerate(contexts,1):
        monitor.check()
        output=list(model.predict_batch([context],horizon=HORIZON,return_quantiles=True,
                                        use_symmetric_averaging=False))[0]
        predictions.append(np.asarray(output.forecast,dtype=np.float32))
        if i%500==0:print(f'Forecasted {i}/{len(contexts)} products',flush=True)
    infer_seconds=perf_counter()-start
    path=write_forecasts('timesfm3_zero_shot',codes,predictions,infer_seconds)
    gpu=monitor.finish()
    save_run_metadata('timesfm3_zero_shot',model_id='google/timesfm-3.0-pytorch',
      method='zero-shot',test_targets_loaded=False,products=len(codes),
      forecast_origin=ORIGIN.isoformat(),selection_targets_loaded=False,
      test_period=None if RUN_VALIDATION else ['2011-10-05','2011-11-01'],
      selection_period=['2011-07-01','2011-07-28'] if RUN_VALIDATION else None,
      context_days=90,horizon_days=HORIZON,
      device=device,load_seconds=load_seconds,inference_seconds=infer_seconds,
      gpu=gpu,peak_pytorch_reserved_mib=__import__('torch').cuda.max_memory_reserved()/1024**2,
      weight_license='TimesFM non-commercial license v1.0; research-only use')
    print(json.dumps({'forecast_file':str(path),'load_seconds':load_seconds,
                      'inference_seconds':infer_seconds,'gpu':gpu},indent=2),flush=True)

if __name__=='__main__':main()
