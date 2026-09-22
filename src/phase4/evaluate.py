"""Freeze validation-selected models, then evaluate without test-driven tuning."""
import argparse
import json
import time
import numpy as np
from . import ROOT
from .data import BASE,METRICS,MODEL_NAMES,diagnostic_labels,guard,settings
from .metrics import binary_metrics,family_diagnostics
from .train import MemoryMonitor,load_model,model_path,predict_partition
from src.baseline.common import json_write,sha256,require

LOCK=METRICS/'development_configs_frozen.json'

def freeze():
    guard()
    models={}
    config_hash=sha256(ROOT/'configs/phase4_seed42.json')
    for name in MODEL_NAMES:
        training=json.loads((METRICS/f'{name}_seed42_training.json').read_text())
        require(training['status']=='complete' and training['config_sha256']==config_hash,'Incomplete or changed training configuration')
        require(sha256(model_path(name))==training['model_sha256'],'Model checksum mismatch')
        models[name]={'sha256':training['model_sha256'],'selected_step':training['selected_step']}
    state=dict(seed=42,phase='development',config_sha256=config_hash,models=models,threshold=settings()['threshold'],selection_partition='validation')
    if LOCK.exists(): require(json.loads(LOCK.read_text())==state,'Frozen model selection changed')
    else: json_write(LOCK,state)
    return state

def run(name):
    state=freeze()
    output=METRICS/f'{name}_seed42_evaluation.json'
    if output.exists():
        saved=json.loads(output.read_text())
        require(saved['model_sha256']==state['models'][name]['sha256'],'Evaluation model changed')
        print('Evaluation already complete:',name,flush=True);return
    check=json.loads((METRICS/'pretraining_check.json').read_text())
    predictions=METRICS/'predictions';predictions.mkdir(exist_ok=True)
    result=dict(model=name,seed=42,model_sha256=state['models'][name]['sha256'],partitions={})
    with MemoryMonitor() as monitor:
        model=load_model(name)
        for part in ('validation','test'):
            y,codes,book=diagnostic_labels(part)
            score,timing=predict_partition(model,name,part)
            path=predictions/f'{name}_seed42_{part}.npy';np.save(path,score)
            result['partitions'][part]=dict(metrics=binary_metrics(y,score,state['threshold']),runtime=timing,
                families=family_diagnostics(codes,book,score,check['attack_family_counts']['train'],state['threshold']),
                probability_path=str(path),probability_sha256=sha256(path))
            print('EVALUATED',name,part,result['partitions'][part]['metrics'],flush=True)
    result['evaluation_peak_memory_bytes']=monitor.peak
    guard();json_write(output,result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODEL_NAMES);args=parser.parse_args()
    run(args.model) if args.model else freeze()
