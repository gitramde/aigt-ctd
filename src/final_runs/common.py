import json,csv,time,random
from pathlib import Path
from . import ROOT,OUT
from .preflight import read,write,sha
SPEC=read(ROOT/'results/final_spec_v1/final_experiment_spec.json')
SEEDS=SPEC['seeds']
GROUPS={'supervised':['logistic_regression','random_forest','xgboost','mlp'],
 'temporal':['transformer_L64','transformer_L1'],'graph':['edge_mlp','gatv2','gatv2_self_only'],
 'graph_temporal':['gat_transformer_L1','gat_transformer_L8'],'anomaly':['autoencoder_B']}
def folder(seed,group):return OUT/f'seed_{seed}'/group
def setup(seed,group):
    p=folder(seed,group)
    for sub in ('configs','models','metrics'): (p/sub).mkdir(parents=True,exist_ok=True)
    return p
def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def require(ok,message):
    if not ok:raise RuntimeError(message)
def seed_all(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def check_lock():
    require(not (OUT/'audit/STOP_FAILURE.json').exists(),'Preserved failure forbids automatic continuation')
    lock=read(OUT/'audit/runner_lock.json')
    require(sha(ROOT/'results/final_spec_v1/final_experiment_spec.json')==lock['spec_sha256'],'Frozen spec changed')
    for row in lock['runner_sources']+lock['effective_configurations']:
        require(sha(ROOT/row['path'])==row['sha256'],'Runner/config changed: '+row['path'])
    require(read(OUT/'audit/preflight_checks.json')['status']=='PASS','Preflight incomplete')
def mark(stage,seed,model,**details):
    with (OUT/'audit/events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stage=stage,seed=seed,model=model,**details))+'\n')
