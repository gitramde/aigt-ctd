"""One final fit per process, using frozen algorithms and new output bindings."""
import argparse,time,importlib
from .common import *

def bind(module,p):
    module.OUT=p;module.CONFIG=p/'configs';module.METRICS=p/'metrics';module.MODELS=p/'models'
    module.guard=check_lock
    if hasattr(module,'integrity'):module.integrity=lambda *a,**k:check_lock()

def supervised(seed,name,p):
    import numpy as np,joblib
    import src.final_runs.vendor_supervised as training
    from src.phase4.data import training_arrays
    from threadpoolctl import threadpool_limits
    cfg=read(ROOT/'configs/phase4_seed42.json');cfg['seed']=seed
    training.SEED=seed
    training.METRICS=p/'metrics';training.MODELS=p/'models'
    # Explicit paths avoid every development seed42 output naming convention.
    def path(name):return p/'models'/(name+('.ubj' if name=='xgboost' else '.joblib'))
    training.model_path=path
    training.guard=check_lock
    x,y=training_arrays();selector=training.Selection(name,cfg);started=time.perf_counter()
    with threadpool_limits(limits=2),training.MemoryMonitor() as memory:
        if name in ('logistic_regression','mlp'):record=training.train_incremental(name,cfg,x,y,selector)
        elif name=='random_forest':record=training.train_forest(cfg,x,y,selector)
        else:record=training.train_xgboost(cfg,x,y,selector)
    record.update(status='complete',seed=seed,model=name,selected_step=selector.best_step,
        selected_validation_macro_f1=selector.best,model_path=str(path(name)),model_sha256=sha(path(name)),
        model_size_bytes=path(name).stat().st_size,peak_memory_bytes=memory.peak,training_stage_seconds=time.perf_counter()-started,
        validation_selection_seconds=selector.overhead_seconds,training_targets=len(y),effective_configuration_sha256=sha(OUT/f'audit/effective/seed_{seed}/{name}.json'))
    write(p/'metrics'/f'{name}_training.json',record)
    state=read(p/'metrics'/f'{name}_selection.json');state['status']='complete';write(p/'metrics'/f'{name}_selection.json',state)

def neural(seed,group,name,p):
    from copy import deepcopy
    alias={'temporal':'vendor_temporal','graph':'vendor_graph','anomaly':'vendor_anomaly','graph_temporal':'vendor_graph_temporal'}[group]
    m=importlib.import_module('src.final_runs.'+alias);m.SEED=seed;bind(m,p)
    if group=='temporal':
        original='transformer_L1_control' if name=='transformer_L1' else 'transformer_L64_d64_n2_p01'
        config=read(ROOT/f'results/temporal_v1/configs/{original}.json');config.update(name=name,seed=seed)
        m.PROTOCOL=deepcopy(SPEC['protocols']['temporal']);m.PROTOCOL.update(seed=seed,lengths=[1,64])
        write(p/'configs/protocol.json',m.PROTOCOL)
        # fit expects its original architecture-only config file; effective config is separately locked.
        m.fit(config)
    elif group=='graph':
        original={'gatv2':'gatv2_w1_h128_p2','edge_mlp':'edge_mlp','gatv2_self_only':'gatv2_self_only'}[name]
        config=read(ROOT/f'results/graph_v1/configs/{original}.json');config.update(name=name,seed=seed)
        m.PROTOCOL=deepcopy(SPEC['protocols']['graph']);m.PROTOCOL['seed']=seed
        write(p/'configs/protocol.json',m.PROTOCOL)
        for filename,source in [('data_manifest.json','metrics/data_manifest.json')]:
            write(p/'metrics'/filename,read(ROOT/'results/graph_v1'/source))
        write(p/'configs/preprocessing.json',read(ROOT/'results/graph_v1/configs/preprocessing.json'))
        m.fit(config)
        if name=='gatv2':write(p/'selected_model.json',dict(config=config,model_sha256=sha(p/'models/gatv2.pt')))
    elif group=='anomaly':
        d=importlib.import_module('src.final_runs.vendor_anomaly_data');d.SEED=seed;d.OUT=p;d.guard=check_lock
        m.PROTOCOL=deepcopy(SPEC['protocols']['anomaly']);m.PROTOCOL.update(seed=seed,architectures={'autoencoder_B':[128,32,128]})
        d.PROTOCOL=m.PROTOCOL
        write(p/'configs/protocol.json',m.PROTOCOL)
        write(p/'anomaly_training_manifest.json',read(ROOT/'results/anomaly_v1/anomaly_training_manifest.json'))
        m.fit(name)
    else:
        m.GRAPH=folder(seed,'graph');m.NAME='gatv2'
        m.PROTOCOL=deepcopy(SPEC['protocols']['graph_temporal']);m.PROTOCOL.update(seed=seed,lengths=[1,8])
        write(p/'configs/protocol.json',m.PROTOCOL)
        for file in ('history_indices.npy','history_identity_types.npy'):
            # Per-seed copies of immutable nonlearned identities, never development embeddings.
            import shutil
            if not (p/'metrics'/file).exists():shutil.copyfile(ROOT/'results/graph_temporal_v1/metrics'/file,p/'metrics'/file)
        write(p/'metrics/coverage_gate.json',read(ROOT/'results/graph_temporal_v1/metrics/coverage_gate.json'))
        m.embed();m.fit(int(name.rsplit('L',1)[1]))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--group',required=True);parser.add_argument('--model',required=True);a=parser.parse_args()
    require(a.seed in SEEDS and a.model in GROUPS[a.group],'Unauthorized seed/model');check_lock()
    p=setup(a.seed,a.group);require(not (p/'metrics'/f'{a.model}_training.json').exists(),'Refusing an already completed fit')
    mark('fit_started',a.seed,a.model);seed_all(a.seed)
    if a.group=='supervised':supervised(a.seed,a.model,p)
    else:neural(a.seed,a.group,a.model,p)
    check_lock();mark('fit_complete',a.seed,a.model)
if __name__=='__main__':main()
