"""Scoped Phase 11B orchestration. Preparation never calls a fitting worker."""
import argparse
import csv
import importlib
import io
import platform
import shutil
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path
from .common import ROOT, OUT, SPEC, SEEDS, read, write, sha, require, check_lock, folder

AUDIT = OUT / 'audit/phase11b/continuation_v2'
ORDER = [('graph', 'edge_mlp'), ('graph', 'gatv2'),
         ('graph_temporal', 'gat_transformer_L1'), ('graph_temporal', 'gat_transformer_L8'),
         ('graph', 'gatv2_self_only')]
HISTORY = ROOT / 'results/graph_temporal_v1/metrics'

def environment():
    """Called inside each actual worker process, before importing its fit entry point."""
    rows = []
    for name, version in SPEC['runtime']['software'].items():
        m = None if name == 'python' else importlib.import_module({'scikit_learn':'sklearn'}.get(name,name))
        actual = platform.python_version() if m is None else m.__version__
        rows.append(dict(package=name, required=version, observed=actual,
                         path=sys.executable if m is None else m.__file__))
        require(actual == version, 'Frozen dependency mismatch: '+name)
    for name, directory in [('sklearn','phase4_runtime'), ('xgboost','phase4_runtime'), ('torch','temporal_runtime')]:
        path = __import__('pathlib').Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to((ROOT/'data'/directory).resolve()), 'Required overlay not used: '+name)
    import torch
    import psutil
    import os
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key,'ProcessorNameString')[0]
    hardware = dict(cpu=cpu,physical_cores=psutil.cpu_count(logical=False),logical_cpus=os.cpu_count(),
                    ram_bytes=psutil.virtual_memory().total,platform=platform.platform())
    require(hardware == SPEC['runtime']['hardware'], 'Frozen hardware mismatch')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,
                software=rows,hardware=hardware,threads=2,deterministic=True)

def sources():
    return sorted((ROOT/'src/final_runs').glob('*.py')) + [ROOT/'run_phase11b.ps1']

def verify_sources():
    check_lock()
    for r in read(AUDIT/'source_lock.json')['files']:
        require(sha(ROOT/r['path']) == r['sha256'], 'Phase11B source changed: '+r['path'])

def protected(recheck=False):
    """Historical hashes plus recovery before/after hashes, without rewriting old audits."""
    baseline = OUT/'audit/phase11b/environment_recovery/protected_artifact_verification.csv'
    expected = {}
    for r in csv.DictReader(baseline.open(encoding='utf-8-sig')):
        expected[r['path']] = r['after_sha256']
    manifest = ROOT/SPEC['artifact_hash_manifest']['path']
    require(sha(manifest)==SPEC['artifact_hash_manifest']['sha256'],'Frozen manifest changed')
    for r in csv.DictReader(manifest.open(encoding='utf-8-sig')):
        p = str((ROOT/r['path']).resolve())
        require(p not in expected or expected[p]==r['sha256'],'Inconsistent protected baseline')
        expected[p] = r['sha256']
    # Preserve the recovery record itself and the original failure history.
    if recheck:
        expected.update(read(AUDIT/'protected_baseline.json'))
    else:
        for p in (OUT/'audit/phase11b/environment_recovery').rglob('*'):
            if p.is_file(): expected[str(p.resolve())]=sha(p)
    records=[]
    for i,(path,digest) in enumerate(sorted(expected.items())):
        actual=sha(Path(path))
        records.append(dict(path=path,expected_sha256=digest,observed_sha256=actual,status='PASS' if digest==actual else 'FAIL'))
        require(digest==actual,'Protected artifact changed: '+path)
        if i%100==0: print('Protected hashes verified',i,flush=True)
    write(AUDIT/('protected_final.json' if recheck else 'protected_baseline.json'),
          records if recheck else expected)
    return records

def semantics():
    import numpy as np
    import torch
    from .build_vendor import SOURCES, transformed
    for name in ('vendor_graph','vendor_graph_temporal'):
        source,replacements=SOURCES[name]
        require((ROOT/f'src/final_runs/{name}.py').read_text(encoding='utf-8') ==
                transformed((ROOT/source).read_text(encoding='utf-8-sig'),replacements),'Frozen algorithm mismatch')
    suite=unittest.TestSuite()
    for module in ('src.graph.test_graph','src.graph_temporal.test_temporal','src.final_runs.test_phase11b'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (AUDIT/'semantic_tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    require(result.wasSuccessful(),'Semantic tests failed')
    from src.graph.data import GraphData
    from src.graph_temporal.prepare import build_history
    from src.graph_temporal.model import TemporalHead
    from src.graph.model import EdgeModel
    from src.phase4.train import MemoryMonitor
    with MemoryMonitor() as memory:
        begin=time.perf_counter(); data=GraphData(); loading=time.perf_counter()-begin
        require(data.x.shape==(474357,80) and data.hx.shape==(7948746,9),'Graph dimensions')
        require([int((data.part==i).sum()) for i in range(3)]==[172984,88782,212591],'Cohort counts')
        require(np.array_equal(data.ids,np.load(HISTORY/'eligible_target_ids.npy')),'Cohort identities')
        for g in data.groups:
            snap=data.snapshot(g,1)
            require(len(snap['history_rows'])<=8192,'Message cap')
            require((data.meta.completion_us.to_numpy()[snap['history_rows']]<snap['cutoff']*1000000).all(),'Future graph history')
        timings=[]
        _,h,types,_=build_history(data.meta,data.ids,timings)
        require(np.array_equal(h,np.load(HISTORY/'history_indices.npy')),'History identity mismatch')
        require(np.array_equal(types,np.load(HISTORY/'history_identity_types.npy')),'History fallback mismatch')
        for name in ('edge_mlp','gatv2_w1_h128_p2','gatv2_self_only'):
            c=read(ROOT/f'results/graph_v1/configs/{name}.json')
            require(c['window_minutes']==1 and c['hidden']==128 and c['dropout']==.2,'Graph configuration mismatch')
            EdgeModel(c,80,9)
        require(sum(p.numel() for p in TemporalHead(336).parameters())==89089,'Temporal capacity')
    write(AUDIT/'history_replay.json',dict(status='PASS',timings=timings,peak_memory_bytes=memory.peak,
        metadata_loading_seconds=loading,scope='Shared offline replay of frozen history, including diagnostic counts; no coverage gate reselection'))
    return dict(status='PASS',tests=result.testsRun,all_graph_prefixes=True,exact_history_replay=True,no_training=True)

def prepare():
    AUDIT.mkdir(parents=True,exist_ok=False)
    write(AUDIT/'authorization.json',dict(scope='Phase11B only; execution requires explicit --execute command',
        seeds=SEEDS,order=ORDER,mandatory_fits=20,planned_self_only=5,
        history='Original restricted invocation stopped before training; recovery verified existing overlays with no installation, environment changes or training. Both records preserved.'))
    try:
        write(AUDIT/'environment.json',environment())
        check_lock()
        from src.final_spec.verify import main as verify_spec
        verify_spec()
        from .phase11a import inventory
        rows=inventory()
        require(len(rows)==35 and all(r['status']=='fit_and_evaluation_complete' for r in rows),'Phase11A incomplete')
        for seed in SEEDS:
            for group in ('graph','graph_temporal'):
                p=folder(seed,group)
                require(not p.exists() or not any(p.rglob('*')),'Existing Phase11B artifacts require reviewed recovery: '+str(p))
        protected()
        snapshot=AUDIT/'source_snapshot';snapshot.mkdir()
        records=[]
        for p in sources():
            shutil.copyfile(p,snapshot/p.name)
            records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
        write(AUDIT/'source_lock.json',dict(files=records))
        write(AUDIT/'semantic_checks.json',semantics())
        verify_sources()
        protected(recheck=True)
        write(AUDIT/'prepared.json',dict(status='PASS',training_started=False))
        print('PREPARED: no training performed. Run --execute separately.',flush=True)
    except Exception:
        write(AUDIT/'STOP_FAILURE.json',dict(stage='prepare',traceback=traceback.format_exc(),training_started=False))
        raise

def selection_seed(record):
    return record.get('seed',record.get('config',{}).get('seed'))

def dependency(seed):
    p=folder(seed,'graph'); t=read(p/'metrics/gatv2_training.json')
    require(selection_seed(t)==seed and t['status']=='complete','Encoder seed/status mismatch')
    require(sha(p/'models/gatv2.pt')==t['model_sha256'],'Encoder checkpoint mismatch')
    manifest=folder(seed,'graph_temporal')/'metrics/representation_manifest.json'
    if manifest.exists():
        r=read(manifest)
        require(r['encoder_sha256']==t['model_sha256'],'Representations use wrong encoder')
        require(sha(manifest.parent/'representations.npy')==r['sha256'],'Representations changed')
    return t['model_sha256']

def stage(action,seed,group,name):
    require((group,name) in ORDER and seed in SEEDS,'Outside authorized scope')
    require(read(AUDIT/'prepared.json')['status']=='PASS','Preparation has not passed')
    require((AUDIT/'execution_started.json').exists(),'Use the scoped execution controller')
    require(not (AUDIT/'STOP_FAILURE.json').exists(),'Preserved failure prevents continuation')
    verify_sources()
    label=f'{action}_{seed}_{name}'
    write(AUDIT/f'{label}_environment.json',environment())
    p=folder(seed,group)
    if group=='graph_temporal': dependency(seed)
    from threadpoolctl import threadpool_limits
    from src.phase4.train import MemoryMonitor
    from . import common
    # Keep new audit events separate from protected Phase11A history.
    def mark(event,s,m,**details):
        with (AUDIT/'events.jsonl').open('a',encoding='utf-8') as f:
            import json
            f.write(json.dumps(dict(event=event,seed=s,model=m,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details))+'\n')
    common.mark=mark
    module=importlib.import_module('src.final_runs.'+('worker' if action=='fit' else 'evaluate'))
    module.mark=mark
    if action=='evaluate':
        original_read=module.read
        module.read=lambda path: original_read(AUDIT/'history_replay.json' if path==OUT/'audit/history_replay.json' else path)
    # Measure graph input construction separately from fit wall time, without changing its order.
    preparation={'training_input_seconds':0.,'metadata_loading_seconds':0.}
    if action=='fit' and group=='graph':
        from . import vendor_graph as vg
        old_data=vg.GraphData; old_tensors=vg.tensors; old_predict=vg.predict
        selecting=[False]
        class MeasuredData(old_data):
            def __init__(self):
                start=time.perf_counter();super().__init__();preparation['metadata_loading_seconds']+=time.perf_counter()-start
            def snapshot(self,*args,**kwargs):
                start=time.perf_counter();result=super().snapshot(*args,**kwargs)
                if not selecting[0]: preparation['training_input_seconds']+=time.perf_counter()-start
                return result
        def tensor_input(*args,**kwargs):
            start=time.perf_counter();result=old_tensors(*args,**kwargs)
            if not selecting[0]: preparation['training_input_seconds']+=time.perf_counter()-start
            return result
        def predict(*args,**kwargs):
            selecting[0]=True
            try:return old_predict(*args,**kwargs)
            finally:selecting[0]=False
        vg.GraphData=MeasuredData;vg.tensors=tensor_input;vg.predict=predict
    sys.argv=[label,'--seed',str(seed),'--group',group,'--model',name]
    start=time.perf_counter()
    with threadpool_limits(limits=2),MemoryMonitor() as memory:
        module.main()
    if group=='graph_temporal': dependency(seed)
    write(AUDIT/f'{label}_runtime.json',dict(stage_seconds=time.perf_counter()-start,
        process_peak_memory_bytes=memory.peak,**preparation,
        scope='Graph snapshot/tensor construction separated; remaining fit wall time includes minibatch overhead. Temporal fit includes lookup. Shared representation/history costs reported separately.'))
    verify_sources()

def execute():
    require(read(AUDIT/'prepared.json')['status']=='PASS','Run preparation first')
    require(not (AUDIT/'execution_started.json').exists(),'Attempt already started; automatic resume forbidden')
    require(not (AUDIT/'STOP_FAILURE.json').exists(),'Review preserved failure first')
    verify_sources(); environment(); protected(recheck=True)
    write(AUDIT/'execution_started.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    try:
        for seed in SEEDS:
            for group,name in ORDER:
                for action in ('fit','evaluate'):
                    label=f'{action}_{seed}_{name}';log=AUDIT/f'{label}.log'
                    print('START',label,flush=True)
                    with log.open('x',encoding='utf-8') as f:
                        result=subprocess.run([sys.executable,'-B','-u','-m','src.final_runs.phase11b','--stage',action,
                            '--seed',str(seed),'--group',group,'--model',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                    require(result.returncode==0,'Stage failed; preserved logs, no retry: '+label)
                    print('PASS',label,flush=True)
        from .report11b import main as report
        report()
    except Exception:
        write(AUDIT/'STOP_FAILURE.json',dict(status='STOPPED',traceback=traceback.format_exc(),
            self_only_policy='Any self-only failure is preserved for measured resource review; no automatic omission or substitution'))
        (AUDIT/'STOPPED.md').write_text('Phase11B incomplete. See STOP_FAILURE.json and stage logs. No automatic retry.\n',encoding='utf-8')
        raise

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--stage',choices=['fit','evaluate'])
    parser.add_argument('--seed',type=int);parser.add_argument('--group');parser.add_argument('--model')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.execute:execute()
    else:stage(args.stage,args.seed,args.group,args.model)

if __name__=='__main__':main()
