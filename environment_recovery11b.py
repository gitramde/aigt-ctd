"""Read-only modeling recovery checks; writes only a new recovery audit directory."""
import csv
import importlib
import json
import platform
import sys
from pathlib import Path
import src.final_runs
from src.final_runs.common import ROOT, OUT, SPEC, sha, read, require, check_lock

DEST = OUT / 'audit/phase11b/environment_recovery'

def save(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def table(name, rows):
    with (DEST / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    DEST.mkdir(exist_ok=False)
    save('environment_identity.json', dict(executable=sys.executable,
        python=platform.python_version(), prefix=sys.prefix, base_prefix=sys.base_prefix,
        environment='Anaconda base interpreter plus existing project-local modeling package overlays; not a standalone venv',
        overlays=[str(ROOT/'data/temporal_runtime'), str(ROOT/'data/phase4_runtime')],
        preparation_venv='Project .venv uses Python 3.11.7 and is not the recorded modeling runtime',
        original_attempt='Stopped before training; original failure record preserved unchanged',
        recovery_access='Executed with access to existing project-local package directories',
        original_observation='Restricted process resolved base sklearn 1.6.1; local sklearn file access was denied',
        recovery_source_sha256=sha(Path(__file__))))
    aliases={'scikit_learn':'sklearn'}
    deps=[]
    for name, expected in SPEC['runtime']['software'].items():
        module=None if name=='python' else importlib.import_module(aliases.get(name,name))
        actual=platform.python_version() if module is None else module.__version__
        deps.append(dict(package=name, required_version=expected, observed_version=actual,
            status='PASS' if actual==expected else 'FAIL', module_path='interpreter' if module is None else module.__file__))
    table('dependency_verification.csv',deps)
    require(all(r['status']=='PASS' for r in deps),'Environment is not compliant')
    check_lock()
    records=[]
    expected={}
    manifest=ROOT/SPEC['artifact_hash_manifest']['path']
    require(sha(manifest)==SPEC['artifact_hash_manifest']['sha256'],'Frozen manifest mismatch')
    for row in csv.DictReader(manifest.open(encoding='utf-8-sig')):
        expected[str((ROOT/row['path']).resolve())]=row['sha256']
    lock=read(OUT/'audit/runner_lock.json')
    expected[str((ROOT/'results/final_spec_v1/final_experiment_spec.json').resolve())]=lock['spec_sha256']
    for row in lock['runner_sources']+lock['effective_configurations']:
        expected[str((ROOT/row['path']).resolve())]=row['sha256']
    from src.final_runs.phase11a import inventory
    inventory_rows=inventory()
    require(len(inventory_rows)==35 and all(r['status']=='fit_and_evaluation_complete' for r in inventory_rows),'Phase11A incomplete')
    save('phase11a_inventory.json',inventory_rows)
    for seed in SPEC['seeds']:
        for group in ('supervised','temporal','anomaly'):
            base=OUT/f'seed_{seed}'/group
            for p in (base/'metrics').glob('*_evaluation.json'):
                entry=read(p)
                for a in entry['artifacts']: expected[str((ROOT/a['path']).resolve())]=a['sha256']
    protected=set(Path(p) for p in expected)
    for folder in [ROOT/'results/final_spec_v1',OUT/'aggregate_phase11a',OUT/'audit/phase11a']:
        protected.update(p.resolve() for p in folder.rglob('*') if p.is_file())
    for seed in SPEC['seeds']:
        protected.update(p.resolve() for p in (OUT/f'seed_{seed}').rglob('*') if p.is_file())
    protected.add((OUT/'PHASE_11A_EXECUTION_SUMMARY.md').resolve())
    failure=OUT/'audit/phase11b/environment_failure_20260922_213037_151.json'
    protected.add(failure.resolve())
    for i,p in enumerate(sorted(protected)):
        actual=sha(p); baseline=expected.get(str(p))
        records.append(dict(path=str(p), expected_sha256=baseline or '', before_sha256=actual,
            after_sha256='', status='PASS' if baseline==actual else 'BASELINED_THIS_RECOVERY' if baseline is None else 'FAIL'))
        if i%50==0: print('Protected artifacts hashed:',i,flush=True)
    table('protected_artifact_verification.csv',records)
    require(not any(r['status']=='FAIL' for r in records),'Protected historical hash mismatch')
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits, threadpool_info
    from src.baseline.preprocess import FrozenPreprocessor
    from src.graph.model import EdgeModel
    from src.graph_temporal.model import TemporalHead
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checks=[]
    with threadpool_limits(limits=2):
        require(torch.get_num_threads()==2 and torch.are_deterministic_algorithms_enabled(),'Thread/determinism mismatch')
        save('thread_settings.json',dict(torch_threads=2, deterministic=True, pools=threadpool_info()))
        for state in (SPEC['data']['full_preprocessing'], SPEC['data']['graph_preprocessing']):
            prep=FrozenPreprocessor(state)
            require(len(prep.output_names)==80 and not prep.means.flags.writeable,'Frozen preprocessing shape/mutability')
            source=SPEC['data']['full_split']['partitions']['train']['files'][0]['cleaned_path']
            batch=next(pq.ParquetFile(source).iter_batches(batch_size=8,columns=list(prep.numeric)+list(prep.categorical)))
            x,_=prep.transform(batch)
            require(x.shape==(8,80) and np.isfinite(x).all(),'Transform-only smoke failure')
        checks.append('Both frozen preprocessing states loaded; eight TRAIN records transformed with immutable arrays; no refitting')
        x=np.load(ROOT/'data/graph_v1/target_features.npy',mmap_mode='r')
        hx=np.load(ROOT/'data/graph_v1/history_features.npy',mmap_mode='r')
        ids=np.load(ROOT/'data/graph_v1/target_ids.npy')
        require(x.shape==(474357,80) and hx.shape==(7948746,9),'Graph dimensions')
        h=np.load(ROOT/'results/graph_temporal_v1/metrics/history_indices.npy')
        types=np.load(ROOT/'results/graph_temporal_v1/metrics/history_identity_types.npy')
        require(h.shape==types.shape==(474357,8),'History shape')
        require(np.array_equal(ids,np.load(ROOT/'results/graph_temporal_v1/metrics/eligible_target_ids.npy')), 'Graph/history targets')
        partitions=pq.read_table(ROOT/'data/graph_v1/metadata.parquet',columns=['partition']).column(0).to_numpy()[ids]
        require([int((partitions==i).sum()) for i in range(3)]==[172984,88782,212591],'Cohort counts')
        checks.append('Graph features, exact target ID/order equality, history dimensions, cohort counts and frozen artifact hashes verified')
        for name in ('edge_mlp','gatv2_w1_h128_p2','gatv2_self_only'):
            model=EdgeModel(read(ROOT/f'results/graph_v1/configs/{name}.json'),80,9)
            require(sum(p.numel() for p in model.parameters())>0,'Empty model')
        head=TemporalHead(336)
        require(sum(p.numel() for p in head.parameters())==89089,'Head capacity mismatch')
        checks.append('Edge MLP, GATv2, self-only and temporal head instantiated on CPU without fitting or evaluation')
    for i,r in enumerate(records):
        r['after_sha256']=sha(Path(r['path']))
        require(r['after_sha256']==r['before_sha256'],'Artifact changed during recovery: '+r['path'])
        if i%50==0: print('Protected artifacts rechecked:',i,flush=True)
    table('protected_artifact_verification.csv',records)
    (DEST/'preprocessing_compatibility.md').write_text('# Transform-only compatibility\n\nFrozenPreprocessor reads the stored JSON arrays, makes learned numeric arrays read-only, and performs imputation, scaling and ordered categorical encoding through transform(). The final workers consume existing model-input caches or this transform-only path. Preparation fitting code exists separately but was not invoked. Both frozen states passed a small TRAIN-only transform smoke test. No serialized sklearn estimator migration, refit, category change, or artifact rewrite was performed.\n',encoding='utf-8')
    (DEST/'smoke_test_report.md').write_text('# Non-training smoke tests: PASS\n\n'+'\n'.join('- '+c for c in checks)+'\n\nNo fit(), optimizer.step(), training loop, checkpoint selection, or test evaluation was called.\n',encoding='utf-8')
    (DEST/'PHASE_11B_ENVIRONMENT_RECOVERY_SUMMARY.md').write_text('# Phase 11B environment recovery\n\nStatus: **READY** (environment and non-training checks only).\n\nExisting runtime: C:\\ProgramData\\anaconda3\\python.exe, Python 3.13.5, with project-local data/phase4_runtime and data/temporal_runtime inserted by src.final_runs before library imports. All 12 frozen dependency versions pass, including sklearn 1.7.2. This is the recorded package-overlay runtime, not an independent venv. No packages or environments were installed or modified.\n\nThe original restricted-process failure remains unchanged and still records that attempt accurately. Recovery used access to the existing local modeling packages; it did not relax versions.\n\nAll non-training smoke checks pass. Protected artifacts match available historical hashes and their before/after recovery hashes. Files lacking historical hashes are explicitly marked BASELINED_THIS_RECOVERY; their earlier history cannot be established by a newly captured hash. Phase 11A completion/checkpoint/score provenance passed its inventory validation.\n\nNo Phase 11B training was started. Scoped Phase 11B runner/report implementation remains separate pending work.\n',encoding='utf-8')
    print('READY: recovery checks complete; no training performed.',flush=True)

if __name__=='__main__':
    main()
