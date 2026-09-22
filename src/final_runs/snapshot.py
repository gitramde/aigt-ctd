"""Freeze complete runner source and effective configuration before fitting."""
import shutil
from .common import *

def main():
    require(not (OUT/'audit/STOP_FAILURE.json').exists(),'Required check failed')
    for file in ('environment_check.json','artifact_check.json','seed_provenance_check.json','semantic_checks.json','calibration_check.json'):
        require(read(OUT/'audit'/file)['status']=='PASS','Preflight did not pass: '+file)
    require(not (OUT/'audit/runner_lock.json').exists(),'Runner snapshot already exists')
    files=sorted((ROOT/'src/final_runs').glob('*.py'))
    for p in files:compile(p.read_text(encoding='utf-8'),str(p),'exec')
    snapshot=OUT/'audit/source_snapshot';snapshot.mkdir()
    records=[]
    for p in files:
        shutil.copyfile(p,snapshot/p.name)
        records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    configs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((OUT/'audit/effective').rglob('*.json'))]
    require(len(configs)==60,'Effective fit configuration count mismatch')
    write(OUT/'audit/preflight_checks.json',dict(status='PASS',training_started=False,checks=['environment','592 artifact hashes','seed provenance','source equivalence','configuration','cohorts','causality','masking','exact history replay']))
    write(OUT/'audit/runner_lock.json',dict(spec_sha256=sha(ROOT/'results/final_spec_v1/final_experiment_spec.json'),runner_sources=records,effective_configurations=configs,
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    for seed in SEEDS:
        path=OUT/f'seed_{seed}';path.mkdir(exist_ok=True)
        write(path/'cohort_provenance.json',dict(seed=seed,full_dataset=SPEC['data']['full_split'],temporal=SPEC['data']['full_temporal_cohort'],graph=SPEC['data']['graph_data_manifest'],graph_temporal=SPEC['data']['graph_temporal_cohort']))
    print('Runner and 60 effective fit configurations snapshotted before first fit',flush=True)

if __name__=='__main__':main()
