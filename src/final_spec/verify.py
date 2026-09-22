"""Read-only semantic/output verification of the Phase10 freeze; no ML imports."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/final_spec_v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    required=['final_experiment_spec.json','model_registry.csv','seed_registry.csv','metric_definitions.md','research_questions.md',
              'claim_freeze.md','artifact_hash_manifest.csv','final_run_plan.md','integrity_report.md','FINAL_EXPERIMENT_SPEC.md']
    assert all((OUT/name).is_file() for name in required),'Missing required deliverable'
    spec=json.loads((OUT/'final_experiment_spec.json').read_text(encoding='utf-8'))
    integrity=json.loads((OUT/'freeze_integrity.json').read_text(encoding='utf-8'))
    assert spec['seeds']==[42,123,456,789,1024]
    assert spec['execution_authorized'] is False and spec['model_training_performed'] is False
    assert integrity['status']=='PASS' and integrity['prior_artifacts_unchanged']
    for name,digest in integrity['final_output_hashes'].items():assert sha(OUT/name)==digest, name+' hash changed'
    for source in spec['specification_generator']:assert sha(ROOT/source['path'])==source['sha256'],'Freeze generator changed'
    document=(OUT/'FINAL_EXPERIMENT_SPEC.md').read_text(encoding='utf-8')
    embedded=json.loads(document.split('```json\n')[-1].split('\n```')[0])
    assert embedded==spec,'Human and machine-readable specs differ'
    models=spec['models']
    assert sum(m['status']=='five_seed_required' for m in models.values())==11
    assert sum(m['status']=='five_seed_planned_resource_conditional' for m in models.values())==1
    assert models['isolation_forest']['status']=='development_seed42_secondary_only'
    assert models['rf_or_ae']['status']=='derived_no_training'
    assert 'gat_transformer_L4' not in models
    assert models['transformer_L64']['configuration']['length']==64
    assert models['transformer_L1']['configuration']['position_offset']==63
    assert models['gat_transformer_L8']['configuration']['length']==8
    assert models['gat_transformer_L8']['dependency']=='gatv2[s]'
    assert models['gatv2']['configuration']['hidden']==128
    assert models['autoencoder_B']['configuration']['hidden_layers']==[128,32,128]
    assert spec['protocols']['graph']['final_windows']==[1]
    assert spec['thresholds']['anomaly_primary']=={'protocol':'A_benign_only','cap':.01}
    for name,expected in [('full_preprocessing','results/baseline_v1/preprocessing.json'),('graph_preprocessing','results/graph_v1/configs/preprocessing.json')]:
        assert spec['data'][name]==json.loads((ROOT/expected).read_text(encoding='utf-8'))
        assert len(spec['data'][name]['output_feature_names'])==80
    with (OUT/'seed_registry.csv').open(newline='') as f:seeds=list(csv.DictReader(f))
    assert [int(r['seed']) for r in seeds]==spec['seeds']
    assert all(r['final_status']=='not_started_awaiting_explicit_approval' for r in seeds)
    with (OUT/'model_registry.csv').open(newline='') as f:registry=list(csv.DictReader(f))
    assert {r['model_id'] for r in registry}==set(models)
    with (OUT/'artifact_hash_manifest.csv').open(newline='') as f:manifest=list(csv.DictReader(f))
    assert len(manifest)==integrity['protected_files']
    assert sum(r['verification']=='historical_match' for r in manifest)==integrity['historical_hash_matches']
    print('PASS: all 10 required files; hashes; exact JSON/Markdown agreement; seeds; 11 required + 1 conditional models; exclusions; dependencies; preprocessing; approval gate.')

if __name__=='__main__':main()
