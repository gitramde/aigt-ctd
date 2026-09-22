"""Bounded seed-42 graph experiment; primary selection precedes held-out tests."""
import subprocess,sys
from .common import *


def execute(config):
    path=CONFIG/(config['name']+'.json')
    if path.exists(): require(read(path)==config,'Configuration changed')
    else: json_write(path,config)
    subprocess.run([sys.executable,'-m','src.graph.train','--config',str(path)],check=True)
    return read(METRICS/f"{config['name']}_training.json")


def run():
    from .data import prepare
    prepare()
    if not (OUT/'graph_diagnostics_summary.csv').exists():
        subprocess.run([sys.executable,'-m','src.graph.data'],check=True)
    candidates=[]
    for arch in PROTOCOL['candidates']:
        config=dict(name=f"gatv2_w1_h{arch['hidden']}_p{int(arch['dropout']*10)}",kind='gatv2',window_minutes=1,seed=42,**arch)
        candidates.append(execute(config))
    best=max(candidates,key=lambda r:r['selected_validation_macro_f1'])
    config=best['config'];arch={k:config[k] for k in ('hidden','heads','layers','dropout')}
    primary=dict(config=config,model_sha256=best['model_sha256'],validation_macro_f1=best['selected_validation_macro_f1'],
        selected_epoch=best['selected_epoch'],selection_partition='validation',primary_window_predeclared=True)
    p=CONFIG/'primary_selection.json'
    if p.exists(): require(read(p)==primary,'Primary selection changed')
    else: json_write(p,primary)
    from .train import make_model
    target_parameters=best['parameters'];edge_dim=len(read(METRICS/'data_manifest.json')['edge_feature_names'])
    # Two hidden layers: D*w+w + w*w+w + w+1.
    width=min(range(16,1025),key=lambda w:abs(w*w+(edge_dim+3)*w+1-target_parameters))
    controls=[]
    controls.append(execute(dict(name='edge_mlp',kind='edge_mlp',window_minutes=1,seed=42,mlp_width=width,**arch)))
    controls.append(execute(dict(name='gatv2_self_only',kind='self_only',window_minutes=1,seed=42,**arch)))
    sensitivities=[]
    for window in PROTOCOL['sensitivity_windows']:
        sensitivities.append(execute(dict(name=f'gatv2_w{window}',kind='gatv2',window_minutes=window,seed=42,**arch)))
    from .baselines import NAMES
    for name in NAMES: subprocess.run([sys.executable,'-m','src.graph.baselines','--name',name],check=True)
    names=[config['name']]+[r['config']['name'] for r in controls+sensitivities]+list(NAMES)
    selected=dict(**primary,evaluation_models=names,candidates=[r['config']['name'] for r in candidates],
        protocol_sha256=sha256(CONFIG/'protocol.json'),cohort_sha256=sha256(METRICS/'evaluation_cohort.parquet'))
    destination=OUT/'selected_model.json'
    if destination.exists(): require(read(destination)==selected,'Selection artifact changed')
    else: json_write(destination,selected)
    write('development_search.csv',[dict(**r['config'],selected_primary=r['config']['name']==config['name'],
        validation_macro_f1=r['selected_validation_macro_f1'],selected_epoch=r['selected_epoch'],executed_epochs=r['executed_epochs'],
        parameters=r['parameters'],training_seconds=r['training_seconds']) for r in candidates+controls+sensitivities])
    subprocess.run([sys.executable,'-m','src.graph.evaluate','--freeze'],check=True)
    for name in names: subprocess.run([sys.executable,'-m','src.graph.evaluate','--name',name],check=True)
    subprocess.run([sys.executable,'-m','src.graph.report'],check=True)

if __name__=='__main__': run()
