"""Sequential, bounded seed-42 search; test remains outside all selection stages."""
import subprocess,sys
from .data import *

def execute(config):
    path=CONFIG/(config['name']+'.json')
    if path.exists(): require(read(path)==config,'Config mismatch')
    else: json_write(path,config)
    subprocess.run([sys.executable,'-m','src.temporal.train','--config',str(path)],check=True)
    return read(METRICS/(config['name']+'_training.json'))

def run():
    prepare()
    from .train import config_for
    results=[]
    for length in PROTOCOL['lengths']:
        results.append(execute(config_for(f'transformer_L{length}_d64_n2_p01',length,PROTOCOL['small_architecture'])))
    best_small=max(results,key=lambda r:r['selected_validation_macro_f1'])
    length=best_small['config']['length']
    results.append(execute(config_for(f'transformer_L{length}_d128_n3_p02',length,PROTOCOL['alternate_architecture'])))
    best=max(results,key=lambda r:r['selected_validation_macro_f1'])
    selected=dict(config=best['config'],model_sha256=best['model_sha256'],selected_epoch=best['selected_epoch'],
        selection_metric=PROTOCOL['selection_metric'],selection_partition='validation',selection_score=best['selected_validation_macro_f1'],
        protocol_sha256=sha256(CONFIG/'protocol.json'),candidate_names=[r['config']['name'] for r in results],seed=42)
    if (OUT/'selected_model.json').exists(): require(read(OUT/'selected_model.json')==selected,'Selected config changed')
    else: json_write(OUT/'selected_model.json',selected)
    arch={k:best['config'][k] for k in ('d_model','layers','heads','dropout')}
    control=execute(config_for('transformer_L1_control',best['config']['length'],arch,control=True))
    require(control['parameters']==best['parameters'],'Ablation capacity mismatch')
    write('development_search.csv',[dict(**r['config'],selected=r['config']['name']==best['config']['name'],
        selected_epoch=r['selected_epoch'],executed_epochs=r['executed_epochs'],validation_macro_f1=r['selected_validation_macro_f1'],
        training_seconds=r['training_seconds'],parameters=r['parameters']) for r in results+[control]])
    subprocess.run([sys.executable,'-m','src.temporal.evaluate','--freeze'],check=True)
    for name in (best['config']['name'],'transformer_L1_control'):
        subprocess.run([sys.executable,'-m','src.temporal.evaluate','--model',name],check=True)
    subprocess.run([sys.executable,'-m','src.temporal.report'],check=True)

if __name__=='__main__': run()
