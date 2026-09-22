import subprocess,sys
from .data import *

def run():
    prepare();records=[]
    for name in list(PROTOCOL['architectures'])+['isolation_forest']:
        subprocess.run([sys.executable,'-m','src.anomaly.train','--name',name],check=True)
        record=read(METRICS/f'{name}_training.json')
        if name!='isolation_forest': records.append(record)
    best=min(records,key=lambda r:r['selected_benign_validation_mse'])
    selected=dict(name=best['config']['name'],architecture=best['config']['hidden_layers'],selected_epoch=best['selected_epoch'],
        benign_validation_mse=best['selected_benign_validation_mse'],selection_uses_malicious_validation=False,
        selection_uses_test=False,model_sha256=best['model_sha256'],protocol_sha256=sha256(CONFIG/'protocol.json'))
    path=OUT/'selected_autoencoder.json'
    if path.exists(): require(read(path)==selected,'AE selection changed')
    else: json_write(path,selected)
    write('development_search.csv',[dict(model=r['config']['name'],architecture=str(r['config']['hidden_layers']),
        selected=r['config']['name']==selected['name'],selected_epoch=r['selected_epoch'],executed_epochs=r['executed_epochs'],
        benign_training_rows=r['training_rows'],benign_validation_mse=r['selected_benign_validation_mse'],training_seconds=r['training_seconds']) for r in records])
    subprocess.run([sys.executable,'-m','src.anomaly.evaluate','--stage','A'],check=True)
    for name in (selected['name'],'isolation_forest'):
        subprocess.run([sys.executable,'-m','src.anomaly.evaluate','--stage','score','--name',name],check=True)
    subprocess.run([sys.executable,'-m','src.anomaly.evaluate','--stage','B'],check=True)
    subprocess.run([sys.executable,'-m','src.anomaly.report'],check=True)

if __name__=='__main__': run()
