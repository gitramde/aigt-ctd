import csv,json,time,hashlib
from pathlib import Path
import numpy as np
from . import ROOT
from src.baseline.common import require,sha256,json_write,csv_write
from src.baseline.preprocess import iter_transformed
from src.phase4.data import BASE,CACHE as BASE_CACHE,guard as baseline_guard,prep_config,training_arrays,diagnostic_labels

OUT=ROOT/'results/anomaly_v1';CONFIG=OUT/'configs';METRICS=OUT/'metrics';MODELS=OUT/'models';FIGURES=OUT/'figures';CACHE=ROOT/'data/anomaly_v1'
PROTOCOL=dict(seed=42,phase='development',training_population='all benign TRAIN rows; no AE subsampling',
    architectures={'autoencoder_A':[128,64,32,64,128],'autoencoder_B':[128,32,128]},
    activation='ReLU',dropout=0.,optimizer='AdamW',learning_rate=.001,weight_decay=.00001,
    batch_size=8192,read_block_size=65536,max_epochs=4,patience=2,min_relative_improvement=.0001,threads=2,
    loss='mean squared reconstruction error',score='per-record feature-mean squared reconstruction error; higher is more anomalous',
    selection='lowest benign-validation mean reconstruction error; malicious validation scores unavailable to selection',
    protocol_A_caps=[.05,.01,.005,.001],protocol_A_rule='conservative upper order statistic with >= detection; nextafter boundary handles ties',
    protocol_B_objectives=['maximum_macro_f1','maximum_binary_f1'],
    isolation_forest=dict(sample_size=250000,sampling='uniform without replacement from benign TRAIN IDs using NumPy seed 42',
        n_estimators=100,max_samples=256,contamination='auto',max_features=1.0,bootstrap=False,n_jobs=2,random_state=42),
    temporal_autoencoder='not run; optional secondary scope deferred',
    preprocessing='reuse baseline_v1 train-fitted preprocessing unchanged; it was fitted on all TRAIN classes, not benign-only')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,rows): csv_write(OUT/name,rows)
def guard(full=False):
    baseline_guard(full=full)
    require(read(CONFIG/'protocol.json')==PROTOCOL,'Anomaly protocol changed')
    for row in read(METRICS/'frozen_artifacts.json'):
        require(sha256(row['path'])==row['sha256'],'Protected artifact changed: '+row['path'])

def prepare():
    for folder in (CONFIG,METRICS,MODELS,FIGURES,CACHE): folder.mkdir(parents=True,exist_ok=True)
    if not (CONFIG/'protocol.json').exists(): json_write(CONFIG/'protocol.json',PROTOCOL)
    if not (METRICS/'frozen_artifacts.json').exists():
        files=[]
        for folder in ('results/baseline_v1','results/temporal_v1','results/graph_v1','src/baseline','src/phase4','src/phase4a','src/temporal','src/graph'):
            files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        json_write(METRICS/'frozen_artifacts.json',[dict(path=str(p),sha256=sha256(p)) for p in files])
    print('Verifying frozen baseline data and prior phases',flush=True);guard(full=True)
    cache=read(BASE_CACHE/'training_cache.json')
    for key in ('x','y'): require(sha256(cache[key+'_path'])==cache[key+'_sha256'],'Training cache changed')
    destination=OUT/'anomaly_training_manifest.json'
    if destination.exists():
        for row in read(destination)['artifacts']: require(sha256(row['path'])==row['sha256'],'Anomaly input cache changed')
        return
    x,y=training_arrays();benign=np.flatnonzero(y==0)
    expected=read(BASE/'split_plan.json')['partitions']
    require(len(benign)==expected['train']['class_counts']['Benign'],'Benign training count mismatch')
    np.save(CACHE/'benign_train_ids.npy',benign)
    selected=np.sort(np.random.default_rng(42).choice(benign,size=PROTOCOL['isolation_forest']['sample_size'],replace=False))
    np.save(CACHE/'isolation_train_ids.npy',selected)
    n=expected['validation']['class_counts']['Benign'];d=cache['features']
    val=np.lib.format.open_memmap(CACHE/'benign_validation_features.npy',mode='w+',dtype=np.float32,shape=(n,d))
    digest=hashlib.sha256();offset=0
    for xb,yb,metadata in iter_transformed(prep_config(),'validation',scaled=True):
        digest.update(np.ascontiguousarray(xb,dtype='<f8').tobytes())
        mask=yb==0;count=int(mask.sum());val[offset:offset+count]=xb[mask];offset+=count
    with (BASE/'preprocessing_application_audit.csv').open(newline='') as f:
        target=next(r['transformed_float64_sha256'] for r in csv.DictReader(f) if r['partition']=='validation')
    require(offset==n and digest.hexdigest()==target,'Frozen validation transform mismatch');val.flush();del val
    artifacts=[CACHE/'benign_train_ids.npy',CACHE/'isolation_train_ids.npy',CACHE/'benign_validation_features.npy',CONFIG/'protocol.json']
    json_write(destination,dict(status='PASS',seed=42,train_rows=len(y),benign_train_rows=len(benign),autoencoder_training_rows=len(benign),
        autoencoder_subsampled=False,isolation_forest_fit_rows=len(selected),isolation_forest_tree_subsample=256,
        benign_validation_rows=n,partitions={p:dict(rows=v['rows'],dates=v['dates'],membership_sha256=v['membership_sha256']) for p,v in expected.items()},
        preprocessing_sha256=sha256(BASE/'preprocessing.json'),training_cache_sha256=cache['x_sha256'],
        benign_ids_sha256=sha256(CACHE/'benign_train_ids.npy'),isolation_sample_sha256=sha256(CACHE/'isolation_train_ids.npy'),
        label_use='binary labels only identify benign fitting/calibration cohorts; no labels in reconstruction objective or Isolation Forest fit',
        artifacts=[dict(path=str(p),sha256=sha256(p)) for p in artifacts]))
    print('All',len(benign),'benign TRAIN rows reserved for every AE epoch',flush=True)

def validation_benign(): return np.load(CACHE/'benign_validation_features.npy',mmap_mode='r')

def train_batches(epoch=None):
    x,y=training_arrays();block=PROTOCOL['read_block_size'];starts=np.arange(0,len(y),block)
    rng=np.random.default_rng(42 if epoch is None else 42+epoch-1)
    if epoch is not None: starts=rng.permutation(starts)
    for start in starts:
        stop=min(int(start)+block,len(y));ids=np.flatnonzero(y[start:stop]==0)+start
        if epoch is not None: ids=rng.permutation(ids)
        for offset in range(0,len(ids),PROTOCOL['batch_size']):
            positions=ids[offset:offset+PROTOCOL['batch_size']]
            if len(positions): yield np.array(x[positions],dtype=np.float32)

def partition_batches(part):
    digest=hashlib.sha256();offset=0
    for x,y,metadata in iter_transformed(prep_config(),part,scaled=True):
        digest.update(np.ascontiguousarray(x,dtype='<f8').tobytes());offset+=len(x)
        yield np.asarray(x,dtype=np.float32)
    with (BASE/'preprocessing_application_audit.csv').open(newline='') as f:
        expected=next(r['transformed_float64_sha256'] for r in csv.DictReader(f) if r['partition']==part)
    require(digest.hexdigest()==expected,'Frozen transform changed: '+part)
    require(offset==read(BASE/'split_plan.json')['partitions'][part]['rows'],'Partition incomplete')

def labels(part):
    require((BASE_CACHE/f'{part}_diagnostics.npz').exists(),'Frozen diagnostic cache missing')
    return diagnostic_labels(part)

if __name__=='__main__': prepare()
