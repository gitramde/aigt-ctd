SEED = 42
import csv, json, time, hashlib
from pathlib import Path
import numpy as np
from src.final_runs import ROOT
from src.baseline.common import require, sha256, json_write, csv_write
from src.baseline.preprocess import iter_transformed
from src.phase4.data import BASE, CACHE as BASE_CACHE, guard as baseline_guard, prep_config, training_arrays, diagnostic_labels
OUT = ROOT / 'results/anomaly_v1'
CONFIG = OUT / 'configs'
METRICS = OUT / 'metrics'
MODELS = OUT / 'models'
FIGURES = OUT / 'figures'
CACHE = ROOT / 'data/anomaly_v1'
PROTOCOL = dict(seed=SEED, phase='development', training_population='all benign TRAIN rows; no AE subsampling', architectures={'autoencoder_A': [128, 64, 32, 64, 128], 'autoencoder_B': [128, 32, 128]}, activation='ReLU', dropout=0.0, optimizer='AdamW', learning_rate=0.001, weight_decay=1e-05, batch_size=8192, read_block_size=65536, max_epochs=4, patience=2, min_relative_improvement=0.0001, threads=2, loss='mean squared reconstruction error', score='per-record feature-mean squared reconstruction error; higher is more anomalous', selection='lowest benign-validation mean reconstruction error; malicious validation scores unavailable to selection', protocol_A_caps=[0.05, 0.01, 0.005, 0.001], protocol_A_rule='conservative upper order statistic with >= detection; nextafter boundary handles ties', protocol_B_objectives=['maximum_macro_f1', 'maximum_binary_f1'], isolation_forest=dict(sample_size=250000, sampling='uniform without replacement from benign TRAIN IDs using NumPy seed 42', n_estimators=100, max_samples=256, contamination='auto', max_features=1.0, bootstrap=False, n_jobs=2, random_state=SEED), temporal_autoencoder='not run; optional secondary scope deferred', preprocessing='reuse baseline_v1 train-fitted preprocessing unchanged; it was fitted on all TRAIN classes, not benign-only')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(name, rows):
    csv_write(OUT / name, rows)

def guard(full=False):
    baseline_guard(full=full)
    require(read(CONFIG / 'protocol.json') == PROTOCOL, 'Anomaly protocol changed')
    for row in read(METRICS / 'frozen_artifacts.json'):
        require(sha256(row['path']) == row['sha256'], 'Protected artifact changed: ' + row['path'])

def validation_benign():
    return np.load(CACHE / 'benign_validation_features.npy', mmap_mode='r')

def train_batches(epoch=None):
    x, y = training_arrays()
    block = PROTOCOL['read_block_size']
    starts = np.arange(0, len(y), block)
    rng = np.random.default_rng(SEED if epoch is None else SEED + epoch - 1)
    if epoch is not None:
        starts = rng.permutation(starts)
    for start in starts:
        stop = min(int(start) + block, len(y))
        ids = np.flatnonzero(y[start:stop] == 0) + start
        if epoch is not None:
            ids = rng.permutation(ids)
        for offset in range(0, len(ids), PROTOCOL['batch_size']):
            positions = ids[offset:offset + PROTOCOL['batch_size']]
            if len(positions):
                yield np.array(x[positions], dtype=np.float32)

def partition_batches(part):
    digest = hashlib.sha256()
    offset = 0
    for x, y, metadata in iter_transformed(prep_config(), part, scaled=True):
        digest.update(np.ascontiguousarray(x, dtype='<f8').tobytes())
        offset += len(x)
        yield np.asarray(x, dtype=np.float32)
    with (BASE / 'preprocessing_application_audit.csv').open(newline='') as f:
        expected = next((r['transformed_float64_sha256'] for r in csv.DictReader(f) if r['partition'] == part))
    require(digest.hexdigest() == expected, 'Frozen transform changed: ' + part)
    require(offset == read(BASE / 'split_plan.json')['partitions'][part]['rows'], 'Partition incomplete')

def labels(part):
    require((BASE_CACHE / f'{part}_diagnostics.npz').exists(), 'Frozen diagnostic cache missing')
    return diagnostic_labels(part)
