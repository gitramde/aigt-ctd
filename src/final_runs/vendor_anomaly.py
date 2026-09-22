SEED = 42
import argparse, time, platform, random
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
import sklearn
from src.final_runs.vendor_anomaly_data import *
from src.phase4.train import MemoryMonitor

class Autoencoder(nn.Module):

    def __init__(self, widths, features=80):
        super().__init__()
        dims = [features] + widths + [features]
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(PROTOCOL['threads'])
    torch.use_deterministic_algorithms(True)

def errors(model, x):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), PROTOCOL['batch_size']):
            batch = torch.from_numpy(np.array(x[start:start + PROTOCOL['batch_size']], dtype=np.float32))
            delta = model(batch).double() - batch.double()
            chunks.append(delta.square().mean(1).numpy())
    values = np.concatenate(chunks)
    require(np.isfinite(values).all(), 'Nonfinite reconstruction score')
    return values

def fit(name):
    guard()
    destination = METRICS / f'{name}_training.json'
    if destination.exists():
        old = read(destination)
        require(sha256(old['model_path']) == old['model_sha256'], 'Model changed')
        require(old['protocol_sha256'] == sha256(CONFIG / 'protocol.json'), 'Protocol changed')
        return
    seed()
    stage = time.perf_counter()
    config = dict(name=name, seed=SEED, hidden_layers=PROTOCOL['architectures'].get(name), isolation_forest=PROTOCOL['isolation_forest'] if name == 'isolation_forest' else None, gradient_clip_norm=10.0 if name != 'isolation_forest' else None)
    json_write(CONFIG / f'{name}.json', config)
    with MemoryMonitor() as memory:
        validation = validation_benign()
        selection_seconds = 0.0
        training_seconds = 0.0
        if name == 'isolation_forest':
            ids = np.load(CACHE / 'isolation_train_ids.npy')
            x, y = training_arrays()
            require((y[ids] == 0).all(), 'Malicious IF fitting row')
            settings = {k: v for k, v in PROTOCOL['isolation_forest'].items() if k not in ('sample_size', 'sampling')}
            model = IsolationForest(**settings)
            start = time.perf_counter()
            model.fit(np.asarray(x[ids]))
            training_seconds = time.perf_counter() - start
            path = MODELS / f'{name}.joblib'
            joblib.dump(model, path)
            start = time.perf_counter()
            score = -model.score_samples(validation)
            selection_seconds = time.perf_counter() - start
            np.save(METRICS / f'{name}_benign_validation.npy', score)
            extra = dict(training_rows=len(ids), unique_rows_used_by_trees=int(len(np.unique(np.concatenate(model.estimators_samples_)))), selected_epoch=None, selected_benign_validation_mse=None, software=dict(sklearn=sklearn.__version__, numpy=np.__version__, python=platform.python_version()))
        else:
            model = Autoencoder(config['hidden_layers'], validation.shape[1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'], weight_decay=PROTOCOL['weight_decay'])
            best = float('inf')
            bad = 0
            history = []
            expected = read(OUT / 'anomaly_training_manifest.json')['benign_train_rows']
            path = MODELS / f'{name}.pt'
            for epoch in range(1, PROTOCOL['max_epochs'] + 1):
                model.train()
                start = time.perf_counter()
                total = 0.0
                seen = 0
                last_progress = 0
                for values in train_batches(epoch):
                    batch = torch.from_numpy(values)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(batch), batch)
                    require(bool(torch.isfinite(loss)), 'Nonfinite AE loss; refusing to alter frozen preprocessing')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                    optimizer.step()
                    seen += len(values)
                    total += float(loss.detach()) * len(values)
                    if seen - last_progress >= 2000000:
                        print(name, 'epoch', epoch, f'{seen:,}/{expected:,} benign rows', flush=True)
                        last_progress = seen
                require(seen == expected, 'Autoencoder did not visit every benign training row')
                training_seconds += time.perf_counter() - start
                start = time.perf_counter()
                score = errors(model, validation)
                mean = float(score.mean())
                selection_seconds += time.perf_counter() - start
                improved = mean < best * (1 - PROTOCOL['min_relative_improvement'])
                if improved:
                    best = mean
                    selected_epoch = epoch
                    bad = 0
                    torch.save(model.state_dict(), path)
                    np.save(METRICS / f'{name}_benign_validation.npy', score)
                else:
                    bad += 1
                history.append(dict(epoch=epoch, training_rows=seen, training_mse=total / seen, benign_validation_mse=mean, selected=improved))
                json_write(METRICS / f'{name}_history.json', history)
                print(name, 'epoch', epoch, 'benign validation MSE', mean, 'selected', improved, flush=True)
                if bad >= PROTOCOL['patience']:
                    break
            extra = dict(training_rows=expected, selected_epoch=selected_epoch, selected_benign_validation_mse=best, executed_epochs=epoch, parameters=sum((p.numel() for p in model.parameters())), software=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(), threads=PROTOCOL['threads']))
        result = dict(status='complete', config=config, model_path=str(path), model_sha256=sha256(path), model_size_bytes=path.stat().st_size, training_seconds=training_seconds, benign_validation_selection_seconds=selection_seconds, stage_seconds=time.perf_counter() - stage, benign_validation_score_sha256=sha256(METRICS / f'{name}_benign_validation.npy'), protocol_sha256=sha256(CONFIG / 'protocol.json'), **extra)
    result['peak_memory_bytes'] = memory.peak
    json_write(destination, result)
