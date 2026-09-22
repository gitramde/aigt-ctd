SEED = 42
"""Frozen GAT embeddings and bounded temporal-head development fits."""
import argparse, time, random
from src.graph_temporal.prepare import *
import torch
from torch import nn
from torch.nn import functional as F
from src.graph.data import GraphData
from src.graph.model import EdgeModel, tensors
from src.phase4.metrics import binary_metrics
from src.phase4.train import MemoryMonitor
from src.graph_temporal.model import TemporalHead, batch

def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def embed():
    require(read(OUT / 'metrics/coverage_gate.json')['passed'], 'Coverage gate failed: training forbidden')
    seed()
    config = read(GRAPH / 'selected_model.json')['config']
    require(config['name'] == NAME and config['hidden'] == 128 and (config['dropout'] == 0.2), 'Frozen graph selection mismatch')
    path = OUT / 'metrics/representations.npy'
    manifest = OUT / 'metrics/representation_manifest.json'
    if manifest.exists():
        require(sha256(path) == read(manifest)['sha256'], 'Embedding cache changed')
        return
    with MemoryMonitor() as memory:
        data = GraphData()
        info = read(GRAPH / 'metrics/data_manifest.json')
        model = EdgeModel(config, len(info['edge_feature_names']), len(info['history_feature_names']))
        model.load_state_dict(torch.load(GRAPH / f'models/{NAME}.pt', weights_only=True, map_location='cpu'))
        model.eval()
        model.requires_grad_(False)
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(data.ids), 256 + data.x.shape[1]))
        timings = []
        for part in PARTS:
            start = time.perf_counter()
            prep = 0.0
            compute = 0.0
            count = 0
            with torch.inference_mode():
                for g in data.groups:
                    if g['partition'] != part:
                        continue
                    begin = time.perf_counter()
                    snap = data.snapshot(g, 1)
                    b = tensors(snap)
                    prep += time.perf_counter() - begin
                    begin = time.perf_counter()
                    z = model.projection(b['nodes'])
                    for layer, norm in zip(model.layers, model.norms):
                        z = norm(z + model.dropout(F.elu(layer(z, b['edges'], b['edge_features']))))
                    representation = torch.cat([z[b['target_src']], z[b['target_dst']], b['x']], dim=1)
                    require(bool(torch.isfinite(representation).all()), 'Nonfinite graph representation')
                    output[snap['positions']] = representation.numpy()
                    compute += time.perf_counter() - begin
                    count += len(snap['positions'])
            timings.append(dict(partition=part, records=count, graph_preparation_seconds=prep, graph_encoder_seconds=compute, representation_seconds=time.perf_counter() - start))
            print('Frozen graph representations', part, count, flush=True)
        output.flush()
        del output
        np.save(OUT / 'metrics/targets_y.npy', data.y)
        np.save(OUT / 'metrics/target_partitions.npy', data.part)
    json_write(manifest, dict(path=str(path), sha256=sha256(path), encoder_sha256=sha256(GRAPH / f'models/{NAME}.pt'), timings=timings, peak_memory_bytes=memory.peak, labels_excluded_from_representations=True))

def predict(model, representations, history, positions, length):
    start = time.perf_counter()
    outputs = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(positions), PROTOCOL['batch_size']):
            ids = positions[offset:offset + PROTOCOL['batch_size']]
            x, mask = batch(representations, history, ids, length)
            outputs.append(torch.sigmoid(model(x, mask)).numpy().astype(np.float64))
    return (np.concatenate(outputs), time.perf_counter() - start)

def fit(length):
    require(read(OUT / 'metrics/coverage_gate.json')['passed'], 'Coverage gate failed')
    seed()
    name = f'gat_transformer_L{length}'
    dest = OUT / f'metrics/{name}_training.json'
    modelpath = OUT / f'models/{name}.pt'
    if dest.exists():
        require(read(dest)['model_sha256'] == sha256(modelpath), 'Checkpoint changed')
        print('REUSE', name, flush=True)
        return
    with MemoryMonitor() as memory:
        representations = np.load(OUT / 'metrics/representations.npy', mmap_mode='r')
        history = np.load(OUT / 'metrics/history_indices.npy')
        y = np.load(OUT / 'metrics/targets_y.npy')
        part = np.load(OUT / 'metrics/target_partitions.npy')
        train = np.flatnonzero(part == 0)
        validation = np.flatnonzero(part == 1)
        model = TemporalHead(representations.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        weight = float((y[train] == 0).sum() / (y[train] == 1).sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
        best = -1.0
        bad = 0
        records = []
        training_seconds = 0.0
        validation_seconds = 0.0
        for epoch in range(1, 5):
            model.train()
            start = time.perf_counter()
            order = np.random.default_rng(SEED + epoch - 1).permutation(train)
            loss_sum = 0.0
            seen = 0
            for offset in range(0, len(order), PROTOCOL['batch_size']):
                ids = order[offset:offset + PROTOCOL['batch_size']]
                x, mask = batch(representations, history, ids, length)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x, mask), torch.tensor(y[ids], dtype=torch.float32))
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                seen += len(ids)
                loss_sum += float(loss.detach()) * len(ids)
            require(seen == len(train), 'Incomplete training epoch')
            training_seconds += time.perf_counter() - start
            probabilities, elapsed = predict(model, representations, history, validation, length)
            validation_seconds += elapsed
            metrics = binary_metrics(y[validation], probabilities, 0.5, include_auc=False)
            improved = metrics['macro_f1'] > best + 1e-05
            if improved:
                best = metrics['macro_f1']
                best_epoch = epoch
                bad = 0
                torch.save(model.state_dict(), modelpath)
                np.save(OUT / f'metrics/{name}_validation.npy', probabilities)
            else:
                bad += 1
            records.append(dict(epoch=epoch, loss=loss_sum / seen, selected=improved, **metrics))
            json_write(OUT / f'metrics/{name}_history.json', records)
            print(name, 'epoch', epoch, 'validation macro-F1', best, 'train seconds', round(training_seconds, 1), flush=True)
            if bad >= 2:
                break
    json_write(dest, dict(status='complete', model=name, length=length, seed=SEED, selected_epoch=best_epoch, executed_epochs=epoch, selected_validation_macro_f1=best, training_seconds=training_seconds, validation_selection_seconds=validation_seconds, peak_memory_bytes=memory.peak, trainable_parameters=sum((p.numel() for p in model.parameters())), positive_class_weight=weight, training_targets=len(train), model_sha256=sha256(modelpath), model_size_bytes=modelpath.stat().st_size, validation_prediction_sha256=sha256(OUT / f'metrics/{name}_validation.npy'), protocol_sha256=sha256(OUT / 'configs/protocol.json'), representation_manifest_sha256=sha256(OUT / 'metrics/representation_manifest.json')))
