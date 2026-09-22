SEED = 42
"""Seed-42 development fitting; test features/results are inaccessible to selection."""
import argparse
import gc
import json
import os
import platform
import threading
import time
from pathlib import Path
import joblib
import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
import sklearn
import xgboost as xgb
from src.phase4.data import BASE, CACHE, METRICS, MODELS, MODEL_NAMES, batches, class_weights, diagnostic_labels, guard, settings, training_arrays
from src.phase4.metrics import binary_metrics
from src.baseline.common import json_write, require, sha256

class MemoryMonitor:

    def __init__(self):
        self.peak = 0
        self.stop_event = threading.Event()
        self.process = psutil.Process()

    def sample(self):
        info = self.process.memory_info()
        self.peak = max(self.peak, info.rss, getattr(info, 'peak_wset', 0))

    def __enter__(self):
        self.sample()

        def watch():
            while not self.stop_event.wait(0.05):
                self.sample()
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop_event.set()
        self.thread.join()

def model_path(name):
    return MODELS / f"{name}_seed42.{('ubj' if name == 'xgboost' else 'joblib')}"

def save_model(model, name):
    path = model_path(name)
    if name == 'xgboost':
        model.save_model(path)
    else:
        joblib.dump(model, path, compress=3)

def load_model(name):
    if name == 'xgboost':
        model = xgb.Booster()
        model.load_model(model_path(name))
        return model
    return joblib.load(model_path(name))

def predict_partition(model, name, partition):
    y, _, _ = diagnostic_labels(partition)
    score = np.empty(len(y), dtype=np.float64)
    offset = 0
    predict_seconds = 0.0
    begin = time.perf_counter()
    for features, target in batches(partition):
        require(np.array_equal(y[offset:offset + len(target)], target), 'Evaluation labels/membership misaligned')
        start = time.perf_counter()
        p = model.inplace_predict(features) if name == 'xgboost' else model.predict_proba(features)[:, 1]
        predict_seconds += time.perf_counter() - start
        score[offset:offset + len(p)] = p
        offset += len(p)
    require(offset == len(y), 'Incomplete inference')
    end_to_end = time.perf_counter() - begin
    require(np.isfinite(score).all(), 'Nonfinite probability output')
    return (score, dict(inference_seconds=end_to_end, predict_only_seconds=predict_seconds, inference_latency_ms_per_1000=end_to_end / len(y) * 1000000, predict_only_latency_ms_per_1000=predict_seconds / len(y) * 1000000))

class Selection:

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.best = -float('inf')
        self.bad = 0
        self.history = []
        self.best_step = None
        self.overhead_seconds = 0.0
        self.y = diagnostic_labels('validation')[0]

    def check(self, model, step):
        begin = time.perf_counter()
        score, timing = predict_partition(model, self.name, 'validation')
        metrics = binary_metrics(self.y, score, self.cfg['threshold'], include_auc=False)
        improved = metrics['macro_f1'] > self.best + self.cfg['minimum_improvement']
        if improved:
            self.best = metrics['macro_f1']
            self.best_step = step
            self.bad = 0
            save_model(model, self.name)
        else:
            self.bad += 1
        self.history.append(dict(step=step, selected_at_this_step=improved, **metrics, **timing))
        json_write(METRICS / f'{self.name}_selection.json', dict(status='running', model=self.name, seed=self.cfg['seed'], selection_partition='validation', selection_metric='macro_f1', threshold=self.cfg['threshold'], best_step=self.best_step, history=self.history))
        elapsed = time.perf_counter() - begin
        self.overhead_seconds += elapsed
        print(f"{self.name}: step {step}, validation macro-F1={metrics['macro_f1']:.6f}, selected={improved}", flush=True)
        return self.bad >= self.cfg['early_stopping_patience']

class CacheIterator(xgb.DataIter):

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.offset = 0
        super().__init__(release_data=True)

    def reset(self):
        self.offset = 0

    def next(self, input_data):
        if self.offset >= len(self.y):
            return False
        end = min(self.offset + self.batch_size, len(self.y))
        input_data(data=np.array(self.x[self.offset:end], dtype=np.float32, copy=True), label=np.asarray(self.y[self.offset:end], dtype=np.float32))
        self.offset = end
        return True

def train_incremental(name, cfg, x, y, selection):
    spec = cfg[name]
    weights = class_weights()
    n = len(y)
    if name == 'logistic_regression':
        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=spec['alpha'], learning_rate='constant', eta0=spec['eta0'], average=spec['average'], class_weight=weights, shuffle=True, random_state=cfg['seed'])
    else:
        model = MLPClassifier(hidden_layer_sizes=tuple(spec['hidden_layer_sizes']), activation='relu', solver='adam', alpha=spec['alpha'], batch_size=spec['minibatch_size'], learning_rate_init=spec['learning_rate_init'], shuffle=True, early_stopping=False, random_state=cfg['seed'])
    fit_seconds = 0.0
    for epoch in range(1, spec['max_epochs'] + 1):
        begin = time.perf_counter()
        order = np.random.default_rng(cfg['seed'] + epoch - 1).permutation(n)
        seen = 0
        for start in range(0, n, cfg['batch_size']):
            indices = order[start:start + cfg['batch_size']]
            xb = np.ascontiguousarray(x[indices])
            yb = np.asarray(y[indices])
            kwargs = dict(classes=np.array([0, 1]))
            if name == 'mlp':
                kwargs['sample_weight'] = np.where(yb == 1, weights[1], weights[0]).astype(np.float32)
            model.partial_fit(xb, yb, **kwargs)
            seen += len(indices)
            if seen % 2000000 == 0:
                print(f'{name}: epoch {epoch}, {seen:,}/{n:,} training rows', flush=True)
        require(seen == n, 'Epoch did not visit every training row')
        fit_seconds += time.perf_counter() - begin
        del order
        if selection.check(model, epoch):
            break
    return dict(training_seconds=fit_seconds, executed_steps=epoch, training_row_visits=n * epoch, class_weight_strategy='global inverse-frequency training class weights' if name == 'logistic_regression' else 'global inverse-frequency training sample weights')

def train_forest(cfg, x, y, selection):
    spec = cfg['random_forest']
    model = RandomForestClassifier(n_estimators=spec['tree_checkpoints'][0], max_depth=spec['max_depth'], min_samples_leaf=spec['min_samples_leaf'], max_features=spec['max_features'], bootstrap=True, max_samples=min(len(y), spec['max_samples_per_tree']), class_weight=class_weights(), n_jobs=spec['n_jobs'], warm_start=True, random_state=cfg['seed'])
    fit_seconds = 0.0
    for trees in spec['tree_checkpoints']:
        model.set_params(n_estimators=trees)
        begin = time.perf_counter()
        model.fit(x, y)
        fit_seconds += time.perf_counter() - begin
        if selection.check(model, trees):
            break
    return dict(training_seconds=fit_seconds, executed_steps=trees, eligible_training_rows=len(y), bootstrap_draws_per_tree=model.max_samples, class_weight_strategy='global inverse-frequency training class weights')

def train_xgboost(cfg, x, y, selection):
    spec = cfg['xgboost']
    weights = class_weights()
    begin = time.perf_counter()
    matrix = xgb.QuantileDMatrix(CacheIterator(x, y, cfg['batch_size']), max_bin=spec['max_bin'], nthread=cfg['threads'])
    matrix_seconds = time.perf_counter() - begin
    params = {k: spec[k] for k in ('max_depth', 'max_bin', 'eta', 'subsample', 'colsample_bytree', 'lambda')}
    params.update(objective='binary:logistic', tree_method='hist', device='cpu', seed=cfg['seed'], seed_per_iteration=True, nthread=cfg['threads'], scale_pos_weight=weights[1] / weights[0])

    class Callback(xgb.callback.TrainingCallback):

        def after_iteration(self, model, epoch, evals_log):
            if epoch + 1 in spec['round_checkpoints']:
                return selection.check(model, epoch + 1)
            return False
    begin = time.perf_counter()
    model = xgb.train(params, matrix, num_boost_round=max(spec['round_checkpoints']), callbacks=[Callback()])
    elapsed = time.perf_counter() - begin
    return dict(training_seconds=matrix_seconds + elapsed - selection.overhead_seconds, quantile_matrix_seconds=matrix_seconds, executed_steps=model.num_boosted_rounds(), class_weight_strategy='scale_pos_weight from training negative/positive counts', scale_pos_weight=params['scale_pos_weight'])
