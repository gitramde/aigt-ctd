"""Validation-only development fitting; separate processes bound peak memory."""
import argparse
import random
import time
import platform
import numpy as np
import torch
from torch import nn
from .common import *
from .data import GraphData
from .model import EdgeModel,tensors
from src.phase4.metrics import binary_metrics
from src.phase4.train import MemoryMonitor


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(PROTOCOL['threads'])
    torch.use_deterministic_algorithms(True)


def make_model(config):
    manifest=read(METRICS/'data_manifest.json')
    return EdgeModel(config,len(manifest['edge_feature_names']),len(manifest['history_feature_names']))


def predict(model,data,part,window):
    model.eval();results=[];positions=[];start=time.perf_counter();graph_seconds=0.;compute_seconds=0.
    with torch.inference_mode():
        for g in data.groups:
            if g['partition']!=part: continue
            begin=time.perf_counter()
            if model.kind=='edge_mlp':
                pos=np.arange(g['start'],g['end']);b={'x':torch.from_numpy(np.array(data.x[pos]))}
            else:
                snap=data.snapshot(g,window);pos=snap['positions'];b=tensors(snap)
            graph_seconds+=time.perf_counter()-begin;begin=time.perf_counter()
            results.append(torch.sigmoid(model(b)).numpy().astype(np.float64));positions.append(pos)
            compute_seconds+=time.perf_counter()-begin
    p=np.concatenate(results);positions=np.concatenate(positions)
    require(np.array_equal(positions,np.flatnonzero(data.part==PARTS.index(part))),'Evaluation order mismatch')
    elapsed=time.perf_counter()-start
    require(np.isfinite(p).all(),'Nonfinite predictions')
    return p,dict(inference_seconds=elapsed,graph_preparation_seconds=graph_seconds,prediction_only_seconds=compute_seconds,
        latency_ms_per_1000=elapsed/len(p)*1e6,evaluated_edges=len(p))


def fit(config):
    guard();name=config['name'];dest=METRICS/f'{name}_training.json';model_path=MODELS/f'{name}.pt'
    if dest.exists():
        old=read(dest);require(old['config']==config and old['model_sha256']==sha256(model_path),'Training provenance changed')
        require(old['protocol_sha256']==sha256(CONFIG/'protocol.json'),'Training protocol changed')
        print('REUSE',name,flush=True);return
    seed();begin_stage=time.perf_counter()
    with MemoryMonitor() as memory:
        data=GraphData();model=make_model(config)
        y=data.y[data.part==0];positive=int(y.sum());negative=len(y)-positive
        require(positive>0 and negative>0,'Training cohort needs both classes')
        optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,dtype=torch.float32))
        groups=[g for g in data.groups if g['partition']=='train'];history=[];best=-1.;bad=0;training_seconds=0.;validation_seconds=0.
        for epoch in range(1,PROTOCOL['max_epochs']+1):
            model.train();begin=time.perf_counter();loss_sum=0.;seen=0
            for index in np.random.default_rng(42+epoch-1).permutation(len(groups)):
                g=groups[index]
                if model.kind=='edge_mlp':
                    positions=np.arange(g['start'],g['end']);b={'x':torch.from_numpy(np.array(data.x[positions])),'y':torch.tensor(data.y[positions],dtype=torch.float32)}
                else: b=tensors(data.snapshot(g,config['window_minutes']))
                optimizer.zero_grad(set_to_none=True);loss=criterion(model(b),b['y'])
                require(bool(torch.isfinite(loss)),'Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
                count=len(b['y']);loss_sum+=float(loss.detach())*count;seen+=count
            require(seen==len(y),'Training target mismatch');training_seconds+=time.perf_counter()-begin
            begin=time.perf_counter();p,timing=predict(model,data,'validation',config['window_minutes'])
            metrics=binary_metrics(data.y[data.part==1],p,include_auc=False);score=metrics['macro_f1']
            improved=score>best+PROTOCOL['min_improvement']
            if improved:
                best=score;best_epoch=epoch;bad=0;torch.save(model.state_dict(),model_path);np.save(METRICS/f'{name}_validation.npy',p)
            else: bad+=1
            validation_seconds+=time.perf_counter()-begin
            history.append(dict(epoch=epoch,loss=loss_sum/seen,selected=improved,**metrics,**timing))
            json_write(METRICS/f'{name}_history.json',history)
            print(name,'epoch',epoch,'macro-F1',score,'selected',improved,'train seconds',round(training_seconds,1),flush=True)
            if bad>=PROTOCOL['patience']: break
        record=dict(status='complete',config=config,selected_epoch=best_epoch,selected_validation_macro_f1=best,executed_epochs=epoch,
            model_sha256=sha256(model_path),model_size_bytes=model_path.stat().st_size,parameters=sum(p.numel() for p in model.parameters()),
            validation_prediction_sha256=sha256(METRICS/f'{name}_validation.npy'),training_seconds=training_seconds,validation_selection_seconds=validation_seconds,
            stage_seconds=time.perf_counter()-begin_stage,positive_class_weight=negative/positive,training_targets=len(y),
            protocol_sha256=sha256(CONFIG/'protocol.json'),preprocessing_sha256=sha256(CONFIG/'preprocessing.json'),
            software=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),device='cpu',threads=PROTOCOL['threads']))
    record['peak_memory_bytes']=memory.peak;json_write(dest,record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);a=parser.parse_args();fit(read(a.config))
