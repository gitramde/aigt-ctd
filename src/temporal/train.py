"""Temporal encoder and validation-only development fitting."""
import argparse,math,time,random,platform
import numpy as np
import torch
from torch import nn
from .data import *
from src.phase4.metrics import binary_metrics
from src.phase4.train import MemoryMonitor

class TemporalTransformer(nn.Module):
    def __init__(self,config,features=80):
        super().__init__();d=config['d_model'];self.position_offset=config.get('position_offset',0)
        self.projection=nn.Linear(features,d)
        layer=nn.TransformerEncoderLayer(d,config['heads'],dim_feedforward=2*d,dropout=config['dropout'],
            activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config['layers'],norm=nn.LayerNorm(d),enable_nested_tensor=False)
        self.head=nn.Linear(d,1)
        position=torch.arange(64,dtype=torch.float32).unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2,dtype=torch.float32)*(-math.log(10000.0)/d))
        pe=torch.zeros(64,d);pe[:,0::2]=torch.sin(position*div);pe[:,1::2]=torch.cos(position*div)
        self.register_buffer('positional_encoding',pe)
        # Avoid identical initial matrices across cloned encoder layers.
        for parameter in self.parameters():
            if parameter.dim()>1: nn.init.xavier_uniform_(parameter)
    def forward(self,x):
        z=self.projection(x)+self.positional_encoding[self.position_offset:self.position_offset+x.shape[1]]
        return self.head(self.encoder(z)[:,-1]).squeeze(-1)

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(PROTOCOL['threads'])
    torch.use_deterministic_algorithms(True)

def config_for(name,length,architecture,control=False):
    return dict(name=name,seed=42,length=1 if control else length,context_length=length,
        position_offset=length-1 if control else 0,control=control,**architecture)

def path(name): return MODELS/f'{name}.pt'
def predict(model,x,targets,length):
    model.eval();p=np.empty(len(targets),dtype=np.float64);start=time.perf_counter();compute=0
    with torch.inference_mode():
        for positions,batch in sequence_batches(x,targets,length,PROTOCOL['batch_size']):
            begin=time.perf_counter();p[positions]=torch.sigmoid(model(torch.from_numpy(batch))).numpy();compute+=time.perf_counter()-begin
    require(np.isfinite(p).all(),'Nonfinite Transformer probabilities')
    elapsed=time.perf_counter()-start
    return p,dict(inference_seconds=elapsed,prediction_only_seconds=compute,latency_ms_per_1000=elapsed/len(targets)*1e6,evaluated_flows=len(targets))

def load(config):
    model=TemporalTransformer(config);model.load_state_dict(torch.load(path(config['name']),map_location='cpu',weights_only=True));return model

def fit(config):
    integrity();name=config['name'];destination=METRICS/f'{name}_training.json'
    config_path=CONFIG/f'{name}.json'
    if config_path.exists(): require(read(config_path)==config,'Development config changed')
    else: json_write(config_path,config)
    if destination.exists():
        prior=read(destination);require(prior['model_sha256']==sha256(path(name)),'Model artifact changed')
        require(prior['config']==config and prior['protocol_sha256']==sha256(CONFIG/'protocol.json'),'Configuration mismatch')
        print('FIT ALREADY COMPLETE',name,flush=True);return prior
    seed();start_stage=time.perf_counter()
    with MemoryMonitor() as memory:
        x=features('train');v=features('validation');tc=cohort('train');vc=cohort('validation')
        y=tc['target_binary'];ids=tc['partition_index'];positive=int(y.sum());negative=len(y)-positive
        require(positive>0 and negative>0,'Training targets need both classes')
        model=TemporalTransformer(config);optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,dtype=torch.float32))
        history=[];best=-1;best_epoch=None;bad=0;training_seconds=0;selection_seconds=0
        for epoch in range(1,PROTOCOL['max_epochs']+1):
            model.train();order=np.random.default_rng(42+epoch-1).permutation(len(ids));begin=time.perf_counter();loss_sum=0;seen=0
            for positions,batch in sequence_batches(x,ids,config['length'],PROTOCOL['batch_size'],order):
                optimizer.zero_grad(set_to_none=True)
                target=torch.from_numpy(y[positions].astype(np.float32));logits=model(torch.from_numpy(batch));loss=criterion(logits,target)
                require(bool(torch.isfinite(loss)),'Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
                loss_sum+=float(loss.detach())*len(positions);seen+=len(positions)
                if seen%25600==0: print(name,'epoch',epoch,f'{seen:,}/{len(ids):,} targets',flush=True)
            training_seconds+=time.perf_counter()-begin;require(seen==len(ids),'Incomplete training epoch')
            begin=time.perf_counter();prob,timing=predict(model,v,vc['partition_index'],config['length'])
            metrics=binary_metrics(vc['target_binary'],prob,include_auc=False);score=metrics['macro_f1']
            improved=score>best+PROTOCOL['min_improvement']
            if improved:
                best=score;best_epoch=epoch;bad=0;torch.save(model.state_dict(),path(name));np.save(METRICS/f'{name}_validation.npy',prob)
            else: bad+=1
            selection_seconds+=time.perf_counter()-begin
            history.append(dict(epoch=epoch,training_loss=loss_sum/seen,selected=improved,**metrics,**timing))
            json_write(METRICS/f'{name}_history.json',history)
            print(name,'epoch',epoch,'validation macro-F1',score,'selected',improved,flush=True)
            if bad>=PROTOCOL['patience']: break
        result=dict(status='complete',config=config,seed=42,training_seconds=training_seconds,selection_seconds=selection_seconds,
            training_stage_seconds=time.perf_counter()-start_stage,selected_epoch=best_epoch,selected_validation_macro_f1=best,
            executed_epochs=epoch,training_targets=len(ids),training_target_positive=positive,training_target_negative=negative,
            positive_class_weight=negative/positive,parameters=sum(p.numel() for p in model.parameters()),
            model_sha256=sha256(path(name)),model_size_bytes=path(name).stat().st_size,protocol_sha256=sha256(CONFIG/'protocol.json'),
            validation_prediction_sha256=sha256(METRICS/f'{name}_validation.npy'),
            environment=dict(torch=torch.__version__,python=platform.python_version(),numpy=np.__version__,device='cpu',threads=PROTOCOL['threads']))
    result['peak_memory_bytes']=memory.peak;integrity();json_write(destination,result)
    print('TRAIN COMPLETE',name,'best epoch',best_epoch,'macro-F1',best,flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();fit(read(args.config))
