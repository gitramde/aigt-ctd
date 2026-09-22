"""Feb-20-specific conventional controls on the exact graph target cohort."""
import argparse,time,platform
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from .common import *
from .data import GraphData
from src.phase4.train import MemoryMonitor
from src.phase4.metrics import binary_metrics

NAMES=('logistic_regression','random_forest','xgboost')
SETTINGS={
    'logistic_regression':dict(loss='log_loss',alpha=.0001,random_state=42,average=True),
    'random_forest':dict(n_estimators=100,max_depth=16,max_features='sqrt',min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=2,random_state=42),
    'xgboost':dict(objective='binary:logistic',eval_metric='logloss',max_depth=5,eta=.05,tree_method='hist',nthread=2,seed=42),
}

def predict(model,name,x):
    return model.predict(xgb.DMatrix(x)) if name=='xgboost' else model.predict_proba(x)[:,1]

def fit(name):
    guard();destination=METRICS/f'{name}_training.json'
    config=dict(name=name,kind='conventional',seed=42,parameters=SETTINGS[name],
        maximum_passes=4 if name=='logistic_regression' else None,maximum_boost_rounds=150 if name=='xgboost' else None,
        early_stopping_rounds=20 if name=='xgboost' else None,
        weighting='training-target class balance; XGBoost negative/positive scale_pos_weight')
    config_path=CONFIG/f'{name}.json'
    if config_path.exists(): require(read(config_path)==config,'Baseline configuration changed')
    else: json_write(config_path,config)
    if destination.exists():
        r=read(destination);require(sha256(r['model_path'])==r['model_sha256'],'Baseline model changed');return
    with MemoryMonitor() as memory:
        data=GraphData();x=np.asarray(data.x[data.part==0]);y=data.y[data.part==0]
        vx=np.asarray(data.x[data.part==1]);vy=data.y[data.part==1]
        positive=int(y.sum());negative=len(y)-positive
        sample=np.where(y==1,len(y)/(2*positive),len(y)/(2*negative)).astype(x.dtype);history=[]
        start=time.perf_counter();path=MODELS/f'{name}.joblib';validation_seconds=0.
        if name=='logistic_regression':
            model=SGDClassifier(**SETTINGS[name])
            best=-1;bad=0
            for epoch in range(1,5):
                order=np.random.default_rng(42+epoch-1).permutation(len(y))
                for offset in range(0,len(y),4096):
                    idx=order[offset:offset+4096];model.partial_fit(x[idx],y[idx],classes=np.array([0,1]),sample_weight=sample[idx])
                begin=time.perf_counter();p=predict(model,name,vx);m=binary_metrics(vy,p,include_auc=False);validation_seconds+=time.perf_counter()-begin
                score=m['macro_f1'];improved=score>best+PROTOCOL['min_improvement'];history.append(dict(epoch=epoch,selected=improved,**m))
                if improved: best=score;selected_epoch=epoch;bad=0;joblib.dump(model,path);np.save(METRICS/f'{name}_validation.npy',p)
                else: bad+=1
                if bad>=2: break
        elif name=='random_forest':
            model=RandomForestClassifier(**SETTINGS[name])
            model.fit(x,y);joblib.dump(model,path);begin=time.perf_counter();p=predict(model,name,vx)
            best=binary_metrics(vy,p,include_auc=False)['macro_f1'];validation_seconds=time.perf_counter()-begin;selected_epoch=1
            np.save(METRICS/f'{name}_validation.npy',p)
        else:
            evaluation={}
            model=xgb.train(dict(**SETTINGS[name],scale_pos_weight=negative/positive),xgb.DMatrix(x,label=y),num_boost_round=150,
                evals=[(xgb.DMatrix(vx,label=vy),'validation')],early_stopping_rounds=20,evals_result=evaluation,verbose_eval=False)
            selected_epoch=model.best_iteration+1;model=model[:selected_epoch];path=MODELS/f'{name}.ubj';model.save_model(path)
            begin=time.perf_counter();p=predict(model,name,vx);best=binary_metrics(vy,p,include_auc=False)['macro_f1']
            validation_seconds=time.perf_counter()-begin;np.save(METRICS/f'{name}_validation.npy',p)
            history=evaluation
        elapsed=time.perf_counter()-start
        json_write(METRICS/f'{name}_history.json',history)
        record=dict(status='complete',config=config,model_path=str(path),model_sha256=sha256(path),
            model_size_bytes=path.stat().st_size,selected_epoch=selected_epoch,selected_validation_macro_f1=best,
            training_seconds=elapsed-validation_seconds,validation_selection_seconds=validation_seconds,stage_seconds=elapsed,
            training_time_note='XGBoost training includes internal validation early-stopping evaluation',
            validation_prediction_sha256=sha256(METRICS/f'{name}_validation.npy'),protocol_sha256=sha256(CONFIG/'protocol.json'),
            preprocessing_sha256=sha256(CONFIG/'preprocessing.json'),training_targets=len(y),
            software=dict(sklearn=sklearn.__version__,xgboost=xgb.__version__,python=platform.python_version()))
    record['peak_memory_bytes']=memory.peak;json_write(destination,record);print('CONTROL COMPLETE',name,best,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=NAMES,required=True);a=p.parse_args();fit(a.name)
