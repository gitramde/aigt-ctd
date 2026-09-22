"""Frozen calibration and metric definitions, including raw anomaly errors."""
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,average_precision_score,auc
from src.phase4a.run import select_thresholds

def thresholds(y,p,group):
    if group=='anomaly':
        b=np.sort(p[y==0]);n=len(b)
        return [dict(criterion=f'benign_fpr_at_most_{cap:g}',threshold=float(np.nextafter(b[n-int(np.floor(cap*n))-1],np.inf))) for cap in (.05,.01,.005,.001)]
    allowed={'fpr_at_most_0.01'}
    if group in ('graph','graph_temporal','temporal'):allowed.add('maximum_macro_f1')
    if group=='temporal':allowed.update(('fpr_at_most_0.005','fpr_at_most_0.001'))
    return [dict(criterion='fixed_0_5',threshold=.5)]+[dict(criterion=r['criterion'],threshold=r['threshold']) for r in select_thresholds(y,p) if r['criterion'] in allowed]

def ranking(y,p):
    if len(np.unique(y))!=2:return dict(roc_auc=None,pr_auc=None,average_precision=None,ranking_reason='Both classes required')
    precision,recall,_=precision_recall_curve(y,p)
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(auc(recall,precision)),average_precision=float(average_precision_score(y,p)))

def counts(y,p,t):
    d=p>=t
    tn=int(((y==0)&~d).sum());fp=int(((y==0)&d).sum());fn=int(((y==1)&~d).sum());tp=int(((y==1)&d).sum())
    div=lambda a,b:a/b if b else 0.
    f1=div(2*tp,2*tp+fp+fn)
    return dict(N=len(y),TN=tn,FP=fp,FN=fn,TP=tp,accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=f1,
                macro_f1=(f1+div(2*tn,2*tn+fp+fn))/2,false_positive_rate=div(fp,fp+tn),false_negative_rate=div(fn,fn+tp))
