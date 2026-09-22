"""Metrics accept unbounded anomaly scores, not artificial probabilities."""
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,average_precision_score
from src.phase4.metrics import binary_metrics
from src.phase4a.run import threshold_candidates

def check_scores(scores):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all(): raise ValueError('Invalid anomaly scores')
    return scores

def benign_threshold(scores,cap):
    values=np.sort(check_scores(scores));allowed=int(np.floor(cap*len(values)))
    threshold=float(np.nextafter(values[len(values)-allowed-1],np.inf))
    fp=int((values>=threshold).sum())
    assert fp<=allowed
    return dict(threshold=threshold,benign_validation_N=len(values),allowed_false_positives=allowed,
        observed_false_positives=fp,observed_validation_fpr=fp/len(values),target_fpr=cap)

def label_thresholds(y,scores):
    scores=check_scores(scores);t,tp,fp,fn,tn,f1,macro=threshold_candidates(y,scores)
    return [dict(criterion=name,threshold=float(t[int(np.argmax(metric))])) for name,metric in [('maximum_macro_f1',macro),('maximum_binary_f1',f1)]]

def metrics(y,scores,threshold,ranking=True):
    scores=check_scores(scores)
    result=binary_metrics(y,(scores>=threshold).astype(float),.5,include_auc=False);result['threshold']=threshold
    if ranking:
        precision,recall,_=precision_recall_curve(y,scores)
        result.update(roc_auc=float(roc_auc_score(y,scores)),pr_auc=float(auc(recall,precision)),average_precision=float(average_precision_score(y,scores)))
    return result
