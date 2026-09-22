"""Binary metrics and attack-family diagnostics computed from saved predictions."""
import numpy as np
from sklearn.metrics import average_precision_score,auc,precision_recall_curve,roc_auc_score


def binary_metrics(y,score,threshold=0.5,include_auc=True):
    y=np.asarray(y,dtype=np.uint8); score=np.asarray(score,dtype=np.float64)
    if y.shape!=score.shape or not np.isfinite(score).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid binary evaluation inputs')
    if ((score<0)|(score>1)).any():
        raise ValueError('Scores must be malicious-class probabilities')
    pred=score>=threshold
    tn=int(np.count_nonzero((y==0)&~pred)); fp=int(np.count_nonzero((y==0)&pred))
    fn=int(np.count_nonzero((y==1)&~pred)); tp=int(np.count_nonzero((y==1)&pred))
    def div(a,b): return a/b if b else 0.0
    f1=div(2*tp,2*tp+fp+fn); benign_f1=div(2*tn,2*tn+fp+fn)
    result=dict(records=len(y),threshold=threshold,accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),
                recall=div(tp,tp+fn),f1=f1,macro_f1=(f1+benign_f1)/2,
                false_positive_rate=div(fp,fp+tn),false_negative_rate=div(fn,fn+tp),
                benign_recall_specificity=div(tn,tn+fp),malicious_recall=div(tp,tp+fn),
                false_positives=fp,false_negatives=fn,true_negatives=tn,true_positives=tp)
    if include_auc:
        if len(np.unique(y))!=2:
            raise ValueError('Both binary classes are required for ROC/PR evaluation')
        precision,recall,_=precision_recall_curve(y,score)
        result.update(roc_auc=float(roc_auc_score(y,score)),pr_auc=float(auc(recall,precision)),
                      average_precision=float(average_precision_score(y,score)),
                      pr_auc_definition='trapezoidal precision-recall area; average_precision reported separately')
    return result


def family_diagnostics(family_codes,codebook,score,train_counts,threshold=0.5):
    detected=np.asarray(score)>=threshold
    rows=[]
    for code,family in enumerate(codebook):
        if family=='Benign':
            continue
        selected=np.asarray(family_codes)==code
        total=int(selected.sum())
        hits=int(np.count_nonzero(selected&detected))
        rows.append(dict(attack_family=family,training_status='KNOWN' if train_counts.get(family,0)>0 else 'UNSEEN',
                         present_in_partition=total>0,total_records=total,detected_as_malicious=hits,
                         missed=total-hits,binary_detection_recall=hits/total if total else None))
    return rows
