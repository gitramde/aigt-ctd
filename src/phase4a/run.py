"""Phase 4A: immutable fitted models; validation-only threshold selection."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score,auc,brier_score_loss
from threadpoolctl import threadpool_limits
from src.phase4 import ROOT
from src.phase4.data import BASE,METRICS,MODEL_NAMES,CACHE,guard,training_arrays,diagnostic_labels
from src.phase4.train import load_model
from src.phase4.metrics import binary_metrics
from src.baseline.common import csv_write,json_write,sha256,require

OUT=BASE/'diagnostics'
PARTS=('train','validation','test')
GROUPS=('train','validation','test','training_benign','training_malicious','validation_Infilteration','test_Infilteration','test_Bot')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(name,rows): csv_write(OUT/name,rows)
def init():
    OUT.mkdir(exist_ok=True)
    for name in ('roc_curves','pr_curves','score_distributions','calibration','threshold_curves'):
        (OUT/'figures'/name).mkdir(parents=True,exist_ok=True)
    manifest=OUT/'input_integrity_manifest.json'
    if not manifest.exists():
        guard(full=True)
        files=[p for p in BASE.rglob('*') if p.is_file() and OUT not in p.parents]
        files += list((ROOT/'src/phase4').glob('*.py'))+list((ROOT/'src/baseline').glob('*.py'))
        files += [ROOT/'configs/phase4_seed42.json',ROOT/'configs/baseline.json']
        json_write(manifest,[dict(path=str(p),sha256=sha256(p)) for p in files])
    for row in read(manifest): require(sha256(row['path'])==row['sha256'],'Existing baseline artifact changed: '+row['path'])

def labels(part):
    if part=='train': return np.asarray(training_arrays()[1]),None,None
    # These existing caches are read only; no new diagnostic cache is created outside OUT.
    require((CACHE/f'{part}_diagnostics.npz').exists(),'Missing original label cache')
    return diagnostic_labels(part)

def scores(model,part):
    evaluation=read(METRICS/f'{model}_seed42_evaluation.json')
    record=evaluation['partitions'][part]
    require(sha256(record['probability_path'])==record['probability_sha256'],'Prediction checksum changed')
    return np.load(record['probability_path'],mmap_mode='r')

def counts_at(y,p,threshold): return binary_metrics(y,p,float(threshold),include_auc=False)

def threshold_candidates(y,p):
    # Distinct scores only, including the all-negative operating point above max score.
    order=np.argsort(p,kind='stable')[::-1];s=np.asarray(p)[order];target=np.asarray(y)[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.r_[0,np.cumsum(target,dtype=np.int64)[ends]]
    fp=np.r_[0,ends+1-tp[1:]]
    thresholds=np.r_[np.nextafter(float(s[0]),np.inf),s[ends]]
    pos=int(y.sum());neg=len(y)-pos;fn=pos-tp;tn=neg-fp
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp),dtype=float),where=(2*tp+fp+fn)>0)
    bf1=np.divide(2*tn,2*tn+fp+fn,out=np.zeros(len(tp),dtype=float),where=(2*tn+fp+fn)>0)
    return thresholds,tp,fp,fn,tn,f1,(f1+bf1)/2

def select_thresholds(y,p):
    t,tp,fp,fn,tn,f1,macro=threshold_candidates(y,p)
    selected=[]
    for criterion,metric in [('maximum_binary_f1',f1),('maximum_macro_f1',macro)]:
        # Candidates descend: ties select the higher threshold (fewer alerts).
        idx=int(np.argmax(metric));selected.append((criterion,idx,None))
    for cap in (.01,.005,.001):
        feasible=np.flatnonzero(fp/(fp+tn)<=cap)
        # Maximize validation recall under the cap, then minimize false positives, then highest threshold.
        best=feasible[tp[feasible]==tp[feasible].max()];idx=int(best[0])
        selected.append((f'fpr_at_most_{cap:g}',idx,cap))
    return [dict(criterion=c,threshold=float(t[i]),validation_fpr_cap=cap,
                 selection_partition='validation',**{f'validation_{k}':v for k,v in counts_at(y,p,t[i]).items()}) for c,i,cap in selected]

def freeze_thresholds():
    path=OUT/'validation_threshold_lock.json'
    if path.exists():
        state=read(path)
        require(state['validation_predictions']=={m:sha256(read(METRICS/f'{m}_seed42_evaluation.json')['partitions']['validation']['probability_path']) for m in MODEL_NAMES},'Validation predictions changed')
        return state['thresholds']
    # No test scores or test labels are loaded by this function.
    y=labels('validation')[0];rows=[];hashes={}
    for model in MODEL_NAMES:
        p=scores(model,'validation');rows.extend(dict(model=model,seed=42,**r) for r in select_thresholds(y,p))
        hashes[model]=sha256(read(METRICS/f'{model}_seed42_evaluation.json')['partitions']['validation']['probability_path'])
    write('validation_selected_thresholds.csv',rows)
    json_write(path,dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),thresholds=rows,
        validation_predictions=hashes,tie_rule='highest threshold among equal optima; FPR constraints maximize recall then minimize FP',
        selection_uses_test=False,threshold_csv_sha256=sha256(OUT/'validation_selected_thresholds.csv')))
    print('Validation thresholds persisted and locked before transfer to test.',flush=True)
    return rows

def training_metrics():
    dest=OUT/'training_partition_metrics.csv'
    if dest.exists(): return
    x,y=training_arrays();rows=[]
    for name in MODEL_NAMES:
        output=OUT/f'{name}_training_metrics.json'
        if output.exists(): rows.append(read(output));continue
        model=load_model(name);p=np.empty(len(y),dtype=np.float64);start=time.perf_counter()
        with threadpool_limits(limits=2):
            for offset in range(0,len(y),50000):
                batch=np.asarray(x[offset:offset+50000]);p[offset:offset+len(batch)]=model.inplace_predict(batch) if name=='xgboost' else model.predict_proba(batch)[:,1]
        row=dict(model=name,seed=42,partition='train',in_sample=True,inference_seconds=time.perf_counter()-start,
                 probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),**binary_metrics(y,p))
        json_write(output,row);rows.append(row);del model,p
        print('TRAINING PARTITION INFERENCE COMPLETE',name,row['roc_auc'],row['malicious_recall'],flush=True)
    write(dest.name,rows)

def summary(p):
    return dict(total_records=len(p),mean_probability=float(np.mean(p)),median_probability=float(np.median(p)),
        **{f'p{q}_probability':float(np.percentile(p,q)) for q in (75,90,95,99)},maximum_probability=float(np.max(p)),
        predicted_malicious_at_0_5=int(np.count_nonzero(p>=.5)),recall_at_0_5=float(np.mean(p>=.5)))

def savefig(fig,folder,name):
    fig.savefig(OUT/'figures'/folder/name,dpi=150,bbox_inches='tight');plt.close(fig)

def score_analysis(thresholds):
    known=[];unseen=[];distributions=[];transfer=[];temporal=[];bots=[];calibration=[];calbins=[];curves=[];rank=[]
    for model in MODEL_NAMES:
        partitions={p:labels(p) for p in ('validation','test')}
        probabilities={p:scores(model,p) for p in ('validation','test')}
        selected=[r for r in thresholds if r['model']==model]
        fig_hist,axs_h=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        fig_cdf,axs_c=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        fig_roc,axr=plt.subplots(figsize=(6,5));fig_pr,axp=plt.subplots(figsize=(6,5))
        for panel,part in enumerate(('validation','test')):
            y,codes,book=partitions[part];p=probabilities[part]
            fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
            aucroc=roc_auc_score(y,p);ap=average_precision_score(y,p);prauc=auc(recall,precision)
            axr.plot(fpr,tpr,label=f'{part} AUC={aucroc:.4f}')
            axp.plot(recall,precision,label=f'{part} AP={ap:.4f}, PR-AUC={prauc:.4f}')
            axp.axhline(y.mean(),ls=':',alpha=.5,label=f'{part} prevalence={y.mean():.4f}')
            rank.append(dict(model=model,partition=part,roc_auc=aucroc,pr_auc=prauc,average_precision=ap,
                prevalence=float(y.mean()),average_precision_lift=ap/y.mean(),recall_at_0_5=float(np.mean(p[y==1]>=.5))))
            families=['Benign','Infilteration']+(['Bot'] if part=='test' else ['Brute Force -Web','Brute Force -XSS','SQL Injection'])
            for family in families:
                values=np.asarray(p[codes==book.index(family)]);s=summary(values)
                row=dict(model=model,partition=part,attack_family=family,**s)
                if family=='Benign': row['false_positive_rate_at_0_5']=row.pop('recall_at_0_5')
                distributions.append(row)
                if family in ('Brute Force -Web','Brute Force -XSS','SQL Injection'):
                    known.append(dict(**row,number_of_records=len(values),predicted_malicious=s['predicted_malicious_at_0_5'],
                        predicted_benign=len(values)-s['predicted_malicious_at_0_5'],recall=s['recall_at_0_5'],small_sample_warning=True))
                elif family!='Benign': unseen.append(dict(**row,training_status='UNSEEN'))
                if family in ('Benign','Infilteration','Bot'):
                    axs_h[panel].hist(values,bins=np.linspace(0,1,101),weights=np.ones(len(values))/len(values),histtype='step',label=family)
                    sv=np.sort(values);indices=np.unique(np.r_[np.linspace(0,len(sv)-1,min(4000,len(sv))).astype(int),np.searchsorted(sv,.5).clip(0,len(sv)-1)])
                    axs_c[panel].plot(sv[indices],(indices+1)/len(sv),label=family)
                if family=='Infilteration':
                    temporal.append(dict(**row,criterion='fixed_0_5',threshold=.5,recall=s['recall_at_0_5']))
                    for t in selected: temporal.append(dict(**row,criterion=t['criterion'],threshold=t['threshold'],recall=float(np.mean(values>=t['threshold']))))
                if family=='Bot':
                    bots.append(dict(**row,criterion='fixed_0_5',threshold=.5,recall=s['recall_at_0_5']))
                    for t in selected: bots.append(dict(**row,criterion=t['criterion'],threshold=t['threshold'],recall=float(np.mean(values>=t['threshold']))))
            for ax in (axs_h[panel],axs_c[panel]):
                ax.axvline(.5,color='black',ls='--',lw=.8);ax.set_xlabel('Predicted malicious probability');ax.set_title(f'{model}: {part}');ax.legend()
            axs_h[panel].set_yscale('log');axs_h[panel].set_ylabel('Fraction of group in score bin (log)')
            axs_c[panel].set_ylabel('Empirical cumulative probability')
        for ax in (axr,axp): ax.legend(fontsize=8);ax.grid(alpha=.2)
        axr.plot([0,1],[0,1],'k--');axr.set(xlabel='False-positive rate',ylabel='Malicious recall',title=model)
        axp.set(xlabel='Malicious recall',ylabel='Precision',title=model)
        savefig(fig_roc,'roc_curves',model+'.png');savefig(fig_pr,'pr_curves',model+'.png')
        savefig(fig_hist,'score_distributions',model+'_histograms.png');savefig(fig_cdf,'score_distributions',model+'_ecdf.png')
        # Transfer only after the persisted validation lock is verified.
        lock=read(OUT/'validation_threshold_lock.json');require(sha256(OUT/'validation_selected_thresholds.csv')==lock['threshold_csv_sha256'],'Threshold CSV changed')
        y,codes,book=partitions['test'];p=probabilities['test']
        for t in selected:
            m=counts_at(y,p,t['threshold']);transfer.append(dict(model=model,criterion=t['criterion'],selection_partition='validation',
                **m,bot_recall=float(np.mean(p[codes==book.index('Bot')]>=t['threshold'])),
                infilteration_recall=float(np.mean(p[codes==book.index('Infilteration')]>=t['threshold']))))
        fig,ax=plt.subplots(figsize=(7,5))
        grid=[]
        for t in np.linspace(0,1,201):
            m=counts_at(y,p,t);grid.append(dict(model=model,threshold=float(t),post_hoc_diagnostic_only=True,
                recall=m['recall'],fpr=m['false_positive_rate'],precision=m['precision']))
        curves.extend(grid)
        for key in ('recall','fpr','precision'): ax.plot([r['threshold'] for r in grid],[r[key] for r in grid],label=key)
        ax.set(xlabel='Threshold',ylabel='Rate',title=f'{model}: TEST POST-HOC ONLY; no selection');ax.legend();savefig(fig,'threshold_curves',model+'.png')
        y=partitions['validation'][0];p=probabilities['validation'];bins=np.minimum((p*20).astype(int),19)
        points=[];ece=0
        for b in range(20):
            mask=bins==b;n=int(mask.sum())
            if not n: continue
            mean=float(p[mask].mean());fraction=float(y[mask].mean());ece+=n/len(y)*abs(mean-fraction)
            points.append(dict(model=model,bin=b,lower=b/20,upper=(b+1)/20,records=n,mean_probability=mean,observed_malicious_fraction=fraction))
        calbins.extend(points);prevalence=float(y.mean());brier=brier_score_loss(y,p);null=prevalence*(1-prevalence)
        calibration.append(dict(model=model,partition='validation',brier_score=brier,constant_prevalence_brier=null,
            brier_skill_score=1-brier/null,ece_20_equal_width_bins=ece,mean_predicted_probability=float(p.mean()),observed_prevalence=prevalence,
            note='Descriptive calibration only; no recalibration fitted'))
        fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'k--',label='Ideal')
        ax.plot([v['mean_probability'] for v in points],[v['observed_malicious_fraction'] for v in points],'o-',label='Validation')
        ax.set(xlabel='Mean predicted probability',ylabel='Observed malicious fraction',title=f'{model}: validation reliability');ax.legend();savefig(fig,'calibration',model+'.png')
        print('SCORE DIAGNOSTICS COMPLETE',model,flush=True)
    for file,rows in [('known_validation_attack_diagnostics.csv',known),('unseen_attack_score_diagnostics.csv',unseen),('score_distribution_summary.csv',distributions),
        ('threshold_transfer_to_test.csv',transfer),('infilteration_temporal_analysis.csv',temporal),('bot_analysis.csv',bots),
        ('calibration_metrics.csv',calibration),('calibration_bins.csv',calbins),('test_threshold_curve_posthoc.csv',curves),('ranking_metrics.csv',rank)]: write(file,rows)

def main():
    init()
    thresholds=freeze_thresholds()
    training_metrics()
    score_analysis(thresholds)
    from .features import run as feature_analysis
    feature_analysis()
    from .report import run as report
    report()

if __name__=='__main__': main()
