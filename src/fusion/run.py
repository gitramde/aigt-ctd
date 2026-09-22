"""Read-only source artifacts; validation locks precede test evaluation."""
import csv
import json
import time
from pathlib import Path
from . import ROOT
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.baseline.common import sha256, json_write, csv_write, require
from src.phase4.metrics import binary_metrics

OUT = ROOT/'results/fusion_v1'
BASE = ROOT/'results/baseline_v1'
AE = ROOT/'results/anomaly_v1'
CAPS = (.05, .01, .005, .001)
METHODS = ('random_forest', 'autoencoder', 'max', 'weighted_0.25', 'weighted_0.50', 'weighted_0.75')
INTERPRETATION = ('At these predeclared 1% operating points, the score-fusion rules do not show an overall improvement over frozen RF: '
    'their aggregate recall and F1 are lower. Weighted fusion improves Infilteration recall but sacrifices nearly all Bot detection. '
    'MAX behaves almost like the AE alone. The percentile transform places ordinary benign AE observations across [0,1], '
    'while RF probabilities have a different distribution; sharing numerical bounds does not give the two signals equal calibration. '
    'OR preserves RF detections and adds complementary AE detections, but approximately doubles actual FPR and does not improve F1. '
    'The evidence supports complementary detection behavior, not superiority under comparable false-alarm control. '
    'These observations do not select a winning method or alpha.\n\n')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write(name, values):
    csv_write(OUT/name, values)

def percentile(reference, values):
    return np.searchsorted(reference, values, side='right') / len(reference)

def benign_threshold(values, cap):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    allowed = int(np.floor(cap*len(ordered)))
    threshold = float(np.nextafter(ordered[len(ordered)-allowed-1], np.inf))
    require(np.count_nonzero(ordered >= threshold) <= allowed, 'FPR cap failed')
    return threshold

def macro_threshold(y, score):
    order = np.argsort(score, kind='stable')[::-1]
    s, target = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.r_[0, np.cumsum(target, dtype=np.int64)[ends]]
    fp = np.r_[0, ends+1-tp[1:]]
    fn, tn = int(y.sum())-tp, int((y==0).sum())-fp
    macro = (np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(tp)), where=(2*tp+fp+fn)>0)
             + np.divide(2*tn, 2*tn+fp+fn, out=np.zeros(len(tp)), where=(2*tn+fp+fn)>0))/2
    thresholds = np.r_[np.nextafter(float(s[0]), np.inf), s[ends]]
    return float(thresholds[np.argmax(macro)])

def signals(s, a, reference):
    normalized = percentile(reference, a)
    return dict(random_forest=s, autoencoder=a, max=np.maximum(s, normalized),
                **{f'weighted_{alpha:.2f}': alpha*s+(1-alpha)*normalized for alpha in (.25,.5,.75)})

def ranking(y, score):
    p, r, _ = precision_recall_curve(y, score)
    return dict(roc_auc=float(roc_auc_score(y, score)), pr_auc=float(auc(r,p)),
                average_precision=float(average_precision_score(y, score)))

def markdown(records, columns):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(fmt(row.get(k,'')) for k in columns)+' |' for row in records])

def integrity_before():
    expected = {}
    def include(path, digest):
        key = str(Path(path).resolve())
        require(key not in expected or expected[key] == digest, 'Conflicting historic hashes: '+key)
        expected[key] = digest
    guard = read(BASE/'metrics/frozen_input_guard.json')
    for item in guard['small_artifacts']+guard['data_artifacts']:
        include(item['path'], item['sha256'])
    for manifest in (BASE/'diagnostics/input_integrity_manifest.json', AE/'metrics/frozen_artifacts.json'):
        for item in read(manifest):
            include(item['path'], item['sha256'])
    for part in read(BASE/'split_plan.json')['partitions'].values():
        include(part['membership_path'], part['membership_sha256'])
    for model in ('autoencoder_A','autoencoder_B','isolation_forest'):
        tr = read(AE/f'metrics/{model}_training.json')
        include(tr['model_path'], tr['model_sha256'])
        include(AE/f'metrics/{model}_benign_validation.npy', tr['benign_validation_score_sha256'])
    for model in ('autoencoder_B','isolation_forest'):
        for entry in read(AE/f'metrics/{model}_scoring.json')['partitions'].values():
            include(entry['score_path'], entry['score_sha256'])
    for item in read(AE/'anomaly_training_manifest.json')['artifacts']:
        include(item['path'], item['sha256'])
    for name in ('benign_only','label_aware'):
        lock = read(AE/f'configs/{name}_threshold_lock.json')
        include(AE/f'{name}_thresholds.csv', lock['csv_sha256'])
    lock = read(BASE/'diagnostics/validation_threshold_lock.json')
    include(BASE/'diagnostics/validation_selected_thresholds.csv', lock['threshold_csv_sha256'])
    for name, digest in read(AE/'metrics/integrity_verification.json')['source_code_sha256'].items():
        include(ROOT/'src/anomaly'/name, digest)
    print(f'Checking {len(expected)} historical SHA-256 references, including raw data', flush=True)
    for i, (path, digest) in enumerate(expected.items()):
        require(sha256(path)==digest, 'Historical checksum mismatch: '+path)
        if i%100==0:
            print('Verified', i+1, 'historical files', flush=True)
    protected = []
    for folder in ('results/baseline_v1','results/anomaly_v1','results/temporal_v1','results/graph_v1',
                   'src/baseline','src/phase4','src/phase4a','src/anomaly','src/temporal','src/graph'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                protected.append(dict(path=str(path), sha256=sha256(path)))
    json_write(OUT/'input_integrity_manifest.json', dict(historical_references=expected, protected=protected))
    return protected

def load(part):
    plan = read(BASE/'split_plan.json')['partitions'][part]
    table = pq.read_table(plan['membership_path'], columns=['target_binary','attack_label'])
    y = table['target_binary'].to_numpy().astype(np.uint8)
    family = table['attack_label'].to_numpy()
    with np.load(ROOT/f'data/phase4_cache/{part}_diagnostics.npz') as cached:
        book = read(BASE/'metrics/pretraining_check.json')['diagnostic_codebook']
        require(np.array_equal(y,cached['y']), 'Cached targets differ from membership')
        require(np.array_equal(family,np.asarray(book)[cached['family']]), 'Cached families differ from membership')
    rf = read(BASE/'metrics/random_forest_seed42_evaluation.json')['partitions'][part]
    ae = read(AE/'metrics/autoencoder_B_scoring.json')['partitions'][part]
    require(sha256(rf['probability_path'])==rf['probability_sha256'], 'RF score hash mismatch')
    require(sha256(ae['score_path'])==ae['score_sha256'], 'AE score hash mismatch')
    s = np.asarray(np.load(rf['probability_path']), dtype=np.float64)
    a = np.asarray(np.load(ae['score_path']), dtype=np.float64)
    require(len(y)==len(s)==len(a)==plan['rows'], 'Full membership count mismatch')
    require(np.isfinite(s).all() and np.isfinite(a).all() and ((s>=0)&(s<=1)).all(), 'Invalid scores')
    return y, family, s, a

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('score_relationship','fusion_roc','fusion_pr','family_comparison'):
        (OUT/'figures'/folder).mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=42, alphas=[.25,.5,.75], caps=list(CAPS), model_retraining=False,
        normalization='A_norm=count(benign validation AE scores <= score)/N; right ECDF; ties share upper rank; tails 0/1',
        selection='Report all primary rules; no test selection; no development winner selected',
        primary='Benign validation only; >= threshold; nextafter conservative order statistic',
        secondary='Maximum macro-F1; validation labels including Infilteration; ties choose higher threshold',
        OR='Existing Phase4A RF and Phase7 AE paired caps .01/.005/.001; mixed calibration; no 5% saved RF threshold',
        OR_ranking='ROC-AUC/PR-AUC/AP computed on binary decision only; not continuous risk ranking',
        visualization='Seed42 uniform sample up to 30000 rows per group; correlations use all group rows')
    json_write(OUT/'protocol.json', protocol)
    protected = integrity_before()
    y, family, s, a = load('validation')
    reference = np.sort(a[y==0])
    require(np.array_equal(a[y==0],np.load(AE/'metrics/autoencoder_B_benign_validation.npy')), 'AE calibration mismatch')
    np.save(OUT/'benign_validation_ecdf_reference.npy', reference)
    val = signals(s,a,reference)
    primary, secondary, operating = [], [], []
    for method in METHODS:
        for cap in CAPS:
            t = benign_threshold(val[method][y==0], cap)
            primary.append(dict(method=method, protocol='A_benign_only', criterion=f'fpr_at_most_{cap:g}',
                target_fpr=cap, threshold=t, benign_validation_N=len(reference),
                observed_validation_fpr=float(np.mean(val[method][y==0]>=t))))
    write('benign_only_thresholds.csv', primary)
    json_write(OUT/'primary_threshold_lock.json',dict(rows=primary, selection_uses_test=False,
        selection_uses_malicious_validation=False, reference_sha256=sha256(OUT/'benign_validation_ecdf_reference.npy'),
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('Primary thresholds locked using benign validation only', flush=True)
    for method in METHODS:
        secondary.append(dict(method=method, protocol='B_label_aware', criterion='maximum_macro_f1',
                              threshold=macro_threshold(y,val[method])))
    write('label_aware_thresholds.csv', secondary)
    rfrows = rows(BASE/'diagnostics/validation_selected_thresholds.csv')
    aerows = rows(AE/'benign_only_thresholds.csv')
    for cap in (.01,.005,.001):
        rt = float(next(r['threshold'] for r in rfrows if r['model']=='random_forest' and r['criterion']==f'fpr_at_most_{cap:g}'))
        at = float(next(r['threshold'] for r in aerows if r['model']=='autoencoder_B' and float(r['target_fpr'])==cap))
        for method in ('random_forest','autoencoder','or'):
            operating.append(dict(method=method, protocol='frozen_existing', criterion=f'paired_cap_{cap:g}',
                target_fpr=cap, threshold=rt if method=='random_forest' else at if method=='autoencoder' else .5,
                rf_threshold=rt, ae_threshold=at))
    json_write(OUT/'secondary_and_existing_threshold_lock.json',dict(rows=secondary+operating,
        selection_uses_test=False, validation_infilteration_used=True))
    write('fusion_methods.csv', [dict(method=m, formula=('S' if m=='random_forest' else 'raw AE MSE (standalone)' if m=='autoencoder'
        else 'max(S,A_norm)' if m=='max' else 'alpha*S+(1-alpha)*A_norm'),
        alpha=m.split('_')[-1] if m.startswith('weighted') else '', calibration='A and B') for m in METHODS]+
        [dict(method='or',formula='(S>=frozen RF threshold) OR (raw AE>=frozen AE threshold)',calibration='existing mixed protocols')])
    all_metrics, families, correlations, overlaps = [], [], [], []
    for part in ('validation','test'):
        if part=='test':
            y, family, s, a = load(part)
        values = signals(s,a,reference)
        rank = {m:ranking(y,v) for m,v in values.items()}
        for setting in primary+secondary+operating:
            method = setting['method']
            score = ((s>=setting['rf_threshold'])|(a>=setting['ae_threshold'])).astype(float) if method=='or' else values[method]
            pred = score>=setting['threshold']
            result = binary_metrics(y,pred.astype(float),.5,include_auc=False)
            result.update(ranking(y,score) if method=='or' else rank[method])
            result.update(setting,partition=part,ranking_basis='binary decision' if method=='or' else 'continuous score')
            all_metrics.append(result)
            for name in sorted(set(family)-{'Benign'}):
                mask = family==name
                n, detected = int(mask.sum()), int(pred[mask].sum())
                families.append(dict(partition=part,**setting,attack_family=name,N=n,detected=detected,missed=n-detected,recall=detected/n))
        refop = next(r for r in operating if r['method']=='or' and r['target_fpr']==.01)
        rp, ap = s>=refop['rf_threshold'], a>=refop['ae_threshold']
        for name in ('Bot','Infilteration'):
            mask = family==name
            if not mask.any():
                continue
            both = int((rp&ap&mask).sum()); union = int(((rp|ap)&mask).sum())
            overlaps.append(dict(partition=part,attack_family=name,N=int(mask.sum()),rf_only=int((rp&~ap&mask).sum()),
                ae_only=int((~rp&ap&mask).sum()),both=both,neither=int((~rp&~ap&mask).sum()),
                jaccard=both/union if union else None,rf_threshold=refop['rf_threshold'],ae_threshold=refop['ae_threshold']))
        normalized = percentile(reference,a)
        for name in (('Benign','Infilteration') if part=='validation' else ('Benign','Bot','Infilteration')):
            mask = family==name
            for label, signal in (('raw_ae',a),('normalized_ae',normalized)):
                correlations.append(dict(partition=part,group=name,N=int(mask.sum()),anomaly_signal=label,
                    pearson=float(pearsonr(s[mask],signal[mask]).statistic),spearman=float(spearmanr(s[mask],signal[mask]).statistic)))
            ids = np.flatnonzero(mask)
            ids = np.sort(np.random.default_rng(42).choice(ids,min(30000,len(ids)),replace=False))
            fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for ax, signal, label in zip(axes,(np.log1p(a),normalized),('log(1 + raw AE MSE)','AE benign-validation percentile')):
                h = ax.hexbin(s[ids],signal[ids],gridsize=45,mincnt=1,bins='log',cmap='viridis')
                ax.set(xlabel='Frozen RF probability',ylabel=label)
                fig.colorbar(h,ax=ax,label='Sample count (log scale)')
            fig.suptitle(f'{part.title()} {name} | {len(ids):,} plotted / {int(mask.sum()):,} total')
            fig.savefig(OUT/f'figures/score_relationship/{part}_{name}.png',dpi=150);plt.close(fig)
        for kind in ('roc','pr'):
            fig, ax = plt.subplots(figsize=(7,5),layout='constrained')
            for method, score in values.items():
                if kind=='roc':
                    xx, yy, _ = roc_curve(y,score)
                else:
                    yy, xx, _ = precision_recall_curve(y,score)
                # Numerical metrics above use all points; reduce rendering only.
                ids = np.unique(np.linspace(0,len(xx)-1,min(15000,len(xx))).astype(int))
                ax.plot(xx[ids],yy[ids],label=method,linewidth=1.1)
            orpred = (rp|ap).astype(float)
            if kind=='roc':
                xx,yy,_ = roc_curve(y,orpred)
            else:
                yy,xx,_ = precision_recall_curve(y,orpred)
            ax.plot(xx,yy,'k--',label='OR 1% (binary decision)',linewidth=1)
            ax.set(xlabel='False-positive rate' if kind=='roc' else 'Recall',ylabel='Recall' if kind=='roc' else 'Precision',
                   title=f'{part.title()} fusion {kind.upper()} | seed 42',xlim=(0,1),ylim=(0,1.02))
            ax.legend(fontsize=8);fig.savefig(OUT/f'figures/fusion_{kind}/{part}.png',dpi=150);plt.close(fig)
        print('Evaluated full',part,'membership:',len(y),flush=True)
    write('validation_metrics.csv',[r for r in all_metrics if r['partition']=='validation'])
    write('test_metrics.csv',[r for r in all_metrics if r['partition']=='test'])
    write('family_metrics.csv',families);write('score_correlations.csv',correlations);write('rf_ae_detection_overlap.csv',overlaps)
    comparison=[]
    for method in METHODS+('or',):
        protocol_name = 'frozen_existing' if method in ('random_forest','autoencoder','or') else 'A_benign_only'
        r=next(r for r in all_metrics if r['partition']=='test' and r['method']==method and r['protocol']==protocol_name and r.get('target_fpr')==.01)
        f={row['attack_family']:row['recall'] for row in families if row['partition']=='test' and row['method']==method and row['protocol']==protocol_name and row.get('target_fpr')==.01}
        comparison.append(dict(method=method,protocol=protocol_name,actual_test_fpr=r['false_positive_rate'],
            aggregate_malicious_recall=r['recall'],precision=r['precision'],f1=r['f1'],AP=r['average_precision'],
            Bot_recall=f['Bot'],Infilteration_recall=f['Infilteration'],ranking_basis=r['ranking_basis']))
    write('fusion_comparison_1pct.csv',comparison)
    baseline=comparison[0]
    deltas=[dict(method=r['method'],**{f'delta_{k}':r[k]-baseline[k] for k in ('actual_test_fpr','aggregate_malicious_recall','Bot_recall','Infilteration_recall')}) for r in comparison[2:]]
    write('fusion_deltas_vs_rf_1pct.csv',deltas)
    fig, axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,key,title in zip(axes,('Bot_recall','Infilteration_recall','actual_test_fpr'),('Bot recall','Infilteration recall','Actual test FPR')):
        ax.barh([r['method'] for r in comparison],[100*r[key] for r in comparison]);ax.set(xlabel='Percent',title=title)
    fig.suptitle('Validation-calibrated 1% comparison; OR uses two component caps')
    fig.savefig(OUT/'figures/family_comparison/test_1pct.png',dpi=150);plt.close(fig)
    for item in protected:
        require(sha256(item['path'])==item['sha256'], 'Prior artifact changed: '+item['path'])
    checks=dict(historical_hashes_verified=True,prior_artifacts_unchanged=True,full_membership_and_label_alignment=True,
        frozen_scores_only=True,no_retraining=True,benign_only_primary_calibration=True,validation_only_secondary=True,
        no_test_selection=True,all_predeclared_rules_reported=True,all_record_correlations=True)
    json_write(OUT/'integrity_verification.json',dict(status='PASS',checks=checks,protected_files=len(protected),
        source_sha256={p.name:sha256(p) for p in (ROOT/'src/fusion').glob('*.py')}))
    (OUT/'integrity_report.md').write_text('# Phase 8 integrity\n\n'+markdown([dict(check=k,status='PASS') for k in checks],['check','status'])+
        f'\n\n{len(protected)} prior files rehashed before/after analysis. Historical hashes, raw/cleaned data and memberships verified; evidence in input_integrity_manifest.json. Labels are checked against membership rows in exact order. No models loaded or fitted.\n',encoding='utf-8')
    summary='# Phase 8 supervised + anomaly risk-fusion development experiment\n\n'
    summary+='Seed 42 only. Frozen Phase-4 RF probabilities and Phase-7 autoencoder_B reconstruction MSE; no retraining. All validation/test rows use the original baseline_v1 membership and order.\n\n'
    summary+='## Transformations and calibration\n\nA_norm = count(benign validation AE scores <= current score) / N, with N = '+str(len(reference))+'. This right-continuous empirical CDF uses only benign validation observations, includes ties at their upper rank, maps below-minimum values to 0 and above-maximum values to 1. It is validation-dependent by the requested protocol, not a train/validation-independent transformation or a calibrated malicious probability. RF probabilities remain unchanged. Raw AE scores are retained for standalone thresholds and OR decisions.\n\n'
    summary+='MAX and weighted alpha 0.25/0.50/0.75 are all reported. Protocol A uses benign-only validation order-statistic thresholds at 5%, 1%, 0.5%, 0.1%, with >= detection and nextafter tie handling. Normalization and calibration reuse the same benign sample; caps are empirical, not population/test guarantees. Standalone RF is additionally recalibrated benign-only in the complete metric tables. Protocol B selects maximum validation macro-F1 for each score rule, ties favoring the highest threshold; this uses Infilteration labels and is non-strict. Neither protocol selects a winning method or alpha from test outcomes. Locks were saved before this run loaded test scores/labels.\n\n'
    summary+='OR reuses frozen Phase-4A label-aware RF and Phase-7 benign-only AE thresholds at paired 1%, 0.5%, 0.1% caps. No saved 5% RF operating threshold exists, so no 5% OR threshold was invented. OR is a mixed-protocol comparator, not benign-only Protocol A. Two component caps do not impose the same cap on their union. OR ranking metrics use its binary decision (a coarse two-level ranking); its AP is not directly equivalent to continuous-score AP.\n\n'
    summary+='## Critical 1% comparison\n\nRF alone uses its frozen Phase-4A threshold; AE alone uses its frozen Phase-7 Protocol-A threshold. The four score fusions use new benign-only validation thresholds. OR uses the two frozen component thresholds.\n\n'+markdown(comparison,list(comparison[0]))+'\n\n'
    summary+='Changes versus frozen RF (fractions, not percentage points):\n\n'+markdown(deltas,list(deltas[0]))+'\n\n'
    for r in comparison[2:]:
        summary+=f"{r['method']}: Bot recall {100*r['Bot_recall']:.3f}% versus RF {100*baseline['Bot_recall']:.3f}%; Infilteration recall {100*r['Infilteration_recall']:.3f}% versus {100*baseline['Infilteration_recall']:.3f}%; actual FPR {100*r['actual_test_fpr']:.3f}% versus {100*baseline['actual_test_fpr']:.3f}%. These family and false-alarm changes must be considered together.\n\n"
    summary+='## Complementary detection behavior\n\nFrozen component 1% operating points; Jaccard = both / (RF only + AE only + both), undefined for an empty union.\n\n'+markdown([r for r in overlaps if r['partition']=='test'],['attack_family','N','rf_only','ae_only','both','neither','jaccard'])+'\n\n'
    summary+='## Interpretation of the operating-point tradeoffs\n\n'+INTERPRETATION
    summary+='## Score relationships\n\nPearson and Spearman use every record in each group, with average ranks for ties. Figures use a deterministic seed-42 sample of at most 30,000 rows per group only for display.\n\n'+markdown(correlations,['partition','group','N','anomaly_signal','pearson','spearman'])+'\n\n'
    summary+='## Scope and outputs\n\nComplete validation/test tables include accuracy, precision, recall, F1, macro-F1, FPR, FNR, ROC-AUC, trapezoidal PR-AUC, AP and confusion counts. family_metrics.csv reports counts and recall for each present malicious family and every operating point. Figures cover score relationships, ROC, PR and family comparison. All integrity checks passed; prior artifacts remain unchanged.\n\nThis is a post-hoc seed-42 development experiment on attack families absent from training. The test cohort was observed in earlier phases. No statistical significance, confirmatory improvement, zero-day detection or general unseen-threat superiority is claimed. The frozen preprocessing was fitted on all TRAIN classes; the AE model itself was fitted on benign rows. No GATv2, Transformer, graph+temporal integration, explainability or new features were introduced. Stop after Phase 8.\n'
    (OUT/'phase8_results_summary.md').write_text(summary,encoding='utf-8')
    print('Phase 8 complete; all integrity checks PASS',flush=True)

if __name__=='__main__':
    run()
