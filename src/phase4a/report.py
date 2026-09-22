"""Generated Phase 4A interpretation and artifact verification."""
import csv,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from .run import OUT,BASE,MODEL_NAMES,labels,scores,read,write,init,guard
from .features import ks_sorted
from src.baseline.common import json_write,sha256,require


def records(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))

def num(row,key): return float(row[key])
def table(rows,columns):
    lines=['| '+' | '.join(label for key,label in columns)+' |','|'+'|'.join('---' for _ in columns)+'|']
    for row in rows:
        cells=[]
        for key,label in columns:
            value=row.get(key,'')
            try:
                number=float(value)
                value=str(int(number)) if key in ('total_records','records','predicted_malicious','predicted_benign','false_positives','false_negatives','rank') else f'{number:.6g}'
            except (ValueError,TypeError): pass
            cells.append(str(value))
        lines.append('| '+' | '.join(cells)+' |')
    return lines


def extra_diagnostics():
    rows=[];temporal=[]
    for model in MODEL_NAMES:
        for part in ('validation','test'):
            _,codes,book=labels(part);p=scores(model,part)
            for family in (['Infilteration'] if part=='validation' else ['Infilteration','Bot']):
                mask=(codes==book.index('Benign'))|(codes==book.index(family))
                binary=(codes[mask]==book.index(family)).astype('uint8')
                rows.append(dict(model=model,partition=part,attack_family=family,
                    family_vs_same_partition_benign_roc_auc=roc_auc_score(binary,p[mask]),
                    family_vs_same_partition_benign_average_precision=average_precision_score(binary,p[mask]),
                    comparison_prevalence=float(binary.mean()),descriptive_only=True))
        a=scores(model,'validation')[labels('validation')[1]==labels('validation')[2].index('Infilteration')]
        b=scores(model,'test')[labels('test')[1]==labels('test')[2].index('Infilteration')]
        temporal.append(dict(model=model,comparison='validation_Infilteration_to_test_Infilteration',score_ks_statistic=ks_sorted(np.sort(a),np.sort(b))))
    write('family_ranking_diagnostics.csv',rows);write('infilteration_score_shift.csv',temporal)


def run():
    from .plots import run as score_plots
    score_plots()
    extra_diagnostics()
    training=records('training_partition_metrics.csv');known=records('known_validation_attack_diagnostics.csv')
    ranking=records('ranking_metrics.csv');transfer=records('threshold_transfer_to_test.csv');thresholds=records('validation_selected_thresholds.csv')
    unseen=records('unseen_attack_score_diagnostics.csv');calibration=records('calibration_metrics.csv')
    shifts=records('top20_feature_shifts.csv');family=records('family_feature_shift.csv');both=records('family_shift_against_both_rankings.csv')
    temporal=records('infilteration_temporal_analysis.csv');bot=records('bot_analysis.csv')
    expected=read(BASE/'preprocessing.json')['output_feature_names']
    feature_rows=records('feature_distribution_shift.csv')
    require(set(r['feature'] for r in feature_rows)==set(expected),'Feature coverage mismatch')
    require(len(feature_rows)==3*len(expected) and len(family)==7*len(expected),'Incomplete feature comparisons')
    require(len(training)==4 and len(known)==12 and len(unseen)==12 and len(transfer)==20 and len(thresholds)==20,'Incomplete model diagnostics')
    require(len(temporal)==48 and len(bot)==24 and len(calibration)==4,'Incomplete temporal/calibration diagnostics')
    require(all(r['selection_partition']=='validation' for r in thresholds+transfer),'Incorrect threshold provenance')
    for r in thresholds:
        if r['validation_fpr_cap']:
            require(num(r,'validation_false_positive_rate')<=num(r,'validation_fpr_cap'),'Validation FPR constraint violated')
    test_counts=read(BASE/'split_plan.json')['partitions']['test']['class_counts']
    for r in transfer:
        require(num(r,'true_positives')+num(r,'false_negatives')==sum(n for label,n in test_counts.items() if label!='Benign'),'Test malicious count mismatch')
        require(num(r,'true_negatives')+num(r,'false_positives')==test_counts['Benign'],'Test benign count mismatch')
    json_write(OUT/'output_validation.json',dict(status='PASS',feature_dimensions=len(expected),feature_comparisons=len(feature_rows),family_feature_comparisons=len(family),frozen_validation_thresholds=len(thresholds)))
    lines=['# Phase 4A: temporal generalization and threshold diagnosis','',
        'Existing seed-42 fitted models only. No fitting, recalibration, preprocessing changes or partition changes were performed. Validation/test probabilities are the saved Phase 4 outputs. Training inference uses the existing read-only model-input cache. All numbers and tables below are generated from diagnostic outputs.','',
        '## Scope and safeguards','',
        'Every malicious test record belongs to a family absent from training: Bot or Infilteration. Infilteration dominates validation malicious traffic. Known-family test recall is not defined and is not computed. This is binary detection, not multiclass recognition; Bot is an attack family absent from training, not a claimed zero-day attack.','',
        'Thresholds were selected exclusively from validation labels/scores, saved in validation_selected_thresholds.csv and bound by validation_threshold_lock.json before any Phase 4A transfer to test. No test result selects a threshold or model. The original Phase 4 test results had already been observed; this is a transparently post-hoc diagnostic phase, not a new untouched confirmatory evaluation.','',
        '## Supervised training performance','',
        'These are in-sample metrics at threshold 0.5, not unbiased generalization estimates. High aggregate training performance shows learning of the training distribution but does not establish performance on every rare training family.','']
    plan=read(BASE/'split_plan.json')
    date_rows=[dict(partition=p,dates=', '.join(v['dates']),records=v['rows'],timestamp_min=v['timestamp_min'],timestamp_max=v['timestamp_max']) for p,v in ((p,plan['partitions'][p]) for p in ('train','validation','test'))]
    position=lines.index('## Supervised training performance')
    lines[position:position]=['Frozen temporal cohorts:','']+table(date_rows,[('partition','Partition'),('dates','Dates'),('records','Records'),('timestamp_min','First timestamp'),('timestamp_max','Last timestamp')])+['']
    lines += table(training,[('model','Model'),('roc_auc','ROC-AUC'),('pr_auc','PR-AUC'),('average_precision','AP'),('precision','Precision'),('malicious_recall','Recall'),('f1','F1'),('macro_f1','Macro-F1'),('false_positive_rate','FPR'),('false_negative_rate','FNR')])
    lines += ['',f'Training malicious recall spans {min(num(r,"malicious_recall") for r in training):.2%}–{max(num(r,"malicious_recall") for r in training):.2%}. Thus the later-period collapse is not a broad inability to fit the aggregate training distribution. This does not rule out overfitting or reliance on date-specific correlations.','',
        '## Known-family validation behavior','',
        'The validation samples are small: Brute Force -Web has 362 records, Brute Force -XSS 151, and SQL Injection 53. Treat these recalls as descriptive, with limited support. These families also had few training examples. No known-family test recall is reported.','']
    lines += table(known,[('model','Model'),('attack_family','Family'),('total_records','N'),('predicted_malicious','Detected'),('predicted_benign','Missed'),('recall','Recall'),('mean_probability','Mean score'),('median_probability','Median'),('p90_probability','p90'),('maximum_probability','Max')])
    lines += ['', 'The known-family results also vary substantially and include many misses. Unseen-family status therefore cannot by itself explain all later-period errors.','',
        '## Unseen-family validation and test behavior','']
    lines += table(unseen,[('model','Model'),('partition','Partition'),('attack_family','Family'),('total_records','N'),('predicted_malicious_at_0_5','Detected at .5'),('recall_at_0_5','Recall at .5'),('mean_probability','Mean'),('median_probability','Median'),('p75_probability','p75'),('p90_probability','p90'),('p95_probability','p95'),('p99_probability','p99'),('maximum_probability','Max')])
    lines += ['', '## Ranking versus fixed-threshold performance','',
        'ROC-AUC evaluates ranking across thresholds; average precision and trapezoidal PR-AUC are distinct summaries and are both retained. AP is compared with the partition malicious prevalence; PR comparisons across partitions must account for their different prevalences. Ranking skill does not guarantee useful detection at threshold 0.5.','']
    lines += table(ranking,[('model','Model'),('partition','Partition'),('roc_auc','ROC-AUC'),('pr_auc','PR-AUC'),('average_precision','AP'),('prevalence','Prevalence'),('average_precision_lift','AP / prevalence'),('recall_at_0_5','Recall at .5')])
    for model in MODEL_NAMES:
        v=next(r for r in ranking if r['model']==model and r['partition']=='validation')
        t=next(r for r in ranking if r['model']==model and r['partition']=='test')
        selected=next(r for r in transfer if r['model']==model and r['criterion']=='maximum_binary_f1')
        lines += ['',f'**{model}:** validation ROC-AUC {num(v,"roc_auc"):.4f}, test ROC-AUC {num(t,"roc_auc"):.4f}, test AP {num(t,"average_precision"):.4f}. At 0.5, test recall is {num(t,"recall_at_0_5"):.3%}. The validation maximum-F1 threshold transfers to test recall {num(selected,"recall"):.2%}, precision {num(selected,"precision"):.2%}, and FPR {num(selected,"false_positive_rate"):.2%}. This describes the preselected operating point; it is not selected from test.']
    lines += ['', 'Random Forest retains materially useful test ranking despite poor classification at 0.5; its validation ranking is much weaker. MLP also has a substantial test ranking/fixed-threshold gap. The other models have weaker overall test ROC ranking. Per-family ranking against benign traffic from the same partition is below; this separates changes in family mixture from a single pooled metric.','']
    lines += table(records('family_ranking_diagnostics.csv'),[('model','Model'),('partition','Partition'),('attack_family','Family'),('family_vs_same_partition_benign_roc_auc','ROC-AUC vs benign'),('family_vs_same_partition_benign_average_precision','AP vs benign'),('comparison_prevalence','Comparison prevalence')])
    lines += ['', '## Validation-only threshold selection and transfer','',
        'The search exhausts distinct validation scores with prediction rule score >= threshold and includes an all-negative point just above the maximum score. Maximum-F1 and maximum-macro-F1 ties prefer the higher threshold. For FPR constraints, maximize validation recall subject to the cap, then minimize false positives and prefer the higher threshold. No minimum recall is assumed. A validation FPR cap is not a guarantee about future FPR.','']
    lines += table(thresholds,[('model','Model'),('criterion','Validation objective'),('threshold','Threshold'),('validation_precision','Val precision'),('validation_recall','Val recall'),('validation_f1','Val F1'),('validation_macro_f1','Val macro-F1'),('validation_false_positive_rate','Val FPR')])
    lines += ['', 'Frozen-threshold test transfer:','']
    lines += table(transfer,[('model','Model'),('criterion','Validation objective'),('threshold','Threshold'),('precision','Test precision'),('recall','Test recall'),('f1','Test F1'),('false_positive_rate','Test FPR'),('false_positives','FP'),('false_negatives','FN'),('bot_recall','Bot recall'),('infilteration_recall','Infilteration recall')])
    rf=next(r for r in transfer if r['model']=='random_forest' and r['criterion']=='fpr_at_most_0.01')
    lines += ['',f'At the threshold selected under validation FPR <= 1%, Random Forest test recall is {num(rf,"recall"):.2%} at FPR {num(rf,"false_positive_rate"):.2%}; Bot recall is {num(rf,"bot_recall"):.2%} and Infilteration recall {num(rf,"infilteration_recall"):.2%}. This supports a threshold mismatch for part of its test performance, while the remaining missed attacks show that threshold adjustment is not a complete solution.', '',
        'The separate threshold curves use a predeclared grid from 0 to 1 in steps of 0.005 and are POST-HOC DESCRIPTIVE ONLY. No optimum is extracted from those test curves.','',
        '## Probability calibration','',
        'Reliability diagrams use 20 equal-width validation score bins; empty bins are omitted, with counts retained in calibration_bins.csv. ECE is weighted absolute bin calibration error and depends on binning. The constant-prevalence Brier reference uses the observed validation prevalence only as a descriptive reference, not a fitted replacement model. Negative Brier skill means worse Brier score than that reference.','']
    lines += table(calibration,[('model','Model'),('brier_score','Brier'),('constant_prevalence_brier','Reference Brier'),('brier_skill_score','Brier skill'),('ece_20_equal_width_bins','ECE'),('mean_predicted_probability','Mean predicted'),('observed_prevalence','Observed prevalence')])
    lines += ['', 'All four models have negative validation Brier skill. Reliability gaps and mismatches between mean predicted risk and observed prevalence support poor probability calibration in this later period. Brier score also reflects discrimination and prevalence; it is not an isolated calibration test. Class-weighted fitting may affect probability interpretation, but these diagnostics do not identify a unique cause. No recalibration was fitted.','',
        '## Temporal feature distribution shift','',
        'All 80 frozen input dimensions are covered: 77 numerical features in cleaned native units before imputation, plus the three fixed Protocol one-hot inputs. Numeric summaries use every finite observed value; missing rates are measured before imputation. Population standard deviation uses ddof=0. Quantiles and KS statistics are exact, with no row sampling. Because scaling is a fixed positive affine transform, it does not change the observed-value KS comparisons. Missingness and its imputation effects should be considered separately; these are not post-imputation summaries. Excluded identifiers are not analyzed.','',
        'PSI uses reference-group decile cut points with duplicate cuts removed, unbounded outer bins, an explicit missing bin, and probability floor 1e-6 followed by renormalization. Constant references get a dedicated constant-value bin with lower/upper tails. For overall comparisons the reference is train; family comparisons use the indicated training group, or validation Infilteration for its temporal comparison. KS excludes missing values; PSI includes missingness. PSI depends on binning and smoothing. No significance p-values or universal shift cutoffs are used.','',
        'Features are ranked by KS, with PSI as a tie-breaker. These marginal shifts are descriptive and are not causal feature importance; class/family mixture and acquisition differences can contribute.','']
    for target in ('validation','test'):
        lines += [f'### Top 20: train to {target}','']+table([r for r in shifts if r['target_group']==target],[('rank','Rank'),('feature','Feature'),('ks_statistic','KS'),('psi','PSI'),('reference_missing_rate','Train missing'),('missing_rate','Target missing')])+['']
    lines += ['## Family-level feature shift','',
        'The following features maximize the smaller of KS against training benign and KS against training malicious, requiring a large shift from both reference populations. Complete per-reference comparisons and both PSI values are in family_feature_shift.csv and family_shift_against_both_rankings.csv.','']
    for target in ('validation_Infilteration','test_Infilteration','test_Bot'):
        lines += [f'### {target}','']+table([r for r in both if r['target_group']==target and int(r['rank'])<=10],[('rank','Rank'),('feature','Feature'),('ks_vs_training_benign','KS vs benign'),('ks_vs_training_malicious','KS vs malicious'),('minimum_ks_against_both','Minimum KS')])+['']
    lines += ['## Infilteration across later periods','',
        'This family is absent from training but appears in both validation and test. The same validation-selected thresholds are applied to both periods; no family-specific threshold is optimized. Changes in scores and recall across these periods are distinct from the fact that the family was initially unseen.','']
    lines += table(temporal,[('model','Model'),('partition','Partition'),('criterion','Threshold source'),('threshold','Threshold'),('recall','Infilteration recall'),('mean_probability','Mean score'),('median_probability','Median score')])
    lines += ['']+table(records('infilteration_score_shift.csv'),[('model','Model'),('score_ks_statistic','Validation-to-test score KS')])+['']
    temporal_features=sorted([r for r in family if r['reference_group']=='validation_Infilteration'],key=lambda r:num(r,'ks_statistic'),reverse=True)
    lines += ['Largest within-Infilteration feature shifts:','']+table(temporal_features[:20],[('feature','Feature'),('ks_statistic','KS'),('psi','PSI')])+['',
        '## Bot: attack family absent from training','',
        'Bot occurs only in test. Score statistics are in unseen_attack_score_diagnostics.csv and bot_analysis.csv. All four models miss Bot at threshold 0.5. Some validation-selected lower thresholds detect a portion of Bot; the table retains the corresponding threshold source rather than selecting on Bot outcomes. Strongest feature differences from both training populations are listed above; separate reference rankings follow.','']
    lines += table(bot,[('model','Model'),('criterion','Validation objective / fixed'),('threshold','Threshold'),('recall','Bot recall'),('mean_probability','Mean score'),('median_probability','Median'),('maximum_probability','Maximum')])
    for reference in ('training_benign','training_malicious'):
        top=sorted([r for r in family if r['target_group']=='test_Bot' and r['reference_group']==reference],key=lambda r:num(r,'ks_statistic'),reverse=True)[:10]
        lines += ['',f'Bot vs {reference}:','']+table(top,[('feature','Feature'),('ks_statistic','KS'),('psi','PSI')])
    lines += ['', '## Interpretation and limits','',
        '- **Supervised training:** high aggregate in-sample ranking and recall rule out an aggregate failure to learn the training data; they do not establish generalization or learning of every rare family.',
        '- **Known-family validation:** small samples show variable and often low recall, so failure is not restricted to families absent from training.',
        '- **Unseen-family validation:** Infilteration has low fixed-threshold recall and generally weak discrimination against validation benign traffic.',
        '- **Unseen-family test:** all attacks are from absent training families. Bot and Infilteration must be assessed separately; pooled metrics depend strongly on this mixture.',
        '- **Ranking versus threshold:** especially for Random Forest, useful test ordering coexists with scores below 0.5. Validation-selected thresholds can recover detection with explicit false-positive tradeoffs; this is partial recovery, not a full solution.',
        '- **Calibration:** validation probabilities are unreliable by the reported reliability/Brier diagnostics. No calibration model was fitted, and no test labels were used for calibration.',
        '- **Feature shift:** exact marginal comparisons document distribution changes. They do not prove that any particular shifted feature caused the performance gap, and correlated features may repeat the same signal.',
        '- **Causal attribution:** unseen-family composition, temporal shift, calibration/threshold mismatch and possible overfitting are not independently controlled here. The evidence supports a combination, not a uniquely identified cause. No new model or operating threshold is recommended based on test outcomes.','',
        '## Reproduction and outputs','',
        'Run `python -m src.phase4a.run` using the existing Phase 4 runtime. No `.fit` or `.partial_fit` call occurs in this phase. Completed training-inference and per-feature checkpoints can be reused. The input manifest and final integrity report protect pre-existing baseline artifacts. All requested CSVs reside in this directory; extra support tables retain threshold-grid values, reliability-bin counts, family ranking, full group summaries and feature rankings.','',
        'Histograms use fixed 0.01 score bins normalized within group and log frequency. ECDF displays use up to 4,000 deterministic order-statistic points per group with a symlog score axis (linear below 1e-12) to reveal low-score structure; numerical score summaries use all records. ROC and PR curves use full saved predictions.','']
    for model in MODEL_NAMES:
        lines += [f'### {model} figures','',f'![Score histograms](figures/score_distributions/{model}_histograms.png)',
                  f'![Score ECDF](figures/score_distributions/{model}_ecdf.png)',f'![ROC](figures/roc_curves/{model}.png)',
                  f'![Precision recall](figures/pr_curves/{model}.png)',f'![Validation calibration](figures/calibration/{model}.png)',
                  f'![Post-hoc test thresholds](figures/threshold_curves/{model}.png)','']
    (OUT/'phase4a_temporal_generalization_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    init();guard(full=True)
    from .run import CACHE
    cache=read(CACHE/'training_cache.json')
    for key in ('x','y'):
        require(sha256(cache[key+'_path'])==cache[key+'_sha256'],'Training input cache changed')
    json_write(OUT/'integrity_verification.json',dict(status='PASS',baseline_artifacts_unchanged=True,models_refitted=False,
        preprocessing_modified=False,split_modified=False,thresholds_selected_from='validation only',
        implementation_sha256={p.name:sha256(p) for p in (OUT.parents[2]/'src/phase4a').glob('*.py')}))
    print('PHASE 4A REPORT COMPLETE; original baseline integrity PASS',flush=True)

if __name__=='__main__': run()

