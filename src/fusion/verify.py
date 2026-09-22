"""Audit completed CSVs against count identities and frozen Phase-7 controls."""
from .run import OUT, AE, rows, read, require, json_write, sha256

def run():
    primary=rows(OUT/'benign_only_thresholds.csv')
    require(len(primary)==24,'Missing primary operating points')
    for row in primary:
        require(float(row['observed_validation_fpr'])<=float(row['target_fpr']),'Primary cap violation')
    for part,n in [('validation',1652943),('test',1374143)]:
        metrics=rows(OUT/f'{part}_metrics.csv')
        require(len(metrics)==39,'Missing evaluation settings')
        for row in metrics:
            tp,fp,tn,fn=[int(row[k]) for k in ('true_positives','false_positives','true_negatives','false_negatives')]
            require(tp+fp+tn+fn==n,'Confusion matrix cohort mismatch')
            require(abs(float(row['recall'])-tp/(tp+fn))<1e-12,'Recall/count mismatch')
        family=rows(OUT/'family_metrics.csv')
        for row in metrics:
            matched=[f for f in family if all(f[k]==row[k] for k in ('partition','method','protocol','criterion'))]
            require(sum(int(f['detected']) for f in matched)==int(row['true_positives']),'Family detections mismatch')
    for row in rows(OUT/'rf_ae_detection_overlap.csv'):
        require(sum(int(row[k]) for k in ('rf_only','ae_only','both','neither'))==int(row['N']),'Overlap does not partition family')
    comparison=rows(OUT/'fusion_comparison_1pct.csv')
    old=rows(AE/'supervised_vs_anomaly_comparison.csv')
    # Frozen standalone operating points must reproduce the already published metrics.
    for method,oldname in [('random_forest','random_forest'),('autoencoder','autoencoder_B')]:
        current=next(r for r in comparison if r['method']==method)
        prior=next(r for r in old if r['model']==oldname)
        for key in ('actual_test_fpr','aggregate_malicious_recall','precision','f1','AP','Bot_recall','Infilteration_recall'):
            require(abs(float(current[key])-float(prior[key]))<1e-12,'Frozen standalone result changed: '+key)
    manifest=read(OUT/'input_integrity_manifest.json')
    for item in manifest['protected']:
        require(sha256(item['path'])==item['sha256'],'Protected artifact changed')
    json_write(OUT/'output_verification.json',dict(status='PASS',checks=['24 benign-only operating points','39 settings per partition',
        'full-cohort confusion counts','family counts reconcile','overlap counts reconcile','frozen standalone metrics reproduce Phase7',
        'all protected artifact hashes unchanged']))
    print('Phase 8 output verification PASS')

if __name__=='__main__':
    run()
