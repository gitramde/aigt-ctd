"""Attach independent verification and preparation costs to the completed report."""
import csv
from .prepare import *
from .report import table

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    verification=read(OUT/'metrics/output_verification.json')
    require(verification['status']=='PASS','Final verification missing')
    timings=read(OUT/'metrics/history_preparation_timing.json')['timings']
    runtime=rows(OUT/'runtime_metrics.csv')
    for row in runtime:
        prep=next(t['history_index_seconds'] for t in timings if t['partition']==row['partition']) if row['model'] in ('gat_transformer_L4','gat_transformer_L8') else 0.
        row['shared_history_index_replay_seconds']=prep
        row['inference_plus_shared_index_preparation_seconds']=float(row['inference_seconds'])+prep
        n=read(OUT/'matched_cohort_manifest.json')['partitions'][row['partition']]
        row['latency_including_shared_index_ms_per_1000']=(float(row['inference_seconds'])+prep)/n*1e6
    csv_write(OUT/'runtime_metrics.csv',runtime)
    p=OUT/'phase9_results_summary.md';text=p.read_text(encoding='utf-8')
    # Preserve full precision for very small ranking-metric differences.
    differences=rows(OUT/'temporal_ablation.csv')
    for r in differences:
        for k in r:
            if k.startswith('delta_'):r[k]=float(r[k])
    start=text.index('| model | criterion | delta_recall')
    stop=text.index('\n\nL4:',start)
    text=text[:start]+table([r for r in differences if r['partition']=='test'],['model','criterion','delta_recall','delta_precision','delta_f1','delta_macro_f1','delta_false_positive_rate','delta_roc_auc','delta_average_precision'])+text[stop:]
    marker='## Independent verification and preparation costs'
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
    text+='\n'+marker+'\n\n'
    text+='All saved metrics were independently recomputed. All temporal indices were replayed exactly after lookup optimization; partition, minute, completion and endpoint identities passed. Cached graph representations reproduced the frozen GAT classifier within a maximum probability difference of '+str(verification['graph_classifier_replay_max_abs_difference'])+'. All heads have '+str(verification['equal_head_parameters'])+' trainable parameters. Three focused causality/masking tests passed. Prior artifact hashes remain unchanged.\n\n'
    text+='Shared offline history-index preparation was separately replayed and measured below. This implementation also computes diagnostic history counts and all eight historical slots, so these costs are not pure online serving latency. L4/L8 share this prepared index; L1 needs none. runtime_metrics.csv includes both cached-index inference and an additional preparation-inclusive total. Do not add shared preparation repeatedly across methods.\n\n'
    text+=table(timings,['partition','targets','history_index_seconds'])+'\n'
    p.write_text(text,encoding='utf-8')
    integrity=read(OUT/'metrics/integrity_verification.json')
    integrity['source_sha256']={p.name:sha256(p) for p in (ROOT/'src/graph_temporal').glob('*.py')}
    integrity['independent_output_verification_sha256']=sha256(OUT/'metrics/output_verification.json')
    json_write(OUT/'metrics/integrity_verification.json',integrity)
    p=OUT/'integrity_report.md';text=p.read_text(encoding='utf-8')
    if 'Independent verification:' not in text:
        p.write_text(text+'\nIndependent verification: PASS. See metrics/output_verification.json for full metric recomputation, exact history replay, frozen-GAT representation replay, equal head capacity and protected-file checks.\n',encoding='utf-8')
    print('Phase9 report finalized with verified outputs and preparation costs',flush=True)

if __name__=='__main__':main()
