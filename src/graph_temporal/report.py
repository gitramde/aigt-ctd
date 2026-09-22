"""Coverage stop report or matched-cohort contribution report."""
from .prepare import *
import csv

def table(rows,columns):
    def fmt(v):
        if isinstance(v,float): return f'{v:.3e}' if 0<abs(v)<1e-6 else f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in rows])

def protected_check():
    items=read(OUT/'metrics/protected_artifacts.json')
    for item in items:require(sha256(item['path'])==item['sha256'],'Prior artifact changed: '+item['path'])
    return len(items)

LIMITS=('Membership remains the approved Feb-20 flow-start split: TRAIN before 10:30, validation 10:30 to before 11:00, '
    'test from 11:00 onward. Some TRAIN and validation flows complete after the next partition starts. '
    'No cross-partition graph or temporal inputs are permitted. This remains an offline start-time-split benchmark, '
    'not a strict deployment replay; no completion-time purge or embargo was silently introduced. '
    'Labels are used for training targets and coverage diagnostics, never history construction or graph inputs. '
    'This is Benign versus DDoS attacks-LOIC-HTTP within-day graph-temporal contribution development, '
    'not unseen-threat, zero-day or multiclass detection. Prior test results were already observed; '
    'no untouched confirmatory test or statistical significance is claimed. Seed 42 only. '
    'No anomaly branch, risk fusion, explainability or five-seed experiments were added.\n')

def coverage_text():
    with (OUT/'temporal_history_coverage.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    cols=['partition','label','targets','pair_ge1_percent','pair_ge4_percent','pair_ge8_percent',
        'source_ge8_percent','destination_ge8_percent','previous_3_slots_complete_percent','previous_7_slots_complete_percent']
    text=('Each historical observation is one Phase-6 sampled target flow represented by its own-minute frozen GAT source/destination embeddings '
        'and complete flow vector. For each current target, each previous calendar minute supplies at most one token: '
        'latest completed same directed pair first, otherwise latest same-source observation, otherwise latest same-destination observation. '
        'Completion must be strictly before the current second-30 graph cutoff. Completion ties use source row. '
        'The fallback follows a role-consistent endpoint but may change the interaction partner; it is not pure pair history. '
        'Only sampled target observations are in this temporal history; unsampled flows still enter the unchanged Phase-6 node aggregates and messages.\n\n'
        'L includes the current target, so L=4 uses the current step plus three preceding calendar minutes and L=8 plus seven. '
        'Missing minutes are masked, not compressed into adjacent unrelated flows. History counts count distinct earlier occupied minute steps '
        'anywhere within the same partition; recent-slot coverage separately measures the actual contiguous L=4/8 input. '
        'Coverage percentages below use all target rows as denominator.\n\n')
    amendment=OUT/'configs/coverage_gate_amendment.json'
    if amendment.exists():
        text+=('The initial diagnostic gate required at least 50% of TRAIN targets to have every L4/L8 historical slot filled. '
            'L8 reached 42.87% (74,150 complete examples), while L4 reached 50.26% (86,935). Before any fitting, '
            'the gate was revised to require at least 10,000 complete TRAIN contexts for each length, retaining masking for the rest. '
            'This avoids treating substantial support as unusable simply because a majority is not fully populated. '
            'The revision used TRAIN coverage, not test performance; it is a documented diagnostic-stage development decision, '
            'not preregistered evidence. The original failed gate and revision are retained in configs/coverage_gate_amendment.json.\n\n')
    return text+table(rows,cols)+'\n\nFull CSV includes source/destination >=1/4/8, counts, percentages, median/p90/max, and fallback token counts by class.\n\n'

def stopped():
    count=protected_check();gate=read(OUT/'metrics/coverage_gate.json')
    summary='# Phase 9 graph-temporal coverage gate: stopped before training\n\n'
    summary+='The pretraining coverage gate did not pass. No Transformer was trained and no new test prediction metrics were generated.\n\n'
    summary+='Predeclared label-free TRAIN gate: '+gate['rule']+'. This is a conservative operational support criterion, not a statistical test.\n\n'
    summary+=coverage_text()+LIMITS
    summary+='\nThe requested development_search, validation/test metric, ablation, progression and model runtime tables are intentionally not produced because their required training stage was stopped. No invented model results or arbitrary unrelated-flow sequences substitute for missing temporal support.\n'
    (OUT/'phase9_results_summary.md').write_text(summary,encoding='utf-8')
    (OUT/'integrity_report.md').write_text(f'# Phase 9 integrity\n\nPASS: historical Phase-6 hashes, approved split, completion-causal same-entity indices, and {count} prior artifact checks.\n\nCoverage gate: STOP. Training/evaluation: NOT RUN.\n',encoding='utf-8')
    json_write(OUT/'metrics/integrity_verification.json',dict(status='PASS',experiment_status='stopped_before_training',protected_files=count,
        source_sha256={p.name:sha256(p) for p in (ROOT/'src/graph_temporal').glob('*.py')}))

if __name__=='__main__': stopped()
