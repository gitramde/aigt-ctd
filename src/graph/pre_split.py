"""Read-only cleaned-cohort timeline and explicitly unapproved split proposal."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from src.baseline.common import csv_write, json_write, require, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/graph_v1/pre_split'
FILENAME = 'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'


def partition_rows(frame, boundaries):
    """Half-open chronological partitions; no row sampling or label stratification."""
    a, b = map(pd.Timestamp, boundaries)
    return np.where(frame.timestamp < a, 0, np.where(frame.timestamp < b, 1, 2))


def summarize_split(frame, boundaries):
    codes = partition_rows(frame, boundaries)
    rows = []
    for code, name in enumerate(('train', 'validation', 'test')):
        part = frame.loc[codes == code]
        malicious = int(part.target_binary.sum())
        # Membership identity is (fixed source file, one-based source_row),
        # in frozen chronological timestamp/source_row order, int64 little endian.
        digest = hashlib.sha256(np.asarray(part.source_row, dtype='<i8').tobytes()).hexdigest()
        rows.append(dict(partition=name, rows=len(part), benign=len(part)-malicious,
                         malicious=malicious, malicious_fraction=malicious/len(part) if len(part) else None,
                         timestamp_min=str(part.timestamp.min()), timestamp_max=str(part.timestamp.max()),
                         source_row_sequence_sha256=digest))
    return rows


def table(rows, columns):
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(str(r[c]) for c in columns)+' |' for r in rows])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'results/baseline_v1/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = next(r for r in manifest['files'] if r['filename'] == FILENAME)
    before = {}
    for kind in ('raw', 'cleaned'):
        path = Path(source[kind+'_path'])
        print('Verifying', kind, path.name, flush=True)
        require(sha256(path) == source[kind+'_sha256'], kind+' checksum differs from frozen manifest')
        before[kind] = (path.stat().st_size, path.stat().st_mtime_ns)
    frame = pq.read_table(source['cleaned_path'], columns=['source_row','timestamp','target_binary','attack_label']).to_pandas()
    require(len(frame) == source['counts']['temporal_rows'], 'Cleaned count mismatch')
    require(frame.source_row.is_unique, 'Duplicate source membership')
    require(frame.timestamp.is_monotonic_increasing, 'Nonchronological cleaned data')
    require(frame.timestamp.notna().all() and (frame.timestamp.dt.strftime('%Y-%m-%d') == '2018-02-20').all(), 'Invalid date')
    require(set(frame.attack_label) == {'Benign','DDoS attacks-LOIC-HTTP'}, 'Unexpected families')
    require(np.array_equal(frame.target_binary, (frame.attack_label != 'Benign').astype(int)), 'Target mismatch')
    start = frame.timestamp.min().floor('min')
    end = frame.timestamp.max().floor('min') + pd.Timedelta(minutes=1)
    windows = pd.date_range(start, end, freq='min', inclusive='left')
    raw_count = source['counts']['raw_rows']
    lookup = np.full(raw_count+1, -1, dtype=np.int32)
    minute = ((frame.timestamp-start).dt.total_seconds().to_numpy()//60).astype(np.int32)
    lookup[frame.source_row] = minute
    labels = np.full(raw_count+1, -1, dtype=np.int8)
    labels[frame.source_row] = frame.target_binary
    benign = np.bincount(minute[frame.target_binary.to_numpy()==0], minlength=len(windows))
    malicious = np.bincount(minute[frame.target_binary.to_numpy()==1], minlength=len(windows))
    sources = [set() for _ in windows]
    destinations = [set() for _ in windows]
    columns = ['Src IP','Dst IP','Src Port','Dst Port','Protocol','Label','Timestamp']
    reader = pcsv.open_csv(source['raw_path'], read_options=pcsv.ReadOptions(block_size=32*1024*1024),
        convert_options=pcsv.ConvertOptions(include_columns=columns, column_types={c:pa.string() for c in columns}))
    offset = 0
    matched = 0
    missing = {c:0 for c in columns[:5]}
    for batch in reader:
        ids = np.arange(offset+1, offset+1+len(batch)); offset += len(batch)
        require(offset <= raw_count, 'Raw row count overflow')
        keep = lookup[ids] >= 0
        retained = ids[keep]
        d = batch.to_pandas().loc[keep].copy()
        d['minute'] = lookup[retained]
        require(np.array_equal((d.Label != 'Benign').astype(int), labels[retained]), 'Raw/cleaned label join mismatch')
        parsed = pd.to_datetime(d.Timestamp, format='%d/%m/%Y %H:%M:%S', errors='raise')
        require(np.array_equal(((parsed-start).dt.total_seconds().to_numpy()//60).astype(int), lookup[retained]), 'Raw/cleaned timestamp join mismatch')
        for c in missing:
            missing[c] += int((d[c].isna() | d[c].str.strip().eq('')).sum())
        for index, group in d.groupby('minute', sort=False):
            sources[index].update(group['Src IP'].dropna())
            destinations[index].update(group['Dst IP'].dropna())
        matched += len(d)
        if offset//1_000_000 != (offset-len(batch))//1_000_000:
            print(f'Joined endpoint identifiers: {offset:,}/{raw_count:,} raw rows', flush=True)
    require(offset == raw_count and matched == len(frame), 'Incomplete source-row join')
    require(not any(missing.values()), 'Missing endpoint identifiers; inspect before proposing graphs')
    timeline = [dict(window_start=str(w), window_end_exclusive=str(w+pd.Timedelta(minutes=1)),
        benign_flow_count=int(benign[i]), malicious_flow_count=int(malicious[i]), total_flow_count=int(benign[i]+malicious[i]),
        unique_source_ips=len(sources[i]), unique_destination_ips=len(destinations[i])) for i,w in enumerate(windows)]
    csv_write(OUT/'feb20_attack_timeline.csv', timeline)
    active = np.flatnonzero(malicious)
    groups = np.split(active, np.flatnonzero(np.diff(active)>1)+1)
    episodes = [dict(start=str(windows[g[0]]), end_exclusive=str(windows[g[-1]]+pd.Timedelta(minutes=1)),
        active_minutes=len(g), malicious=int(malicious[g].sum()), benign=int(benign[g].sum())) for g in groups if len(g)]
    candidates = []
    for name, fractions in [('elapsed_80_10_10',(0.8,0.9)), ('elapsed_70_15_15',(0.7,0.85)), ('elapsed_60_20_20',(0.6,0.8))]:
        bounds = [str(start+pd.Timedelta(minutes=int(np.floor(len(windows)*f/10+0.5))*10)) for f in fractions]
        stats = summarize_split(frame, bounds)
        candidates.append(dict(name=name,boundaries=bounds,partitions=stats,
            both_classes_in_all_partitions=all(r['benign']>0 and r['malicious']>0 for r in stats),
            all_malicious_in_one_partition=sum(r['malicious']>0 for r in stats)==1))
    bounds = ['2018-02-20 10:30:00','2018-02-20 11:00:00']
    proposed = summarize_split(frame,bounds)
    require(all(r['benign']>0 and r['malicious']>0 for r in proposed), 'Proposed split lacks a class')
    for boundary in map(pd.Timestamp,bounds):
        require(boundary.minute % 10 == 0 and boundary.second == 0, 'Boundary not aligned with window sensitivities')
    for kind in ('raw','cleaned'):
        p = Path(source[kind+'_path'])
        require((p.stat().st_size,p.stat().st_mtime_ns)==before[kind], 'Input changed during audit')
    proposal = dict(status='PROPOSED_AWAITING_USER_REVIEW', approved=False, training_permitted=False, seed=42,
        purpose='within-day graph contribution; not unseen-attack or zero-day evaluation',
        source_file=FILENAME, source_manifest_sha256=sha256(manifest_path),
        inputs={k:source[k] for k in ('raw_path','raw_sha256','cleaned_path','cleaned_sha256')},
        cleaning='Reuse exact frozen temporal-eligible source_row membership; no recleaning or sampling',
        endpoint_recovery='Read raw identifiers only for source_row IDs retained in the verified cleaned parquet',
        source_row_definition='1-based CSV data record excluding header; (source_file, source_row) is unique',
        timestamp_policy='Frozen parsed timestamps; dataset-local naive times, no inferred timezone or AM/PM repair',
        primary_window_minutes=1, sensitivity_window_minutes=[5,10], window_alignment='calendar clock, half-open [start,end)',
        boundaries=bounds, partition_rules=dict(train='timestamp < 2018-02-20 10:30:00',
            validation='2018-02-20 10:30:00 <= timestamp < 2018-02-20 11:00:00',test='timestamp >= 2018-02-20 11:00:00'),
        rationale='Timeline-informed proposal: all partitions include both classes and a portion of the main attack period; boundaries align with 1/5/10-minute windows. Not selected from model performance.',
        partitions=proposed, membership_hash_encoding='SHA256 of source_row int64 little-endian sequence in frozen chronological order, scoped to source_file',
        simple_chronological_candidates=candidates, attack_episodes=episodes, endpoint_missing_counts=missing,
        timeline_sha256=sha256(OUT/'feb20_attack_timeline.csv'), software_versions=dict(numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__),
        next_stage_requirements=['Explicit split review/approval before preprocessing or training',
            'Fit imputation/scaling only on approved Feb-20 TRAIN; do not reuse baseline_v1 fitted preprocessing',
            'Keep identifiers separate; raw IPs, labels and timestamps are not numeric model features',
            'Causal node history and message-passing neighborhoods must exclude later flows, including later flows in the same snapshot',
            'Declare tie handling and historical state policy before fitting',
            'Matched edge MLP and GATv2 use identical targets, features, splits and preprocessing'])
    json_write(OUT/'proposed_split.json',proposal)
    flat = [dict(candidate=c['name'], **r) for c in candidates for r in c['partitions']]
    summary = '# Feb-20 pre-split review — no model training\n\n'
    summary += f"Status: **PROPOSED, awaiting review**. No preprocessing was fitted, graph model trained, or test model performance measured. Seed 42 is reserved for development.\n\nVerified frozen raw and cleaned SHA-256 checksums. The cleaned cohort has {len(frame):,} flows: {int(benign.sum()):,} benign and {int(malicious.sum()):,} DDoS attacks-LOIC-HTTP. Exactly {raw_count-len(frame):,} raw rows are excluded by existing cleaning. All {len(windows)} minutes from {start} through {end} (exclusive) are included in the timeline, including any empty minutes.\n\n"
    summary += 'The cleaned parquet excludes IP metadata. Unique endpoint counts therefore come from a read-only join to the original CSV using retained source_row IDs; labels and minute timestamps were cross-checked. Excluded raw rows never enter the timeline. Source IP, destination IP, source port, destination port and protocol are present with no missing values in retained rows. Raw identifier values are not encoded as numeric features.\n\n'
    summary += '## Attack timeline\n\n'+table(episodes,['start','end_exclusive','active_minutes','benign','malicious'])+'\n\n'
    summary += 'Times above are the existing audited, parsed timestamp values. The early 01:14 attack cluster is retained as recorded; no timezone or AM/PM correction is inferred. The source provides second-resolution timestamps, so equal-time causal ordering must be explicitly specified before graph training.\n\n'
    summary += '## Simple elapsed-time split checks\n\nBoundaries are rounded to the nearest ten-minute clock boundary so none of the planned window sizes straddles a split. Fractions refer to elapsed time, not equal flow counts.\n\n'
    summary += table(flat,['candidate','partition','rows','benign','malicious'])+'\n\n'
    summary += 'The 80/10/10 candidate leaves test without malicious flows. Earlier training cutoffs in the 70/15/15 and 60/20/20 candidates retain only the 797-example early attack cluster in training. None puts all malicious traffic in one partition, but class presence alone does not ensure a useful protocol. No random or stratified split has been substituted.\n\n'
    summary += '## Proposed chronological split for review\n\nTrain: before **10:30**; validation: **10:30–11:00**; test: **11:00 onward**, all on 2018-02-20. Boundaries are half-open and aligned with the predeclared primary 1-minute and sensitivity 5/10-minute windows. Every eligible flow belongs to exactly one partition.\n\n'
    summary += table(proposed,['partition','timestamp_min','timestamp_max','rows','benign','malicious','malicious_fraction'])+'\n\n'
    summary += 'This is an explicitly timeline-informed proposal, not a finalized split. It gives all three partitions observations from the main attack period, but evaluates continuation of the same within-day attack episode, with potentially recurring endpoints and correlated adjacent windows. It cannot establish generalization to unseen attacks, independent incidents or other days. Different partition durations and class prevalence must remain visible in later results.\n\n'
    summary += '## Review gate and next-stage constraints\n\nThe Phase 6 request says **“STOP BEFORE MODEL TRAINING”** and **“Do not train GATv2 until this split has been reviewed.”** Review the proposed 10:30/11:00 boundaries before proceeding. No split is approved automatically. Membership hashes in proposed_split.json identify the proposed source-row sets in chronological order; they are not approval or fitted preprocessing artifacts.\n\n'
    summary += 'After approval, fit preprocessing exclusively on the approved Feb-20 train partition. The earlier baseline_v1 preprocessing includes future Feb-20 rows and must not be reused. Node aggregates alone being historical is insufficient: message passing must also exclude future flows within a target snapshot. Declare this causal graph construction, timestamp-tie policy and history-state handling before training. The matched edge MLP must share target edges, eligible flow features, preprocessing and partitions. Phase 4 test metrics are not comparable to this cohort.\n\n'
    summary += 'Reproduce with `python -m src.graph.pre_split`. This command reads verified sources and writes only results/graph_v1/pre_split; it does not construct graphs, fit preprocessing or train models.\n'
    (OUT/'feb20_attack_timeline_summary.md').write_text(summary,encoding='utf-8')
    print(table(proposed,['partition','rows','benign','malicious']),flush=True)
    print('STOP: split review required; no training performed.',flush=True)


if __name__ == '__main__':
    run()
