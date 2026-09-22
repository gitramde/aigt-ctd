"""Produce a human-readable companion from the seven completed Phase 2 CSVs."""
import csv
import json
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parents[2]/'results'


def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if name=='timestamp_anomalies.csv':
            return [r for r in reader if r['anomaly_type']!='backward_transition']
        return list(reader)


def table(headers, rows):
    def cell(v):
        if isinstance(v,int): return f'{v:,}'
        if isinstance(v,float): return f'{v:,.3f}'
        return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])


def short(name): return name.replace('_TrafficForML_CICFlowMeter.csv','')


def main():
    nan=read('nan_analysis.csv'); inf=read('infinity_analysis.csv')
    dup=read('duplicate_analysis.csv'); attack=read('attack_distribution.csv')
    anomaly=read('timestamp_anomalies.csv'); feb=read('feb20_graph_audit.csv')
    windows=read('temporal_window_analysis.csv')
    lines=['# Phase 2: data quality and experiment design',
           'All ten original CSVs were scanned in full. No source records were modified or removed. No imputation, normalization, splitting, graph construction, or model training was performed.',
           '## Scope and definitions',
           '- Counts include every record after each initial header, including 59 repeated-header records. `Label` is unclassified, not malicious. Labels retain their exact source spelling.',
           '- NaN and infinity counts measure cells in the named column. Percentages are on a 0–100 scale. `dataset_column` uses all 16,233,002 rows; `file_column` uses all rows in that file; `file_class` uses rows of that class in that file. Summary scopes overlap: never add summaries to detail.',
           '- NaN means an explicit NaN token, case-insensitive and allowing a sign/whitespace. Infinity signs are reported separately. The per-column counts reconcile against the earlier full numeric audit. Blank/null markers are separate concepts; the earlier audit found none.',
           '- Duplicates are additional occurrences after the first equal full parsed CSV record within a file. Fingerprints identify candidates; a second full scan verifies original field tuples, including label and timestamp. Numeric values and whitespace are not canonicalized. Cross-file duplicates are outside this definition and the requested 410,763 count.',
           '- `duplicate_groups` counts distinct repeated records. `all_occurrences_in_duplicate_groups` includes the first occurrences, so it equals groups plus extra occurrences. Repeated headers contribute to duplicate counts and remain unclassified.',
           '- Timestamps use day-first `DD/MM/YYYY HH:MM:SS`. The timezone and AM/PM are not supplied. No 12-hour correction or timezone conversion is inferred. Source-order reversals are diagnostics, not proof of bad flow records; capture/export order may differ from event time.',
           '- Windows are fixed, non-overlapping intervals `[start, end)`, aligned to clock-minute multiples of 1, 5, or 10. Rows are assigned by recorded timestamp. Duplicate flows remain included. These are timestamp-bin counts, not concurrent flows or flows apportioned by duration.',
           '- Empty intervals between the first and last observation on each observed date are included as zero-count windows. No windows are invented outside this span, and the gap between 1970 and 2018 is not bridged. Invalid timestamps cannot be assigned a window; they remain in the anomaly and class reports. Valid 1970 timestamps have their own flagged date windows.',
           '- A pair is ordered `(Src IP, Dst IP)`, ignoring ports and protocol. Reverse-direction pairs are distinct. Recurrence can reflect exact duplicates or distinct flows between the same endpoints. Pair counts summed across windows are pair-window counts, not global pair cardinalities. Endpoint fields are available only on February 20.',
           '## Reports',
           table(['File','Contents'],[
               ('[nan_analysis.csv](nan_analysis.csv)','Dataset/column, file/column, and file/class counts and percentages; affected labels/files.'),
               ('[infinity_analysis.csv](infinity_analysis.csv)','The same breakdown separately for positive and negative infinity, including zero negative counts.'),
               ('[duplicate_analysis.csv](duplicate_analysis.csv)','File/class detail, file totals, dataset class totals and benign/malicious/unclassified totals.'),
               ('[timestamp_anomalies.csv](timestamp_anomalies.csv)','Every wrong-date or unparseable record; three preceding/following records and ranges; all backward transitions grouped by successive parseable timestamp pair and current label; file summaries.'),
               ('[attack_distribution.csv](attack_distribution.csv)','Every observed file/class and dataset/class count, status and percentage.'),
               ('[feb20_graph_audit.csv](feb20_graph_audit.csv)','February 20 scalar audit, classes, protocols, window summaries and every 1/5/10-minute window.'),
               ('[temporal_window_analysis.csv](temporal_window_analysis.csv)','Every 1/5/10-minute window across all files, class composition and February 20 directed-pair recurrence.')]),
           '## NaN and infinity',
           table(['Column','Type','Count','Dataset rows affected (%)'],[(r['column_name'],r['kind'],int(r['count']),float(r['percentage'])) for r in nan+inf if r['scope']=='dataset_column']),
           'The CSV detail identifies every affected file and class. Per-file nonzero column counts follow.',
           table(['File','Column','Type','Count','File rows affected (%)','Affected labels'],[
               (short(r['filename']),r['column_name'],r['kind'],int(r['count']),float(r['percentage']),', '.join(json.loads(r['affected_labels_json'])))
               for r in nan+inf if r['scope']=='file_column' and int(r['count'])]),
           '## Exact duplicates',
           table(['File','Extra occurrences','Groups','All participating rows','Extra duplicates (%)'],[
               (short(r['filename']),int(r['extra_occurrences']),int(r['duplicate_groups']),int(r['all_occurrences_in_duplicate_groups']),float(r['extra_duplicate_percentage']))
               for r in dup if r['scope']=='file_total']),
           table(['Status','Extra occurrences','Groups','All participating rows'],[
               (r['status'],int(r['extra_occurrences']),int(r['duplicate_groups']),int(r['all_occurrences_in_duplicate_groups'])) for r in dup if r['scope']=='dataset_status']),
           table(['Class','Extra occurrences','Groups'],[(r['label'],int(r['extra_occurrences']),int(r['duplicate_groups'])) for r in dup if r['scope']=='dataset_class']),
           '## Timestamp findings']
    epochs=[r for r in anomaly if r['anomaly_type']=='year_1970']
    lines.append(table(['File','Source line','Timestamp','Class','Previous 3 range','Following 3 range'],[
        (short(r['filename']),int(r['source_line']),r['timestamp'],r['label'],r['preceding_timestamp_min']+' to '+r['preceding_timestamp_max'],r['following_timestamp_min']+' to '+r['following_timestamp_max']) for r in epochs]))
    lines.append('There are '+str(sum(int(r['count']) for r in epochs))+' year-1970 records. Their surrounding records and expected-date file ranges are retained in the CSV. Their cause cannot be established from timestamps alone; no epoch correction is applied.')
    lines.append(table(['File','Minimum','Maximum','Unparseable','1970','Backward transitions'],[
        (short(r['filename']),r['timestamp_min'],r['timestamp_max'],int(r['invalid_timestamp_count']),int(r['year_1970_count']),int(r['backward_transition_count']))
        for r in anomaly if r['anomaly_type']=='file_summary']))
    backwards=sum(int(r['backward_transition_count']) for r in anomaly if r['anomaly_type']=='file_summary')
    lines.append(f'There are {backwards:,} backward transitions between successive parseable timestamps in source order. The detailed CSV is large because it retains these grouped transition diagnostics. Filter `anomaly_type=year_1970` for the 14 epoch-date records, `unparseable_timestamp` for the 59 header records, or `file_summary` for ten compact summaries.')
    lines.append('All 59 unparseable timestamp records contain repeated headers. The CSV preserves each record location and surrounding context. Backward transitions, repeated timestamps and the 01:00–12:59 clock range mean source row order must not be treated as a verified continuous chronology. Identical seconds are expected to permit multiple flows and are summarized separately, not classified as parsing errors.')
    lines.extend(['## Attack-label distribution',table(['File','Class','Status','Rows','File share (%)'],[
        (short(r['filename']),r['label'],r['status'],int(r['count']),float(r['percentage'])) for r in attack if r['scope']=='file_class']),
        table(['Class','Status','Rows','Dataset share (%)'],[(r['label'],r['status'],int(r['count']),float(r['percentage'])) for r in attack if r['scope']=='dataset_class']),
        '## February 20 graph audit',
        table(['Metric','Value'],[(r['metric'],r['value']) for r in feb if r['section']=='file_summary']),
        table(['Class','Status','Rows'],[(r['metric'],r['status'],int(r['value'])) for r in feb if r['section']=='class_distribution']),
        table(['Protocol (source code)','Rows','Share (%)'],[(r['metric'],int(r['value']),float(r['percentage'])) for r in feb if r['section']=='protocol_distribution']),
        'Every flow count by minute, 5 minutes and 10 minutes appears in both the February 20 report and the all-file temporal report. Summary percentiles use linear interpolation across all covered windows, including empty ones.',
        table(['Minutes','Windows','Min flows','Median flows','Mean flows','P95 flows','Max flows','Pair-window count','Repeated pair-window count','Flows in repeated pairs'],[
            (int(r['window_minutes']),int(r['window_count']),int(r['min']),float(r['median']),float(r['mean']),float(r['p95']),int(r['max']),int(r['unique_pairs_across_windows']),int(r['repeated_pairs_across_windows']),int(r['flows_in_repeated_pairs'])) for r in feb if r['section']=='window_summary'])])
    summary=[r for r in feb if r['section']=='window_summary']
    total=int(next(r['value'] for r in feb if r['section']=='file_summary' and r['metric']=='row_count'))
    feb_duplicates=next(r for r in dup if r['scope']=='file_total' and r['filename']==summary[0]['filename'])
    lines.append(f'February 20 contains only {int(feb_duplicates["extra_occurrences"]):,} extra exact duplicate records. Its repeated-pair counts therefore primarily measure different full flow records sharing the same ordered endpoints.')
    lines.append(table(['Minutes','Windows with attacks','Mixed benign/attack windows','Windows with repeated pairs','Flows in repeated pairs (%)'],[
        (int(s['window_minutes']),sum(int(r['malicious_count'])>0 for r in windows if r['filename']==s['filename'] and r['window_minutes']==s['window_minutes']),
         sum(int(r['malicious_count'])>0 and int(r['benign_count'])>0 for r in windows if r['filename']==s['filename'] and r['window_minutes']==s['window_minutes']),
         sum(int(r['repeated_directed_pairs'])>0 for r in windows if r['filename']==s['filename'] and r['window_minutes']==s['window_minutes']),
         100*int(s['flows_in_repeated_pairs'])/total) for s in summary]))
    lines.extend(['## Verification and reproduction',
        'Run from the repository root with Python 3.11 or later (standard library only):',
        '```powershell\n./.venv/Scripts/python.exe src/data/audit_phase2.py\n./.venv/Scripts/python.exe -m unittest discover -s src/data -p test_audit_phase2.py\n./.venv/Scripts/python.exe src/data/validate_phase2.py\n./.venv/Scripts/python.exe src/data/summarize_phase2.py\n```',
        'The auditor checks source size and modification time before/after each file, exact row width, class totals and all three window totals. This is metadata preservation verification, not cryptographic source provenance. The synthetic test forces fingerprint collisions and checks exact duplicates, signed infinities, NaNs, header anomalies, epoch context, directed pairs and window boundaries. The validator reconciles per-file labels, every quality column and duplicate counts against prior source controls, then independently rolls up minute counts into 5/10-minute counts and checks pair-count identities. The earlier controls are used only for counts, not their outdated label-status or timestamp inference.',
        'Totals reconcile to 16,233,002 rows, 59,721 NaN cells, 131,799 infinite cells and 410,763 extra exact duplicates.',
        '## Candidate-window recommendations based on recorded data'])
    for r in summary:
        w=int(r['window_minutes'])
        role={1:'Use as the primary candidate for preserving temporal detail and keeping each graph smaller.',5:'Use as the intermediate comparison for accumulating more repeated interactions with fewer snapshots.',10:'Use as the coarser comparison for longer interaction accumulation, accepting larger graphs and less temporal detail.'}[w]
        attacks=sum(int(x['malicious_count'])>0 for x in windows if x['filename']==r['filename'] and x['window_minutes']==r['window_minutes'])
        lines.append(f'- **{w} minute{"s" if w>1 else ""}:** {role} The dataset yields {int(r["window_count"]):,} windows, {float(r["median"]):,.0f} median flows, {int(r["max"]):,} peak flows and {100*int(r["flows_in_repeated_pairs"])/total:.2f}% of flows belonging to repeated directed pairs within a window. There are {attacks} attack-containing windows; all also contain benign flows.')
    lines.append('These are candidate design choices, not measured model performance or a validated optimal window. February 20 has only the observed attack class listed above, so its graph-window statistics cannot establish graph behavior for attack classes found only in the other nine files. Windows describe the recorded clock; timestamp ambiguities remain unresolved.')
    (OUT/'phase2_audit_notes.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote results/phase2_audit_notes.md')


if __name__=='__main__': main()
