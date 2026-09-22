"""Read-only Phase 2 audit; no cleaning, splits, graph construction or training."""
import csv
import hashlib
import json
import math
import re
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results'
SPECIAL = re.compile(r'(?:^|\x1f)\s*[+-]?(?:nan|inf(?:inity)?)\s*(?:\x1f|$)', re.I)


def status(label):
    return 'benign' if label == 'Benign' else 'unclassified' if label == 'Label' or not label.strip() else 'malicious'


def write(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fingerprint(row):
    # Length-safe encoding matches exact parsed-field equality, verified on pass 2.
    return hashlib.blake2b('\x1f'.join(row).encode(), digest_size=16).digest()


def describe(values):
    a = sorted(values)
    i = (len(a)-1)*0.95
    p95 = a[math.floor(i)] + (a[math.ceil(i)]-a[math.floor(i)])*(i-math.floor(i))
    return dict(min=a[0], median=statistics.median(a), mean=statistics.mean(a), p95=p95, max=a[-1])


def audit(path):
    before = path.stat()
    labels, quality, times, time_labels = Counter(), Counter(), Counter(), Counter()
    seen, candidates, cache = set(), set(), {}
    anomalies, reversals = [], Counter()
    context = deque(maxlen=3)
    pending = []
    pairs = Counter()
    cardinalities = {c: set() for c in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port']}
    protocols = Counter()
    expected = datetime.strptime(re.search(r'(\d{2}-\d{2}-\d{4})', path.name)[1], '%d-%m-%Y').date()
    previous = None
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        columns = next(reader)
        li, ti = columns.index('Label'), columns.index('Timestamp')
        graph = 'Src IP' in columns and 'Dst IP' in columns
        indices = {c: columns.index(c) for c in cardinalities if c in columns}
        pi = columns.index('Protocol')
        for n, row in enumerate(reader, 1):
            assert len(row) == len(columns)
            assert all('\x1f' not in v for v in row)
            joined = '\x1f'.join(row)
            h = hashlib.blake2b(joined.encode(), digest_size=16).digest()
            if h in seen:
                candidates.add(h)
            else:
                seen.add(h)
            label, raw = row[li], row[ti]
            labels[label] += 1
            if SPECIAL.search(joined):
                for col, value in zip(columns, row):
                    v = value.strip().lower()
                    if v in {'nan', '+nan', '-nan'}:
                        quality[(col, 'nan', label)] += 1
                    elif v in {'inf', '+inf', 'infinity', '+infinity', '-inf', '-infinity'}:
                        quality[(col, 'negative_infinity' if v.startswith('-') else 'positive_infinity', label)] += 1
            if raw not in cache:
                try:
                    cache[raw] = datetime.strptime(raw, '%d/%m/%Y %H:%M:%S')
                except ValueError:
                    cache[raw] = None
            dt = cache[raw]
            for a in pending[:]:
                a['following'].append(dict(record=n, timestamp=raw, label=label))
                if len(a['following']) == 3:
                    pending.remove(a)
            anomaly_type = 'unparseable_timestamp' if dt is None else 'year_1970' if dt.year == 1970 else 'date_mismatch' if dt.date() != expected else None
            if anomaly_type:
                a = dict(filename=path.name, anomaly_type=anomaly_type, count=1, source_record=n,
                         source_line=reader.line_num, timestamp=raw, label=label, status=status(label),
                         preceding=list(context), following=[])
                anomalies.append(a)
                pending.append(a)
            if dt is not None:
                times[dt] += 1
                time_labels[(dt.replace(second=0), label)] += 1
                if previous and dt < previous:
                    reversals[(previous, dt, label)] += 1
                previous = dt
                if graph:
                    pairs[(dt.replace(second=0), row[indices['Src IP']], row[indices['Dst IP']])] += 1
            context.append(dict(record=n, timestamp=raw, label=label))
            if graph:
                for c, i in indices.items():
                    cardinalities[c].add(row[i])
                protocols[row[pi]] += 1
            if n % 500000 == 0:
                print(f'{path.name}: {n:,} rows', flush=True)
    del seen
    print(f'{path.name}: verifying exact duplicate candidates', flush=True)
    repeated = Counter()
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if fingerprint(row) in candidates:
                repeated[tuple(row)] += 1
    dup = Counter()
    for row, count in repeated.items():
        if count > 1:
            dup[(row[li], 'extra_occurrences')] += count - 1
            dup[(row[li], 'duplicate_groups')] += 1
            dup[(row[li], 'all_occurrences_in_duplicate_groups')] += count
    normal = [t for t in times if t.date() == expected]
    for a in anomalies:
        a['file_expected_date'] = str(expected)
        a['file_expected_date_min'] = min(normal).isoformat() if normal else ''
        a['file_expected_date_max'] = max(normal).isoformat() if normal else ''
        for key in ('preceding', 'following'):
            valid = [cache[x['timestamp']] for x in a[key] if cache[x['timestamp']] is not None]
            a[key + '_timestamp_min'] = min(valid).isoformat() if valid else ''
            a[key + '_timestamp_max'] = max(valid).isoformat() if valid else ''
            a[key + '_records_json'] = json.dumps(a.pop(key))
    # All backwards transitions are reported, grouped by exact adjacent times and current label.
    for (prev, curr, label), count in reversals.items():
        anomalies.append(dict(filename=path.name, anomaly_type='backward_transition', count=count,
                              timestamp=curr.isoformat(), previous_timestamp=prev.isoformat(),
                              backward_seconds=(prev-curr).total_seconds(), label=label, status=status(label)))
    anomalies.append(dict(filename=path.name, anomaly_type='file_summary', count=n,
                          timestamp_min=min(times).isoformat(), timestamp_max=max(times).isoformat(),
                          unique_timestamps=len(times), same_second_extra_flows=sum(times.values())-len(times),
                          invalid_timestamp_count=sum(1 for a in anomalies if a['anomaly_type']=='unparseable_timestamp'),
                          year_1970_count=sum(1 for a in anomalies if a['anomaly_type']=='year_1970'),
                          backward_transition_count=sum(reversals.values()), file_expected_date=str(expected)))
    windows = []
    graph_rows = []
    for width in (1, 5, 10):
        def bucket(t):
            return t.replace(minute=t.minute // width * width, second=0)
        flows, classes = Counter(), {}
        for (t, label), count in time_labels.items():
            b = bucket(t)
            flows[b] += count
            classes.setdefault(b, Counter())[label] += count
        # Fill gaps independently per observed date; do not bridge 1970 to 2018.
        for date in sorted({t.date() for t in flows}):
            day = [t for t in flows if t.date() == date]
            t, end = min(day), max(day)
            while t <= end:
                flows.setdefault(t, 0)
                t += timedelta(minutes=width)
        pair_stats = {}
        if graph:
            combined = Counter()
            for (t, src, dst), count in pairs.items():
                combined[(bucket(t), src, dst)] += count
            for (t, src, dst), count in combined.items():
                s = pair_stats.setdefault(t, Counter())
                s['unique_directed_pairs'] += 1
                s['repeated_directed_pairs'] += count > 1
                s['flows_in_repeated_pairs'] += count if count > 1 else 0
                s['extra_flows_same_pair'] += count-1
                s['max_flows_per_pair'] = max(s['max_flows_per_pair'], count)
            del combined
        for t, count in sorted(flows.items()):
            ls = classes.get(t, {})
            r = dict(filename=path.name, window_minutes=width, window_start=t.isoformat(),
                     window_end_exclusive=(t+timedelta(minutes=width)).isoformat(), flows=count,
                     benign_count=ls.get('Benign', 0), malicious_count=sum(v for k,v in ls.items() if status(k)=='malicious'),
                     class_counts_json=json.dumps(dict(sorted(ls.items()))),
                     date_matches_filename=t.date()==expected, pair_analysis='available' if graph else 'endpoints_unavailable')
            if graph:
                s = pair_stats.get(t, Counter())
                for k in ('unique_directed_pairs', 'repeated_directed_pairs', 'flows_in_repeated_pairs', 'extra_flows_same_pair', 'max_flows_per_pair'):
                    r[k] = s[k]
                r['repeated_pair_flow_percentage'] = 100*s['flows_in_repeated_pairs']/count if count else 0
            windows.append(r)
        if graph:
            graph_rows.append(dict(filename=path.name, section='window_summary', metric='flows_per_window',
                                   window_minutes=width, window_count=len(flows), **describe(list(flows.values())),
                                   repeated_pairs_across_windows=sum(s['repeated_directed_pairs'] for s in pair_stats.values()),
                                   unique_pairs_across_windows=sum(s['unique_directed_pairs'] for s in pair_stats.values()),
                                   flows_in_repeated_pairs=sum(s['flows_in_repeated_pairs'] for s in pair_stats.values()),
                                   extra_flows_same_pair=sum(s['extra_flows_same_pair'] for s in pair_stats.values())))
    if graph:
        metrics = dict(row_count=n, benign_count=labels['Benign'], malicious_count=sum(v for k,v in labels.items() if status(k)=='malicious'),
                       unclassified_count=sum(v for k,v in labels.items() if status(k)=='unclassified'),
                       timestamp_min=min(times).isoformat(), timestamp_max=max(times).isoformat())
        metrics.update({c.replace(' ', '_').lower()+'_cardinality': len(v) for c,v in cardinalities.items()})
        graph_rows.extend(dict(filename=path.name, section='file_summary', metric=k, value=v) for k,v in metrics.items())
        graph_rows.extend(dict(filename=path.name, section='class_distribution', metric=k, value=v, status=status(k)) for k,v in labels.items())
        graph_rows.extend(dict(filename=path.name, section='protocol_distribution', metric=k, value=v, percentage=100*v/n) for k,v in protocols.items())
        graph_rows.extend(dict(r, section='window_detail', metric='flows_per_window', value=r['flows']) for r in windows)
    after = path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert sum(labels.values()) == n
    for width in (1,5,10):
        assert sum(r['flows'] for r in windows if r['window_minutes']==width) == sum(times.values())
    print(f'DONE {path.name}: rows={n:,}; duplicates={sum(v for (k,m),v in dup.items() if m=="extra_occurrences"):,}', flush=True)
    return dict(filename=path.name, rows=n, labels=labels, quality=quality, duplicates=dup,
                anomalies=anomalies, windows=windows, graph=graph_rows,
                source_size=before.st_size, source_mtime_ns=before.st_mtime_ns)


def main():
    paths = sorted((ROOT/'data/raw/Processed Traffic Data for ML Algorithms').glob('*.csv'))
    assert len(paths)==10
    audits = [audit(p) for p in paths]
    total = sum(a['rows'] for a in audits)
    attack, duplicates, quality_rows = [], [], []
    for a in audits:
        for label, count in sorted(a['labels'].items()):
            attack.append(dict(scope='file_class', filename=a['filename'], label=label, status=status(label), count=count,
                               denominator_rows=a['rows'], percentage=100*count/a['rows']))
            duplicates.append(dict(scope='file_class', filename=a['filename'], label=label, status=status(label),
                                   class_rows=count, extra_occurrences=a['duplicates'][(label,'extra_occurrences')],
                                   duplicate_groups=a['duplicates'][(label,'duplicate_groups')],
                                   all_occurrences_in_duplicate_groups=a['duplicates'][(label,'all_occurrences_in_duplicate_groups')]))
        for (col,kind,label),count in sorted(a['quality'].items()):
            quality_rows.append(dict(scope='file_class', filename=a['filename'], column_name=col, kind=kind, label=label,
                                     status=status(label), count=count, denominator_rows=a['labels'][label], percentage=100*count/a['labels'][label]))
    for kindset,name in [({'nan'},'nan_analysis.csv'),({'positive_infinity','negative_infinity'},'infinity_analysis.csv')]:
        detail=[r for r in quality_rows if r['kind'] in kindset]
        rows=[]
        cols=sorted({r['column_name'] for r in detail})
        for col in cols:
            for kind in sorted(kindset):
                matching=[r for r in detail if r['column_name']==col and r['kind']==kind]
                count=sum(r['count'] for r in matching)
                rows.append(dict(scope='dataset_column',filename='ALL',column_name=col,kind=kind,count=count,
                                 denominator_rows=total,percentage=100*count/total,
                                 affected_labels_json=json.dumps(sorted({r['label'] for r in matching})),
                                 affected_files_json=json.dumps(sorted({r['filename'] for r in matching}))))
                for a in audits:
                    m=[r for r in matching if r['filename']==a['filename']]
                    c=sum(r['count'] for r in m)
                    rows.append(dict(scope='file_column',filename=a['filename'],column_name=col,kind=kind,count=c,
                                     denominator_rows=a['rows'],percentage=100*c/a['rows'],affected_labels_json=json.dumps(sorted({r['label'] for r in m}))))
        write(name,rows+detail)
    for label in sorted({r['label'] for r in attack}):
        count=sum(r['count'] for r in attack if r['label']==label and r['scope']=='file_class')
        attack.append(dict(scope='dataset_class',filename='ALL',label=label,status=status(label),count=count,denominator_rows=total,percentage=100*count/total))
    for scope,groupkey in [('file_total','filename'),('dataset_class','label'),('dataset_status','status')]:
        for value in sorted({r[groupkey] for r in duplicates if r['scope']=='file_class'}):
            m=[r for r in duplicates if r['scope']=='file_class' and r[groupkey]==value]
            row=dict(scope=scope,filename=value if groupkey=='filename' else 'ALL',label=value if groupkey=='label' else '',status=value if groupkey=='status' else '')
            row.update({k:sum(r[k] for r in m) for k in ('class_rows','extra_occurrences','duplicate_groups','all_occurrences_in_duplicate_groups')})
            duplicates.append(row)
    for r in duplicates:
        r['extra_duplicate_percentage']=100*r['extra_occurrences']/r['class_rows']
        r['comparison_scope']='exact parsed fields within each file; no cross-file deduplication'
    write('duplicate_analysis.csv',duplicates)
    write('attack_distribution.csv',attack)
    write('timestamp_anomalies.csv',[r for a in audits for r in a['anomalies']])
    write('feb20_graph_audit.csv',[r for a in audits for r in a['graph']])
    write('temporal_window_analysis.csv',[r for a in audits for r in a['windows']])
    totals=dict(rows=total,nan=sum(r['count'] for r in quality_rows if r['kind']=='nan'),
                infinity=sum(r['count'] for r in quality_rows if 'infinity' in r['kind']),
                duplicates=sum(r['extra_occurrences'] for r in duplicates if r['scope']=='file_class'))
    assert totals==dict(rows=16233002,nan=59721,infinity=131799,duplicates=410763), totals
    print(json.dumps(totals),flush=True)
    print('Seven Phase 2 reports written; all totals reconcile.',flush=True)


if __name__ == '__main__':
    main()
