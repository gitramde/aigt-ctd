"""Reconcile Phase 2 reports, prior source controls, and window rollups."""
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results'


def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if name=='timestamp_anomalies.csv':
            kept=[]
            reversals=Counter()
            for r in reader:
                if r['anomaly_type']=='backward_transition':
                    assert float(r['backward_seconds'])>0 and int(r['count'])>0
                    reversals[r['filename']]+=int(r['count'])
                else:
                    kept.append(r)
            for r in kept:
                if r['anomaly_type']=='file_summary':
                    assert reversals[r['filename']]==int(r['backward_transition_count'])
            return kept
        return list(reader)


def main():
    attack=read('attack_distribution.csv')
    cls=[r for r in attack if r['scope']=='file_class']
    old=read('class_by_file.csv')
    assert {(r['filename'],r['label']):int(r['count']) for r in cls} == {(r['filename'],r['label']):int(r['count']) for r in old}
    sizes={}
    for r in cls:
        sizes[r['filename']]=int(r['denominator_rows'])
    assert sum(sizes.values())==16233002
    for name in sizes:
        assert sum(int(r['count']) for r in cls if r['filename']==name)==sizes[name]
    oldquality=read('data_quality_report.csv')
    for report,metric in [('nan_analysis.csv','nan_values'),('infinity_analysis.csv','infinite_values')]:
        rows=read(report)
        for r in rows:
            assert abs(float(r['percentage'])-100*int(r['count'])/int(r['denominator_rows']))<1e-10
            if r['scope']=='file_column':
                assert int(r['count'])==sum(int(d['count']) for d in rows if d['scope']=='file_class' and (d['filename'],d['column_name'],d['kind'])==(r['filename'],r['column_name'],r['kind']))
        for oldrow in oldquality:
            if oldrow.get('scope','column')!='column': continue
            observed=sum(int(r['count']) for r in rows if r['scope']=='file_column' and (r['filename'],r['column_name'])==(oldrow['filename'],oldrow['column_name']))
            assert observed==int(oldrow[metric]),(report,oldrow,observed)
    duplicates=read('duplicate_analysis.csv')
    for r in duplicates:
        assert int(r['all_occurrences_in_duplicate_groups'])==int(r['extra_occurrences'])+int(r['duplicate_groups'])
        assert int(r['all_occurrences_in_duplicate_groups'])<=int(r['class_rows'])
    assert sum(int(r['extra_occurrences']) for r in duplicates if r['scope']=='file_total')==410763
    for r in duplicates:
        if r['scope']=='file_total':
            previous=next(x for x in oldquality if x['filename']==r['filename'])
            assert int(r['extra_occurrences'])==int(previous['duplicate_records'])
    anomalies=read('timestamp_anomalies.csv')
    invalid=Counter()
    for r in anomalies:
        if r['anomaly_type']=='unparseable_timestamp': invalid[r['filename']]+=int(r['count'])
    assert sum(invalid.values())==59
    epoch_count=sum(int(r['count']) for r in anomalies if r['anomaly_type']=='year_1970')
    assert epoch_count==14
    targets={}
    for r in anomalies:
        if r['anomaly_type'] in ('unparseable_timestamp','year_1970','date_mismatch'):
            targets.setdefault(r['filename'],{})[int(r['source_line'])]=r
    checked=0
    for name,locations in targets.items():
        last_line=max(locations)
        # These audited source files contain one physical line per CSV record.
        with (ROOT/'data/raw/Processed Traffic Data for ML Algorithms'/name).open(encoding='utf-8-sig',newline='') as f:
            header=next(csv.reader([next(f)]))
            ti,li=header.index('Timestamp'),header.index('Label')
            for line_number,line in enumerate(f,2):
                if line_number in locations:
                    row=next(csv.reader([line]))
                    r=locations[line_number]
                    assert row[ti]==r['timestamp'] and row[li]==r['label']
                    if r['anomaly_type']=='unparseable_timestamp': assert row==header
                    checked+=1
                if line_number>=last_line: break
    assert checked==73
    windows=read('temporal_window_analysis.csv')
    for name,n in sizes.items():
        for width in (1,5,10):
            selected=[r for r in windows if r['filename']==name and int(r['window_minutes'])==width]
            assert sum(int(r['flows']) for r in selected)==n-invalid[name]
            for r in selected:
                assert sum(json.loads(r['class_counts_json']).values())==int(r['flows'])
                if r['pair_analysis']=='available':
                    assert int(r['extra_flows_same_pair'])+int(r['unique_directed_pairs'])==int(r['flows'])
                    assert int(r['repeated_directed_pairs'])<=int(r['unique_directed_pairs'])
                    assert int(r['flows_in_repeated_pairs'])==int(r['extra_flows_same_pair'])+int(r['repeated_directed_pairs'])
            if width>1:
                rolled=Counter()
                for r in windows:
                    if r['filename']==name and r['window_minutes']=='1':
                        t=datetime.fromisoformat(r['window_start'])
                        rolled[t.replace(minute=t.minute//width*width).isoformat()]+=int(r['flows'])
                assert dict(rolled)=={r['window_start']:int(r['flows']) for r in selected}
    feb=read('feb20_graph_audit.csv')
    for r in feb:
        if r['section']=='window_summary':
            vals=[int(x['flows']) for x in windows if x['filename']==r['filename'] and x['window_minutes']==r['window_minutes']]
            assert len(vals)==int(r['window_count'])
            assert abs(sum(vals)/len(vals)-float(r['mean']))<1e-8
    print('PASS: class/file controls, every quality column, duplicates, all 73 anomalous source records, percentages, window totals and 1-to-5/10-minute rollups.')
    print(json.dumps({'rows':sum(sizes.values()),'files':len(sizes),'year_1970':epoch_count,'invalid_timestamps':59,'exact_extra_duplicates':410763}))


if __name__=='__main__': main()
