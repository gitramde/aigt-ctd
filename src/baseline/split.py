"""Choose and freeze a whole-date chronological split after the class/date report."""
import argparse
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from .clean import META
from .common import ROOT, csv_write, json_write, load_config, require, sha256

PARTITIONS = ('train', 'validation', 'test')


def context(config):
    reports = Path(config['report_dir'])
    manifest = json.loads((reports / 'source_manifest.json').read_text())
    require(manifest['status'] == 'cleaning_complete', 'Cleaning is incomplete')
    require(json.loads((reports / 'run_config.json').read_text()) == config, 'Configuration changed since cleaning')
    require(sha256(reports / 'run_config.json') == manifest['config_sha256'], 'Configuration hash mismatch')
    require(sha256(reports / 'pre_split_class_distribution.csv') == manifest['pre_split_report_sha256'], 'Pre-split report changed')
    return reports, manifest


def choose_split(distribution, spec, benign_label='Benign'):
    dates = sorted(distribution)
    candidates = []
    total = sum(sum(v.values()) for v in distribution.values())
    for i in range(spec['minimum_train_dates'], len(dates)):
        for j in range(i + spec['minimum_validation_dates'], len(dates)-spec['minimum_test_dates']+1):
            groups = (dates[:i], dates[i:j], dates[j:])
            labels = [sum((distribution[d] for d in group), Counter()) for group in groups]
            if any(c[benign_label] < spec['minimum_benign_per_partition'] or sum(c.values())-c[benign_label] < spec['minimum_malicious_per_partition'] for c in labels):
                continue
            counts = [sum(c.values()) for c in labels]
            attacks = [set(c)-{benign_label} for c in labels]
            error = sum(abs(count/total-target) for count,target in zip(counts,spec['target_fractions']))
            row = dict(train_end=groups[0][-1], validation_start=groups[1][0],
                       validation_end=groups[1][-1], test_start=groups[2][0], test_end=groups[2][-1],
                       training_attack_classes=len(attacks[0]), fraction_absolute_error=error)
            for part, group, counts_by_class, attack_set in zip(PARTITIONS,groups,labels,attacks):
                row.update({part+'_rows':sum(counts_by_class.values()), part+'_dates':len(group),
                            part+'_benign':counts_by_class[benign_label],
                            part+'_malicious':sum(counts_by_class.values())-counts_by_class[benign_label],
                            part+'_attack_classes_json':json.dumps(sorted(attack_set)),
                            part+'_unseen_attack_classes_json':json.dumps(sorted(attack_set-attacks[0]))})
            candidates.append(((-len(attacks[0]), error, row['train_end'], row['validation_end']), row, groups))
    require(bool(candidates), 'No chronological split meets the configured binary representation minimums.')
    candidates.sort(key=lambda item:item[0])
    report = [dict(rank=rank, selected=(rank==1), **row) for rank,(_,row,_) in enumerate(candidates,1)]
    return dict(zip(PARTITIONS,candidates[0][2])), report


def run(config):
    reports, manifest = context(config)
    require(not (reports/'split_plan.json').exists(), 'A split is already frozen; use a new output version.')
    distribution = {}
    with (reports/'pre_split_class_distribution.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['stage']=='temporal_eligible':
                distribution.setdefault(row['file_date'],Counter())[row['attack_class']]+=int(row['records'])
    dates, candidates = choose_split(distribution, config['split'], config['benign_label'])
    csv_write(reports/'split_candidates.csv', candidates)
    membership_folder = Path(config['data_dir'])/'membership'
    membership_folder.mkdir(exist_ok=False)
    partitions, coverage = {}, []
    all_classes = set().union(*(set(c) for c in distribution.values()))
    train_labels = set().union(*(set(distribution[d]) for d in dates['train']))
    previous_partition_max = None
    for part in PARTITIONS:
        files = [f for f in manifest['files'] if f['date'] in dates[part]]
        membership = membership_folder/f'{part}.parquet'
        writer, previous_time = None, None
        count = 0
        for file in files:
            for batch in pq.ParquetFile(file['cleaned_path']).iter_batches(batch_size=config['batch_size'],columns=META):
                first, last = batch.column(batch.schema.get_field_index('timestamp'))[0].as_py(), batch.column(batch.schema.get_field_index('timestamp'))[-1].as_py()
                require(previous_time is None or first>=previous_time, 'Source files are not chronological')
                previous_time=last
                if writer is None:
                    writer=pq.ParquetWriter(membership,batch.schema,compression='zstd')
                    start_time=first
                writer.write_batch(batch)
                count+=len(batch)
        require(writer is not None, f'Empty {part}')
        writer.close()
        require(previous_partition_max is None or previous_partition_max<start_time, 'Chronological partitions overlap')
        previous_partition_max=previous_time
        labels=sum((distribution[d] for d in dates[part]),Counter())
        require(count==sum(labels.values()), 'Membership/report count mismatch')
        partitions[part]=dict(dates=dates[part], files=[dict(filename=f['filename'],raw_sha256=f['raw_sha256'],
                              cleaned_path=f['cleaned_path'],cleaned_sha256=f['cleaned_sha256'],rows=f['counts']['temporal_rows']) for f in files],
                              timestamp_min=start_time.isoformat(),timestamp_max=previous_time.isoformat(),
                              interval_start_inclusive=dates[part][0]+'T00:00:00',
                              interval_end_exclusive=(date.fromisoformat(dates[part][-1])+timedelta(days=1)).isoformat()+'T00:00:00',
                              rows=count,class_counts=dict(sorted(labels.items())),
                              unseen_attack_classes=sorted(set(labels)-train_labels),
                              membership_path=str(membership),membership_sha256=sha256(membership))
        for label in sorted(all_classes):
            coverage.append(dict(partition=part,attack_class=label,status='benign' if label==config['benign_label'] else 'malicious',
                                 records=labels[label],present=labels[label]>0,seen_in_training=label in train_labels,
                                 fraction=labels[label]/count))
    csv_write(reports/'partition_class_distribution.csv',coverage)
    plan=dict(schema_version=1,target='binary',target_mapping={'Benign':0,'all_observed_attack_labels':1},
              source_manifest_sha256=sha256(reports/'source_manifest.json'),
              pre_split_class_report='pre_split_class_distribution.csv',
              pre_split_report_sha256=manifest['pre_split_report_sha256'],
              selection='Whole dates only. Require configured benign/malicious minimums. Maximize training attack-class coverage, then minimize L1 deviation from configured row fractions; chronological cutoffs break ties.',
              selection_spec=config['split'], partitions=partitions,
              exact_row_rule='Each listed source file: discard complete repeated headers; retain the earliest original source_row for each exact full parsed record; exclude nonmatching dates/invalid timestamps from temporal partitions. Every retained source_row is enumerated in the hashed membership Parquet.',
              ordering=['recorded_timestamp_ascending','source_filename_ascending','original_source_row_ascending'],
              timestamp_interpretation='Day-first, timezone unspecified. No AM/PM reconstruction; ordering is by recorded timestamps, not claimed physical capture chronology.',
              limitations=['Attack types occur on different dates; a globally chronological split cannot cover every attack type in every partition.',
                           'Unseen attack types remain positive binary targets and are explicitly reported; this is a temporal generalization evaluation, not a closed-set multiclass claim.',
                           'Feature statistics and preprocessing are not used to select date boundaries. Labels are used only for this predeclared cohort-design report.'],
              status='split_frozen')
    json_write(reports/'split_plan.json',plan)
    print(json.dumps({p:dict(rows=v['rows'],dates=v['dates'],unseen=v['unseen_attack_classes']) for p,v in partitions.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline.json')
    run(load_config(parser.parse_args().config))
