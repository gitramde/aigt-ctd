"""Verify conservation, exact partition membership, chronology and source integrity."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .clean import META
from .common import ROOT, json_write, load_config, require, sha256, target_alias
from .preprocess import load_plan
from .split import PARTITIONS


def row_keys(batch, file_codes):
    names=batch.column(batch.schema.get_field_index('source_file')).to_pylist()
    ids=batch.column(batch.schema.get_field_index('source_row')).to_numpy()
    timestamps=batch.column(batch.schema.get_field_index('timestamp')).to_numpy().astype('datetime64[us]').view('int64')
    codes=np.asarray([file_codes[name] for name in names],dtype=np.int64)
    return np.column_stack([codes,ids,timestamps]).astype('<i8').tobytes()


def run(config):
    reports,manifest,plan=load_plan(config)
    state=json.loads((reports/'preprocessing.json').read_text())
    require(state['fit_partition']=='train','Preprocessing was not training-only')
    require(state['training_rows']==plan['partitions']['train']['rows'],'Incorrect fitting population')
    require(state['split_plan_sha256']==sha256(reports/'split_plan.json'),'Preprocessing/split mismatch')
    require(state['source_manifest_sha256']==sha256(reports/'source_manifest.json'),'Preprocessing/source mismatch')
    require(sha256(reports/'cleaning_audit.csv')==manifest['cleaning_audit_sha256'],'Cleaning audit was changed')
    selected=set(state['numeric_columns']+state['categorical_columns'])
    require(selected<=set(manifest['features']),'Unknown selected feature')
    require(not any(target_alias(c) for c in selected),'Target-named feature survived')
    require(not selected.intersection(config['metadata_only_columns']+META),'Metadata leaked into model inputs')
    excluded={r['column'] for r in state['excluded_target_encodings']}
    require(not selected.intersection(excluded),'Detected target encoding survived')
    masks={f['filename']:np.zeros(f['counts']['raw_rows']+1,dtype=np.uint8) for f in manifest['files']}
    events=Counter()
    duplicate_links=[]
    with (reports/'cleaning_audit.csv').open(newline='',encoding='utf-8') as f:
        for event in csv.DictReader(f):
            name,source_row=event['source_file'],int(event['source_row'])
            mask=masks[name]
            require(0<source_row<len(mask) and mask[source_row]==0,'Repeated or invalid cleaning audit row')
            mask[source_row]=1 if event['disposition']=='remove' else 3
            events[(name,event['reason'])]+=1
            if event['reason']=='exact_duplicate':
                retained=int(event['retained_source_row'])
                require(0<retained<source_row,'Duplicate did not retain earliest source row')
                duplicate_links.append((name,retained))
    file_codes={f['filename']:i for i,f in enumerate(manifest['files'])}
    file_to_part={f['filename']:part for part in PARTITIONS for f in plan['partitions'][part]['files']}
    require(len(file_to_part)==len(manifest['files']),'Not every source file is assigned once')
    require(sum(len(plan['partitions'][p]['files']) for p in PARTITIONS)==len(file_to_part),'File appears in multiple partitions')
    key_hashes={part:hashlib.sha256() for part in PARTITIONS}
    observed_classes={part:Counter() for part in PARTITIONS}
    verified_files=[]
    for file in manifest['files']:
        name=file['filename']; part=file_to_part[name]
        print(f'Verify original hash and row coverage: {name}',flush=True)
        require(sha256(file['raw_path'])==file['raw_sha256'],f'Original source changed: {name}')
        require(sha256(file['cleaned_path'])==file['cleaned_sha256'],f'Cleaned file changed: {name}')
        require(events[(name,'repeated_header')]==file['counts'].get('repeated_headers',0),'Header audit mismatch')
        require(events[(name,'exact_duplicate')]==file['counts'].get('exact_nonheader_duplicates',0),'Duplicate audit mismatch')
        last_key=None
        count=0
        expected_day=np.datetime64(file['date'],'D').astype('int64')
        for batch in pq.ParquetFile(file['cleaned_path']).iter_batches(batch_size=config['batch_size'],columns=META):
            ids=batch.column(batch.schema.get_field_index('source_row')).to_numpy()
            times=batch.column(batch.schema.get_field_index('timestamp')).to_numpy().astype('datetime64[us]').view('int64')
            require(np.unique(ids).size==len(ids),'Duplicate source row within a cleaned batch')
            require((ids>0).all() and (ids<len(masks[name])).all(),'Source row out of range')
            require((masks[name][ids]==0).all(),'Excluded or repeated source row present in temporal data')
            masks[name][ids]=2
            require((np.diff(times)>=0).all(),'Rows not sorted chronologically')
            require((np.diff(ids)[np.diff(times)==0]>0).all(),'Nondeterministic timestamp tie order')
            require(last_key is None or last_key<(times[0],ids[0]),'Batch boundary is not chronological')
            last_key=(times[-1],ids[-1])
            require((times//86400000000==expected_day).all(),'Wrong-date row included in temporal data')
            labels=batch.column(batch.schema.get_field_index('attack_label')).to_pylist()
            binary=batch.column(batch.schema.get_field_index('target_binary')).to_numpy()
            require(np.array_equal(binary,np.asarray([int(v!=config['benign_label']) for v in labels])),'Incorrect target mapping')
            require(set(batch.column(batch.schema.get_field_index('source_file')).to_pylist())=={name},'Source filename mismatch')
            observed_classes[part].update(labels)
            key_hashes[part].update(row_keys(batch,file_codes))
            count+=len(batch)
        require(count==file['counts']['temporal_rows'],'Cleaned row count mismatch')
        require((masks[name][1:]!=0).all(),'A raw row is missing from both cleaning audit and temporal data')
        if 'non_temporal_path' in file:
            require(sha256(file['non_temporal_path'])==file['non_temporal_sha256'],'Quarantine changed')
            q=pq.read_table(file['non_temporal_path'])
            ids=q['source_row'].to_numpy()
            require(len(q)==file['counts']['temporal_exclusions'] and np.unique(ids).size==len(ids),'Quarantine count mismatch')
            require((masks[name][ids]==3).all(),'Quarantined rows do not match audit')
        verified_files.append(dict(filename=name,raw_sha256=file['raw_sha256'],raw_unchanged=True,
                                   all_raw_rows_accounted_for=True,chronology_verified=True,temporal_rows=count))
    for name,retained in duplicate_links:
        require(masks[name][retained] in (2,3),'Duplicate points to a removed first occurrence')
    partitions=[]
    for part in PARTITIONS:
        entry=plan['partitions'][part]
        require(sha256(entry['membership_path'])==entry['membership_sha256'],'Membership file changed')
        keys=hashlib.sha256()
        for batch in pq.ParquetFile(entry['membership_path']).iter_batches(batch_size=config['batch_size']):
            keys.update(row_keys(batch,file_codes))
        require(keys.hexdigest()==key_hashes[part].hexdigest(),'Membership differs from chronological feature rows')
        require(dict(observed_classes[part])==entry['class_counts'],'Class coverage mismatch')
        partitions.append(dict(partition=part,rows=entry['rows'],row_key_sha256=keys.hexdigest(),membership_matches_features=True))
    with (reports/'preprocessing_application_audit.csv').open(newline='',encoding='utf-8') as f:
        application=list(csv.DictReader(f))
    require({r['partition'] for r in application}==set(PARTITIONS),'Missing preprocessing application')
    for row in application:
        require(int(row['rows'])==plan['partitions'][row['partition']]['rows'],'Not all rows were transformed')
        require(row['preprocessing_sha256']==sha256(reports/'preprocessing.json'),'Different preprocessing applied across partitions')
        require(row['transformed_values_all_finite']=='True','Nonfinite preprocessing output')
    result=dict(status='PASS',raw_files_verified=verified_files,partitions=partitions,
                all_original_hashes_unchanged=True,all_raw_rows_accounted_for=True,
                preprocessing_fitted_on_train_only=True,same_preprocessing_applied_to_all_partitions=True,
                model_training_performed=False,totals=manifest['totals'],
                implementation_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    json_write(reports/'verification.json',result)
    print('PASS: original hashes, cleaning conservation, exclusion audit, chronology, disjoint membership, feature exclusions and frozen preprocessing.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline.json')
    run(load_config(parser.parse_args().config))
