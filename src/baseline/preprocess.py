"""Training-only fitting and immutable, batchwise preprocessing for all partitions."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from .clean import META
from .common import ROOT, csv_write, json_write, load_config, require, sha256, versions
from .split import PARTITIONS, context


def load_plan(config):
    reports, manifest = context(config)
    plan = json.loads((reports/'split_plan.json').read_text())
    require(plan['status']=='split_frozen', 'Split must be frozen before fitting')
    require(plan['source_manifest_sha256']==sha256(reports/'source_manifest.json'), 'Split/source provenance mismatch')
    return reports, manifest, plan


def iter_partition(config, plan, partition, columns=None):
    require(partition in PARTITIONS, f'Unknown partition: {partition}')
    for file in plan['partitions'][partition]['files']:
        yield from pq.ParquetFile(file['cleaned_path']).iter_batches(batch_size=config['batch_size'],columns=columns)


def numeric_matrix(batch, names):
    if not names:
        return np.empty((len(batch),0),dtype=np.float64)
    require(all(batch.schema.get_field_index(name)>=0 for name in names),'Required numeric feature is missing')
    return np.column_stack([batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                            for name in names]).astype(np.float64,copy=False)


def string_values(batch, name):
    require(batch.schema.get_field_index(name)>=0,f'Required feature is missing: {name}')
    return np.asarray(batch.column(batch.schema.get_field_index(name)).to_pylist(),dtype=object)


class FrozenPreprocessor:
    """Transform-only object. Learned arrays cannot be modified in place."""
    def __init__(self, state):
        self.numeric=tuple(state['numeric_columns'])
        self.categorical=tuple(state['categorical_columns'])
        self.output_names=tuple(state['output_feature_names'])
        self.means=np.asarray(state['numeric_imputation_values'],dtype=np.float64)
        self.centers=np.asarray(state['scaler_mean'],dtype=np.float64)
        self.scales=np.asarray(state['scaler_scale'],dtype=np.float64)
        self.levels={k:tuple(v) for k,v in state['categories'].items()}
        self.modes=dict(state['categorical_imputation_values'])
        for arr in (self.means,self.centers,self.scales):
            require(len(arr)==len(self.numeric) and np.isfinite(arr).all(),'Invalid preprocessing state')
            arr.setflags(write=False)
        require((self.scales>0).all(),'Invalid scaling denominators')
        require(len(self.output_names)==len(self.numeric)+sum(len(v) for v in self.levels.values()),'Feature schema mismatch')

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text()))

    def transform(self,batch,scaled=True):
        x=numeric_matrix(batch,self.numeric)
        require(not np.isinf(x).any(),'Unclean infinity passed to preprocessing')
        missing=np.isnan(x)
        x=np.where(missing,self.means,x)
        if scaled:
            x=(x-self.centers)/self.scales
        blocks=[x]
        unknown={}
        for name in self.categorical:
            values=string_values(batch,name)
            values=np.asarray([self.modes[name] if v is None else v for v in values],dtype=object)
            levels=self.levels[name]
            encoded=np.column_stack([values==v for v in levels]).astype(np.float64)
            unknown[name]=int(np.count_nonzero(encoded.sum(axis=1)==0))
            blocks.append(encoded)
        result=np.column_stack(blocks)
        require(np.isfinite(result).all(),'Nonfinite transformed values')
        return result,dict(imputed_numeric_cells=int(missing.sum()),unknown_categories=unknown)


def fit_training(factory,features,categorical,all_missing_fill=0.0):
    """The factory must produce TRAIN batches afresh for each of two fitting passes."""
    categorical=[c for c in categorical if c in features]
    numeric=[c for c in features if c not in categorical]
    counts=np.zeros(len(numeric),dtype=np.int64)
    sums=np.zeros(len(numeric),dtype=np.float64)
    category_counts={c:Counter() for c in categorical}
    encodings={c:{} for c in features}
    label_set=set()
    training_rows=0
    for batch in factory():
        x=numeric_matrix(batch,numeric)
        require(not np.isinf(x).any(),'Training data contains unclean infinity')
        valid=~np.isnan(x)
        counts+=valid.sum(axis=0)
        sums+=np.where(valid,x,0.0).sum(axis=0)
        labels=string_values(batch,'attack_label')
        label_set.update(labels)
        training_rows+=len(batch)
        for name in categorical:
            category_counts[name].update(v for v in string_values(batch,name) if v is not None)
        for name in features:
            mapping=encodings[name]
            if mapping is None:
                continue
            values=string_values(batch,name) if name in categorical else x[:,numeric.index(name)]
            # A direct target encoding must be low-cardinality and one-to-one.
            if name in categorical:
                values=np.asarray(['<MISSING>' if v is None else v for v in values],dtype=str)
            unique=np.unique(values)
            keys=['<MISSING>' if name not in categorical and np.isnan(v) else str(v) for v in unique]
            if len(unique)>64 or len(set(mapping)|set(keys))>64:
                encodings[name]=None
                continue
            for value,key in zip(unique,keys):
                mask=np.isnan(values) if name not in categorical and np.isnan(value) else values==value
                mapping.setdefault(key,set()).update(labels[mask])
            # A value shared by benign and malicious cannot be a direct encoding.
            if any('Benign' in labels_for_value and len(labels_for_value)>1 for labels_for_value in mapping.values()):
                encodings[name]=None
    require(training_rows>0,'Empty training partition')
    excluded=[]
    for name,mapping in encodings.items():
        if mapping is None:
            continue
        binary_codes={frozenset(int(label!='Benign') for label in labels) for labels in mapping.values()}
        binary=len(mapping)==2 and binary_codes=={frozenset([0]),frozenset([1])}
        multiclass=(len(mapping)==len(label_set)>1 and all(len(labels)==1 for labels in mapping.values()))
        if binary or multiclass:
            excluded.append(dict(column=name,stage='training_only_value_check',decision='exclude',
                                 reason='one_to_one_binary_target_encoding' if binary else 'one_to_one_attack_class_encoding',
                                 mapping_json=json.dumps({k:sorted(v) for k,v in sorted(mapping.items())})))
    drop={r['column'] for r in excluded}
    selected=[i for i,c in enumerate(numeric) if c not in drop]
    numeric=[numeric[i] for i in selected]
    counts=counts[selected]; sums=sums[selected]
    means=np.divide(sums,counts,out=np.full(len(counts),all_missing_fill,dtype=np.float64),where=counts>0)
    categorical=[c for c in categorical if c not in drop]
    categories={c:sorted(category_counts[c]) or ['<ALL_TRAIN_MISSING>'] for c in categorical}
    modes={c:sorted(category_counts[c],key=lambda v:(-category_counts[c][v],v))[0] if category_counts[c] else '<ALL_TRAIN_MISSING>' for c in categorical}
    require(bool(numeric),'No numeric features remain after leakage checks')
    scaler=StandardScaler()
    second_pass_rows=0
    for batch in factory():
        x=numeric_matrix(batch,numeric)
        scaler.partial_fit(np.where(np.isnan(x),means,x))
        second_pass_rows+=len(batch)
    require(second_pass_rows==training_rows,'Training partition changed between fitting passes')
    state=dict(schema_version=1,fit_partition='train',training_rows=training_rows,
               numeric_columns=numeric,categorical_columns=categorical,
               numeric_imputation_strategy='training_mean',numeric_imputation_values=means.tolist(),
               numeric_observed_counts=counts.tolist(),all_missing_numeric_columns=[c for c,n in zip(numeric,counts) if n==0],
               all_missing_numeric_fill=all_missing_fill,categorical_imputation_values=modes,categories=categories,
               unknown_category_policy='all_zero_one_hot',scaler_mean=scaler.mean_.tolist(),
               scaler_variance=scaler.var_.tolist(),scaler_scale=scaler.scale_.tolist(),
               constant_training_numeric_columns=[c for c,v in zip(numeric,scaler.var_) if v==0],
               output_feature_names=numeric+[f'{c}={v}' for c in categorical for v in categories[c]],
               excluded_target_encodings=excluded)
    return state


def iter_transformed(config,partition,scaled=True):
    """Reusable downstream interface; yields (X, binary y, provenance) in chronology."""
    reports,manifest,plan=load_plan(config)
    saved=json.loads((reports/'preprocessing.json').read_text())
    require(saved['split_plan_sha256']==sha256(reports/'split_plan.json'),'Preprocessor belongs to a different split')
    processor=FrozenPreprocessor(saved)
    for batch in iter_partition(config,plan,partition,list(processor.numeric)+list(processor.categorical)+META):
        x,_=processor.transform(batch,scaled=scaled)
        y=batch.column(batch.schema.get_field_index('target_binary')).to_numpy()
        yield x,y,batch.select(META)


def run(config):
    reports,manifest,plan=load_plan(config)
    destination=reports/'preprocessing.json'
    require(not destination.exists(),'Preprocessing is already fitted; use a new output version.')
    print('Fitting imputation, categorical vocabulary, leakage checks and scaler on TRAIN ONLY',flush=True)
    def training_batches():
        return iter_partition(config,plan,'train')
    state=fit_training(training_batches,manifest['features'],config['categorical_columns'],config['all_missing_numeric_fill'])
    require(state['training_rows']==plan['partitions']['train']['rows'],'Training row-count mismatch')
    state.update(split_plan_sha256=sha256(reports/'split_plan.json'),source_manifest_sha256=sha256(reports/'source_manifest.json'),
                 environment=versions(),fitting_order='Chronological files and recorded timestamps; original source_row breaks timestamp ties.',
                 materialization='Transform batches on demand; no second full scaled feature copy.')
    json_write(destination,state)
    with (reports/'schema_exclusions.csv').open(newline='',encoding='utf-8') as f:
        exclusions=list(csv.DictReader(f))
    csv_write(reports/'feature_exclusion_audit.csv',exclusions+state['excluded_target_encodings'])
    # Reload the persisted object and apply it, unchanged, to every row in every partition.
    processor=FrozenPreprocessor.load(destination)
    fitted_hash=sha256(destination)
    checks=[]
    for part in PARTITIONS:
        rows=imputed=0
        unknown=Counter()
        transformed_hash=hashlib.sha256()
        for batch in iter_partition(config,plan,part,list(processor.numeric)+list(processor.categorical)+META):
            x,stats=processor.transform(batch)
            transformed_hash.update(np.ascontiguousarray(x,dtype='<f8').tobytes())
            rows+=len(x)
            imputed+=stats['imputed_numeric_cells']
            unknown.update(stats['unknown_categories'])
        require(rows==plan['partitions'][part]['rows'],'Transform row-count mismatch')
        require(sha256(destination)==fitted_hash,'Preprocessing changed during transform')
        checks.append(dict(partition=part,rows=rows,features=len(processor.output_names),
                           numeric_cells_imputed=imputed,unknown_category_counts_json=json.dumps(dict(unknown)),
                           transformed_values_all_finite=True,preprocessing_sha256=fitted_hash,
                           transformed_float64_sha256=transformed_hash.hexdigest()))
        print(f'Applied frozen preprocessing: {part}, {rows:,} rows',flush=True)
    csv_write(reports/'preprocessing_application_audit.csv',checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline.json')
    run(load_config(parser.parse_args().config))
