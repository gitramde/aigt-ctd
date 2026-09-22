import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

from . import clean
from .common import ROOT, load_config, sha256
from . import split, preprocess, verify


def fixture(base):
    config = json.loads((ROOT / 'configs/baseline.json').read_text())
    config.update(raw_dir=str(base/'raw'), data_dir=str(base/'data'), report_dir=str(base/'reports'),
                  expected_raw_files=3, expected_totals={}, batch_size=3)
    config['split'].update(minimum_train_dates=1, minimum_validation_dates=1, minimum_test_dates=1,
                           minimum_benign_per_partition=1, minimum_malicious_per_partition=1)
    Path(config['raw_dir']).mkdir()
    header = ['Timestamp', 'Label', 'Protocol', 'Flow Byts/s', 'Flow Pkts/s', 'measurement', 'encoded', 'copied_label']
    for day in (14,15,16):
        rows = [
            [f'{day}/02/2018 02:00:00','Benign','6','NaN','2','4','0','Benign'],
            [f'{day}/02/2018 01:00:00','Attack','17','Infinity','3','2','1','Attack'],
            [f'{day}/02/2018 01:30:00','Benign','17','4','-inf','8','0','Benign'],
            [f'{day}/02/2018 01:00:00','Attack','6','2','1','10','1','Attack'],
        ]
        if day == 14:
            rows += [rows[0], header, header,
                     ['05/01/1970 03:00:00','Benign','6','4','1','6','0','Benign']]
        with (Path(config['raw_dir'])/f'Traffic-{day}-02-2018.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
    return config


class CleaningTests(unittest.TestCase):
    def test_collision_safe_dedup_and_chronology(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp))
            originals={p.name:sha256(p) for p in Path(cfg['raw_dir']).glob('*.csv')}
            with patch.object(clean, 'digest_record', return_value=b'forced-collision'):
                clean.run(cfg)
            manifest=json.loads((Path(cfg['report_dir'])/'source_manifest.json').read_text())
            self.assertEqual(manifest['totals']['repeated_headers'],2)
            self.assertEqual(manifest['totals']['exact_nonheader_duplicates'],1)
            self.assertEqual(manifest['totals']['year_1970_temporal_exclusions'],1)
            self.assertEqual(manifest['totals']['temporal_rows'],12)
            self.assertNotIn('copied_label',manifest['features'])
            table=pq.read_table(manifest['files'][0]['cleaned_path'])
            self.assertEqual(table['source_row'].to_pylist(),[2,4,3,1])
            self.assertEqual(table['Flow Byts/s'].null_count,2)
            self.assertEqual(table['Flow Pkts/s'].null_count,1)
            self.assertEqual(originals,{p.name:sha256(p) for p in Path(cfg['raw_dir']).glob('*.csv')})
            with self.assertRaises(ValueError): clean.run(cfg)

    def test_full_pipeline_train_only_fit_unknown_categories_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp))
            test_file=Path(cfg['raw_dir'])/'Traffic-16-02-2018.csv'
            with test_file.open(newline='') as f:
                rows=list(csv.reader(f))
            for row in rows[1:]:
                row[2]='99'
                row[5]='1000000'
            with test_file.open('w',newline='') as f:
                csv.writer(f).writerows(rows)
            clean.run(cfg)
            split.run(cfg)
            original=preprocess.iter_partition
            calls=[]
            def track(*args,**kwargs):
                calls.append(args[2])
                return original(*args,**kwargs)
            with patch.object(preprocess,'iter_partition',side_effect=track):
                preprocess.run(cfg)
            self.assertEqual(calls,['train','train','train','validation','test'])
            reports=Path(cfg['report_dir'])
            state=json.loads((reports/'preprocessing.json').read_text())
            self.assertEqual(state['training_rows'],4)
            self.assertNotIn('encoded',state['numeric_columns'])
            self.assertEqual(state['categories']['Protocol'],['17','6'])
            self.assertEqual(state['numeric_imputation_values'][state['numeric_columns'].index('measurement')],6)
            self.assertEqual(state['numeric_imputation_values'][state['numeric_columns'].index('Flow Byts/s')],3)
            frozen_hash=sha256(reports/'preprocessing.json')
            for x,y,metadata in preprocess.iter_transformed(cfg,'test',scaled=False):
                self.assertTrue(np.isfinite(x).all())
                self.assertTrue((x[:,-2:]==0).all())
                self.assertEqual(x.shape[1],len(state['output_feature_names']))
                self.assertEqual(len(x),len(metadata))
            self.assertEqual(frozen_hash,sha256(reports/'preprocessing.json'))
            verify.run(cfg)
            self.assertEqual(json.loads((reports/'verification.json').read_text())['status'],'PASS')
            # Changing an original source is detected instead of silently reusing artifacts.
            with test_file.open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError,'Original source changed'):
                verify.run(cfg)

    def test_no_feasible_split_fails_explicitly(self):
        from collections import Counter
        cfg=json.loads((ROOT/'configs/baseline.json').read_text())
        with self.assertRaisesRegex(ValueError,'No chronological split'):
            split.choose_split({'2018-02-14':Counter(Benign=1000)},cfg['split'])

    def test_all_missing_and_constant_features_remain_finite(self):
        batch=pa.record_batch({'all_missing':pa.array([None,None,None,None],type=pa.float64()),
                               'constant':[4.0]*4,'varying':[1.0,2.0,3.0,4.0],
                               'Protocol':pa.array([None]*4,type=pa.string()),
                               'attack_label':['Benign','Attack','Benign','Attack']})
        state=preprocess.fit_training(lambda:iter([batch]),['all_missing','constant','varying','Protocol'],['Protocol'])
        self.assertEqual(state['all_missing_numeric_columns'],['all_missing'])
        self.assertEqual(state['categories']['Protocol'],['<ALL_TRAIN_MISSING>'])
        processor=preprocess.FrozenPreprocessor(state)
        x,_=processor.transform(batch)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x[:,:2]==0).all())
        with self.assertRaisesRegex(ValueError,'Required numeric feature is missing'):
            processor.transform(batch.select(['all_missing','varying','Protocol','attack_label']))


if __name__=='__main__': unittest.main()
