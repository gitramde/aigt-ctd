"""Dedicated Phase12A tests. All preparation runs use temporary synthetic data."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.metadata
import numpy as np
import pandas as pd
from . import runner as r

def lock(role,t=.5):
    return dict(seed=42,model=r.MODELS[role][1],frozen_before_test=True,
        thresholds=[dict(criterion=r.CRITERIA[role],threshold=t)])

class Phase12ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[2]
        print('Hashing real protected artifacts before synthetic tests',flush=True)
        cls.before=r.snapshot(cls.root)

    @classmethod
    def tearDownClass(cls):
        print('Rechecking real protected artifacts after synthetic tests',flush=True)
        r.verify_snapshot(cls.root,cls.before)
        print(f'Protected file hashes unchanged: {len(cls.before)}',flush=True)

    def test_01_primary_seed(self):self.assertEqual(r.SEED,42);self.assertEqual(r.protocol()['primary_seed'],42)
    def test_02_all_criteria(self):
        for role in r.MODELS:self.assertEqual(r.threshold(lock(role,.123),role),.123)
        self.assertEqual(r.CRITERIA,dict(rf='fpr_at_most_0.01',ae='benign_fpr_at_most_0.01',gat='fpr_at_most_0.01',self_only='fpr_at_most_0.01'))
    def test_03_wrong_seed(self):
        v=lock('rf');v['seed']=123
        with self.assertRaises(ValueError):r.threshold(v,'rf')
    def test_04_missing_duplicate_threshold(self):
        for values in ([],lock('rf')['thresholds']*2):
            v=lock('rf');v['thresholds']=values
            with self.assertRaises(ValueError):r.threshold(v,'rf')
    def test_05_full_join_and_equality(self):
        m=pd.DataFrame(dict(source_row=[9,5],true_label=[0,1]))
        d=r.join_scores(m,np.array([0,1]),[.5,.4],np.array([0,1]),[1.,.5],'rf','ae',.5,.5)
        self.assertEqual(d.source_row.tolist(),[9,5]);self.assertEqual(d.rf_decision.tolist(),[1,0]);self.assertEqual(d.ae_decision.tolist(),[1,1])
    def test_06_bad_ids(self):
        m=pd.DataFrame(dict(source_row=[9,5],true_label=[0,1]))
        for a,b in [([0,1],[1,0]),([0,0],[0,0]),([0,2],[0,2]),([1,0],[1,0])]:
            with self.assertRaises(ValueError):r.join_scores(m,np.array(a),[.2,.8],np.array(b),[.2,.8],'rf','ae',.5,.5)
    def test_07_graph_indices(self):
        m=pd.DataFrame(dict(source_row=[501,901,22,78],true_label=[0,1,0,1],is_target_test=[False,True,False,True]))
        d=r.join_scores(m,np.array([1,3]),[.1,.9],np.array([1,3]),[.8,.1],'gat','self_only',.5,.5,True)
        self.assertEqual(d.metadata_index.tolist(),[1,3]);self.assertEqual(d.source_row.tolist(),[901,78])
    def test_08_nan_rejected(self):
        with self.assertRaises(ValueError):r.join_scores(pd.DataFrame({'true_label':[0]}),np.array([0]),[float('nan')],np.array([0]),[.2],'rf','ae',.5,.5)
    def test_09_categories(self):
        rf=pd.DataFrame(dict(target_id=range(5),source_row=range(5),true_label=[1,1,1,1,0],attack_family=['Bot','Bot','Infilteration','Infilteration','Benign'],rf_decision=[1,0,1,0,1],ae_decision=[0,1,1,0,0],rf_score=[.8,.1,.9,.2,.7],ae_score=[.1,.9,.8,.1,.1]))
        gat=pd.DataFrame(dict(target_id=[0,1,2],source_row=[0,1,2],true_label=[1,1,0],gat_decision=[1,0,1],self_only_decision=[0,1,1],gat_score=[.9,.1,.8]))
        cats=r.categories(rf,gat)
        expected={'RF_TP_BOT':[0],'RF_FN_BOT':[1],'RF_TP_INFILTERATION':[2],'RF_FN_INFILTERATION':[3],'RF_FP_BENIGN':[4],
          'RF_ONLY_ATTACK':[0],'AE_ONLY_ATTACK':[1],'RF_AND_AE_ATTACK':[2],'GAT_TP':[0],'GAT_FN':[1],'GAT_FP':[2],
          'GAT_POS_SELF_NEG':[0],'GAT_NEG_SELF_POS':[1]}
        self.assertEqual(set(cats),set(expected))
        for name,ids in expected.items():self.assertEqual(cats[name][0].target_id.tolist(),ids)
    def test_10_quantiles(self):
        d=pd.DataFrame(dict(target_id=range(11),source_row=range(11),rf_score=range(11)))
        rows=r.select_cases(d,'rf_score','x');self.assertEqual([x['target_id'] for x in rows],[1,5,9])
    def test_11_ties_and_distinct(self):
        d=pd.DataFrame(dict(target_id=[3,1,2],source_row=[30,10,20],rf_score=[.5]*3))
        self.assertEqual([x['target_id'] for x in r.select_cases(d,'rf_score','x')],[1,2,3])
        self.assertEqual(len(r.select_cases(d.iloc[:1],'rf_score','x')),1)
        self.assertEqual(len(r.select_cases(d.iloc[:2],'rf_score','x')),2)
    def test_12_empty(self):
        d=pd.DataFrame(columns=['target_id','source_row','rf_score'])
        self.assertEqual(r.select_cases(d,'rf_score','EMPTY_TEST')[0]['status'],'EMPTY')
    def test_13_explanation_cannot_affect_selection(self):
        d=pd.DataFrame(dict(target_id=range(7),source_row=range(7),rf_score=range(7)))
        expected=r.select_cases(d,'rf_score','x');d['shap_appearance']=list(reversed(range(7)))
        self.assertEqual(expected,r.select_cases(d.sample(frac=1,random_state=7),'rf_score','x'))
    def test_14_dependency_detection_no_install(self):
        def absent(name):raise importlib.metadata.PackageNotFoundError(name)
        self.assertEqual(set(r.dependencies(absent).values()),{'UNAVAILABLE'})
        self.assertEqual(r.dependencies(lambda _: 'test')['shap'],'test')
    def test_15_replay_plan(self):
        p=r.REPLAY
        self.assertEqual(p['cutoff_second'],30);self.assertEqual(p['message_cap'],8192)
        self.assertIn('<',p['completion_rule']);self.assertIn('all eligible',p['node_aggregates'])
        self.assertTrue(p['directed_edges'] and p['parallel_edges']);self.assertIn('source_row',p['message_selection'])
        self.assertIn('explicit model',p['self_loops']);self.assertFalse(p['aggregate_reconstruction_from_capped_edges'])
        self.assertFalse(p['execution_in_phase12a'])
    def test_16_no_training_or_explanation_apis(self):
        tree=ast.parse(Path(r.__file__).read_text())
        banned={'fit','partial_fit','step','predict','predict_proba','load_model','TreeExplainer','Explainer','shap_values','backward','system','Popen','check_call'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                self.assertNotIn(name,banned)
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                self.assertFalse(any(x.split('.')[0] in {'torch','shap','captum','torch_geometric','joblib','subprocess'} for x in modules))
    def test_17_hash_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in r.PROTECTED:(root/folder).mkdir(parents=True)
            f=root/r.PROTECTED[0]/'example';f.write_bytes(b'original');s=r.snapshot(root)
            r.verify_snapshot(root,s);f.write_bytes(b'changed')
            with self.assertRaises(ValueError):r.verify_snapshot(root,s)
    def test_18_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'results/xai_v1').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'already exists'):r.run(root)
    def test_19_feature_mapping_complete(self):
        rows=r.feature_groups([f'f{i}' for i in range(80)])
        self.assertEqual(sorted(x['index'] for x in rows),list(range(1,81)))
        self.assertEqual(sum(x['perturbation_unit']=='protocol_one_hot' for x in rows),3)
    def test_20_actual_schema_paths_read_only(self):
        import pyarrow.parquet as pq
        expected={'source_file','source_row','timestamp','attack_label','target_binary'}
        self.assertTrue(expected<=set(pq.read_schema(self.root/'data/baseline_v1/membership/test.parquet').names))
        self.assertTrue({'source_row','partition','second','target_binary'}<=set(pq.read_schema(self.root/'data/graph_v1/metadata.parquet').names))
        for role,(group,name,ext) in r.MODELS.items():
            base=self.root/f'results/final_runs_v1/seed_42/{group}'
            self.assertTrue((base/f'models/{name}.{ext}').is_file())
            r.threshold(r.read_json(base/f'configs/{name}_threshold_lock.json'),role)
        print('Dependency metadata only:',r.dependencies(),flush=True)
    def test_21_synthetic_end_to_end_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in r.PROTECTED:(root/folder).mkdir(parents=True)
            def js(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            m=root/'data/baseline_v1/membership/test.parquet';m.parent.mkdir(parents=True)
            pd.DataFrame(dict(source_file=['a','a','a'],source_row=[10,11,12],timestamp=pd.to_datetime(['2020-01-01']*3),
                attack_label=['Benign','Bot','Infilteration'],target_binary=[0,1,1])).to_parquet(m,index=False)
            names=[f'feature_{i}' for i in range(80)]
            spec={'data':{'full_preprocessing':{'output_feature_names':names},'full_split':{'partitions':{'test':{'membership_sha256':r.digest(m)}}}}}
            sp=root/'results/final_spec_v1/final_experiment_spec.json';js(sp,spec)
            js(sp.parent/'freeze_integrity.json',{'final_output_hashes':{'final_experiment_spec.json':r.digest(sp)}})
            js(root/'results/baseline_v1/preprocessing.json',{'output_feature_names':names})
            graph=root/'data/graph_v1';pd.DataFrame(dict(source_row=[100,200,300,400],target_binary=[0,1,0,1],partition=[0,2,2,2],second=[1,2,3,4])).to_parquet(graph/'metadata.parquet',index=False)
            np.save(graph/'target_ids.npy',np.array([1,2,3]))
            js(root/'results/graph_v1/metrics/data_manifest.json',{'artifacts':[dict(path=str(p),sha256=r.digest(p)) for p in graph.iterdir()]})
            for role,(group,name,ext) in r.MODELS.items():
                base=root/f'results/final_runs_v1/seed_42/{group}'
                for folder in ['models','metrics','configs']:(base/folder).mkdir(parents=True,exist_ok=True)
                ck=base/f'models/{name}.{ext}';ck.write_bytes(b'not a model; never loaded')
                ids=np.array([1,2,3]) if group=='graph' else np.arange(3)
                paths=[]
                for kind,array in [('ids',ids),('scores',np.array([.5,.2,.9]))]:
                    p=base/f'metrics/{name}_test_{kind}.npy';np.save(p,array);paths.append(p)
                l=lock(role);l.update(checkpoint_sha256=r.digest(ck),spec_sha256=r.digest(sp))
                lp=base/f'configs/{name}_threshold_lock.json';js(lp,l)
                js(base/f'metrics/{name}_evaluation.json',dict(status='complete',seed=42,model=name,checkpoint_sha256=r.digest(ck),
                    threshold_lock_sha256=r.digest(lp),artifacts=[dict(path=str(p.relative_to(root)),sha256=r.digest(p)) for p in paths]))
            baseline=r.snapshot(root)
            r.run(root)
            r.verify_snapshot(root,baseline)
            out=root/'results/xai_v1'
            self.assertEqual({str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()},set(r.OUTPUTS))
            d=pd.read_parquet(out/'manifests/gat_seed42_decisions.parquet');self.assertEqual(d.source_row.tolist(),[200,300,400])
            self.assertTrue((out/'audit/integrity_report.md').is_file())

if __name__=='__main__':unittest.main()
