"""Required no-fit semantic, cohort, configuration, causality and masking gates."""
import ast,io,unittest,traceback,time
import numpy as np
from .common import *

def main():
    passed=[]
    try:
        from .build_vendor import SOURCES,transformed
        for name,(source,replacements) in SOURCES.items():
            expected=transformed((ROOT/source).read_text(encoding='utf-8-sig'),replacements)
            require((ROOT/f'src/final_runs/{name}.py').read_text(encoding='utf-8')==expected,'Adapted source differs from declared transformation: '+name)
            compile(expected,name,'exec')
        passed.append('Adapted source equivalence and compilation')
        from . import vendor_temporal as temporal, vendor_graph as graph, vendor_anomaly as anomaly, vendor_graph_temporal as gt
        import torch
        from src.graph.model import EdgeModel
        torch.set_num_threads(2)
        for s in SEEDS:
            for module in (temporal,graph,anomaly,gt):
                module.SEED=s;module.seed();actual=torch.rand(3);torch.manual_seed(s)
                require(torch.equal(actual,torch.rand(3)),'Torch seed propagation mismatch')
            for group,names in GROUPS.items():
                for name in names:
                    path=OUT/f'audit/effective/seed_{s}/{name}.json';path.parent.mkdir(parents=True,exist_ok=True)
                    write(path,dict(seed=s,model=name,group=group,frozen_model=SPEC['models'][name],epoch_rng='seed+epoch-1',output=str(folder(s,group)),spec_sha256=sha(ROOT/'results/final_spec_v1/final_experiment_spec.json')))
        passed.append('Five-seed RNG propagation and effective configurations')
        a=temporal.TemporalTransformer(SPEC['models']['transformer_L1']['configuration']);b=temporal.TemporalTransformer(SPEC['models']['transformer_L64']['configuration'])
        require(a.position_offset==63 and b.position_offset==0,'Current positional index mismatch')
        require(sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters()),'Temporal capacity mismatch')
        require(sum(p.numel() for p in gt.TemporalHead(336).parameters())==89089,'Graph-temporal dimension/capacity mismatch')
        require(anomaly.Autoencoder([128,32,128]).network[0].in_features==80,'AE feature dimension mismatch')
        for name,old in [('edge_mlp','edge_mlp'),('gatv2','gatv2_w1_h128_p2'),('gatv2_self_only','gatv2_self_only')]:
            c=read(ROOT/f'results/graph_v1/configs/{old}.json')
            require(c['window_minutes']==1 and c['hidden']==128 and c['dropout']==.2,'Graph fixed configuration mismatch')
            if name=='edge_mlp':require(c['mlp_width']==298,'Edge MLP width mismatch')
            else:require(c['heads']==4,'Graph head count mismatch')
            EdgeModel(c,80,9)
        passed.append('Frozen architecture dimensions and capacity controls')
        suite=unittest.TestSuite()
        for module in ('src.graph.test_graph','src.graph_temporal.test_temporal','src.temporal.test_temporal','src.anomaly.test_anomaly'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (OUT/'audit/causality_masking_tests.txt').write_text(stream.getvalue(),encoding='utf-8')
        require(result.wasSuccessful(),'Causality/masking semantic tests failed; see audit/causality_masking_tests.txt')
        passed.append(f'{result.testsRun} focused causality/masking/threshold tests')
        from src.phase4.data import training_arrays
        x,y=training_arrays();require(x.shape==(12795136,80) and y.shape==(12795136,),'Full train shape mismatch')
        require(int((y==0).sum())==10885643,'Benign train count mismatch')
        benign=np.load(ROOT/'data/anomaly_v1/benign_train_ids.npy')
        require(np.array_equal(benign,np.flatnonzero(y==0)),'Benign train IDs mismatch')
        from src.temporal.data import cohort,cohort_indices
        for part,entry in SPEC['data']['full_temporal_cohort']['partitions'].items():
            c=cohort(part);require(np.array_equal(c['partition_index'],cohort_indices(entry['spans'],entry['stride'])),'Temporal cohort order mismatch')
            require(len(c['partition_index'])==entry['targets'],'Temporal target count mismatch')
        passed.append('Frozen full-data, benign, and temporal cohort identities')
        from src.graph.data import GraphData
        from src.graph_temporal.prepare import build_history
        data=GraphData();require(data.x.shape==(474357,80) and data.hx.shape==(7948746,9),'Graph feature dimensions mismatch')
        for i,n in enumerate((172984,88782,212591)):require(int((data.part==i).sum())==n,'Graph partition count mismatch')
        for g in data.groups:
            snap=data.snapshot(g,1)
            require(len(snap['history_rows'])<=8192,'Graph message cap mismatch')
            require((data.meta.completion_us.to_numpy()[snap['history_rows']]<snap['cutoff']*1000000).all(),'Noncausal graph input')
        print('All graph prefixes checked; replaying fixed history identities',flush=True)
        timings=[];targets,history,types,counts=build_history(data.meta,data.ids,timings)
        require(np.array_equal(history,np.load(ROOT/'results/graph_temporal_v1/metrics/history_indices.npy')),'Historical token identity mismatch')
        require(np.array_equal(types,np.load(ROOT/'results/graph_temporal_v1/metrics/history_identity_types.npy')),'Historical endpoint fallback mismatch')
        write(OUT/'audit/history_replay.json',dict(status='PASS',timings=timings,scope='Offline replay including diagnostic counts and eight slots; shared by L4/L8; L1 needs no history index'))
        passed.append('All graph prefixes and exact historical identity replay')
        write(OUT/'audit/semantic_checks.json',dict(status='PASS',checks=passed,no_fitting=True))
        print('Semantic/cohort/causality checks PASS',flush=True)
    except Exception as exc:
        record=dict(status='STOPPED',stage='semantic_preflight',reason=str(exc),passed=passed,traceback=traceback.format_exc(),completed_final_fits=0)
        write(OUT/'audit/semantic_checks.json',record);write(OUT/'audit/STOP_FAILURE.json',record)
        (OUT/'FINAL_EXECUTION_SUMMARY.md').write_text('# Phase 11 stopped before fitting\n\n'+str(exc)+'\n\nNo final model fit was started. See audit/semantic_checks.json.\n',encoding='utf-8')
        raise

if __name__=='__main__':main()
