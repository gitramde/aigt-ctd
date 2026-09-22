"""Independent matched-cohort, causality, checkpoint and metric reconciliation."""
import csv,time
from .train import *
from .report import protected_check

def csvrows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    seed();data=GraphData();history=np.load(OUT/'metrics/history_indices.npy');types=np.load(OUT/'metrics/history_identity_types.npy')
    targets=data.meta.iloc[data.ids].reset_index(drop=True);n=len(targets)
    timings=[]
    _,replayed,replayed_types,_=build_history(data.meta,data.ids,timings)
    require(np.array_equal(history,replayed) and np.array_equal(types,replayed_types),'Optimized history replay differs from original')
    json_write(OUT/'metrics/history_preparation_timing.json',dict(timings=timings,scope='exact deterministic history-index replay; excludes metadata loading; includes diagnostic coverage counting; no model inference'))
    require(np.array_equal(np.load(OUT/'metrics/eligible_target_ids.npy'),data.ids),'Eligible IDs mismatch')
    require(np.array_equal(np.load(OUT/'metrics/eligible_source_rows.npy'),targets.source_row.to_numpy()),'Source row IDs mismatch')
    for lag in range(8):
        positions=np.flatnonzero(history[:,lag]>=0);old=history[positions,lag];kind=types[positions,lag]
        require((data.part[positions]==data.part[old]).all(),'Cross partition history')
        require((targets.second.to_numpy()[old]//60==targets.second.to_numpy()[positions]//60-lag-1).all(),'Wrong historical minute')
        require((targets.completion_us.to_numpy()[old]<(targets.second.to_numpy()[positions]//60*60+30)*1_000_000).all(),'Future history')
        source=targets.src.to_numpy()[positions]==targets.src.to_numpy()[old]
        destination=targets.dst.to_numpy()[positions]==targets.dst.to_numpy()[old]
        require(((source&destination)[kind==1]).all(),'Pair identity mismatch')
        require(source[kind==2].all() and destination[kind==3].all(),'Endpoint fallback mismatch')
    # Frozen encoder representations must reproduce the already saved Phase6 classifier.
    manifest=read(GRAPH/'metrics/data_manifest.json');encoder=EdgeModel(read(GRAPH/'selected_model.json')['config'],len(manifest['edge_feature_names']),len(manifest['history_feature_names']))
    encoder.load_state_dict(torch.load(GRAPH/f'models/{NAME}.pt',weights_only=True,map_location='cpu'));encoder.eval()
    representations=np.load(OUT/'metrics/representations.npy',mmap_mode='r');maximum=0.
    with torch.inference_mode():
        for partition in ('validation','test'):
            positions=np.flatnonzero(data.part==PARTS.index(partition));p=[]
            for start in range(0,len(positions),2048):
                p.extend(torch.sigmoid(encoder.classifier(torch.from_numpy(np.array(representations[positions[start:start+2048]]))).squeeze(-1)).numpy())
            old=np.load(GRAPH/f'metrics/{NAME}_{partition}_evaluation.npy')
            maximum=max(maximum,float(np.max(np.abs(old-np.asarray(p)))))
            require(np.allclose(old,p,atol=1e-6,rtol=1e-6),'Frozen representation/classifier replay mismatch')
    for partition in ('validation','test'):
        rows=csvrows(OUT/f'{partition}_metrics.csv');mask=data.part==PARTS.index(partition)
        require(len(rows)==15,'Missing model operating points')
        for row in rows:
            p=np.load(OUT/f"metrics/{row['model']}_{partition}.npy")
            recomputed=binary_metrics(data.y[mask],p,float(row['threshold']))
            for key in ('accuracy','precision','recall','f1','macro_f1','false_positive_rate','false_negative_rate','roc_auc','pr_auc','average_precision','true_positives','false_positives','true_negatives','false_negatives'):
                require(abs(float(row[key])-recomputed[key])<1e-12,'Metric mismatch: '+key)
            if partition=='validation' and row['criterion']=='fpr_at_most_0.01':
                require(float(row['false_positive_rate'])<=.01,'Validation FPR cap violated')
    params=[read(OUT/f'metrics/gat_transformer_L{l}_training.json')['trainable_parameters'] for l in (1,4,8)]
    require(len(set(params))==1,'Capacity-control parameter counts differ')
    count=protected_check()
    json_write(OUT/'metrics/output_verification.json',dict(status='PASS',matched_targets=n,protected_files=count,
        graph_classifier_replay_max_abs_difference=maximum,equal_head_parameters=params[0],
        checks=['saved source-row cohort exact','minute/partition/completion constraints','pair and endpoint identity',
                'frozen graph representation classifier replay','all saved metrics recomputed','validation FPR cap','capacity counts','prior artifacts unchanged']))
    print('Phase9 independent output verification PASS',flush=True)

if __name__=='__main__':main()
