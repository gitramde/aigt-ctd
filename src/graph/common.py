import json
from pathlib import Path
from . import ROOT
from src.baseline.common import require,sha256,json_write,csv_write

OUT=ROOT/'results/graph_v1'
CONFIG=OUT/'configs'; METRICS=OUT/'metrics'; MODELS=OUT/'models'; FIGURES=OUT/'figures'
CACHE=ROOT/'data/graph_v1'
PARTS=('train','validation','test')
PROTOCOL=dict(seed=42,primary_window_minutes=1,sensitivity_windows=[5,10],
    cutoff_second=30,target_seconds=[30,60],train_stride=16,evaluation_stride=4,
    history_edge_cap=8192,history_cap_policy='most recently completed eligible flows; source_row breaks completion ties',
    node_history='all completed flows within the assigned calendar window before cutoff; no labels',
    target_rule='current completed flow binary label; current features available at classification time',
    snapshot_rule='one causal prefix per occupied target minute at second 30; reset calendar window and partition',
    message_passing='directed historical Source IP -> Destination IP, two layers, explicit self loops with zero edge attributes',
    node_features=['log1p_in_count','log1p_out_count','log1p_in_bytes','log1p_out_bytes','log1p_tcp_incident','log1p_udp_incident','log1p_low_source_port','log1p_low_destination_port'],
    historical_edge_feature_names=['Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts','TotLen Fwd Pkts','TotLen Bwd Pkts','Dst Port','Protocol=0','Protocol=6','Protocol=17'],
    max_epochs=4,patience=2,batch_graphs=1,threads=2,learning_rate=0.0005,weight_decay=0.01,
    selection_metric='validation_macro_f1_at_0.5',min_improvement=0.00001,gradient_clip_norm=1.0,
    candidates=[dict(hidden=64,heads=4,layers=2,dropout=0.1),dict(hidden=128,heads=4,layers=2,dropout=0.2)],
    additional_baselines=dict(logistic_regression='SGD log_loss; four passes',random_forest='100 trees max_depth 16 balanced_subsample',xgboost='150 rounds max_depth 5; validation early stopping 20'),
    sensitivity_selection='primary validation-selected architecture held fixed; each window selects epoch and thresholds on validation only',
    endpoint_id_policy='indices only, never learned identity embeddings or numerical IP features',
    history_state='reset at partition/window boundaries; validation/test history may use earlier unlabeled completed flows from that same partition',
    detection_time='flow completion; target features may summarize the full current flow. No claim of start-time detection.')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(name,rows): csv_write(OUT/name,rows)

def guard():
    require(read(CONFIG/'protocol.json')==PROTOCOL,'Graph protocol changed')
    approval=read(CONFIG/'split_approval.json')
    require(approval['approved'] and approval['proposal_sha256']==sha256(OUT/'pre_split/proposed_split.json'),'Split approval changed')
    for row in read(METRICS/'frozen_artifacts.json'):
        require(sha256(row['path'])==row['sha256'],'Protected artifact changed: '+row['path'])

def initialize():
    for folder in (CONFIG,METRICS,MODELS,FIGURES,CACHE): folder.mkdir(parents=True,exist_ok=True)
    if not (CONFIG/'protocol.json').exists(): json_write(CONFIG/'protocol.json',PROTOCOL)
    if not (CONFIG/'split_approval.json').exists():
        json_write(CONFIG/'split_approval.json',dict(approved=True,approval_source='User replied yes to explicit 10:30/11:00 split approval question',
            proposal_sha256=sha256(OUT/'pre_split/proposed_split.json'),boundaries=read(OUT/'pre_split/proposed_split.json')['boundaries']))
    if not (METRICS/'frozen_artifacts.json').exists():
        files=[]
        for folder in ('results/baseline_v1','results/temporal_v1','src/baseline','src/phase4','src/phase4a','src/temporal'):
            files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        json_write(METRICS/'frozen_artifacts.json',[dict(path=str(p),sha256=sha256(p)) for p in files])
    guard()
