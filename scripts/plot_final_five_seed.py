"""Plot saved final metrics only; no model, score, or experiment execution."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/final_runs_v1'
OUT = ROOT / 'results/final_figures_v1'
SEEDS = [42, 123, 456, 789, 1024]
CAP = 'fpr_at_most_0.01'
COLORS = ['#276690', '#D77A26', '#388369', '#9467A5', '#C45461', '#648B9A']
NAMES = {'logistic_regression':'LR', 'random_forest':'RF', 'xgboost':'XGB',
         'mlp':'MLP', 'autoencoder_B':'AE-B', 'transformer_L1':'Transformer-L1',
         'transformer_L64':'Transformer-L64', 'rf_or_ae':'RF+AE OR'}
inputs = {}
exports = []

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(relative):
    path = BASE / relative
    inputs[path] = digest(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def select(rows, **conditions):
    return [r for r in rows if all(r[k] == v for k,v in conditions.items())]

def values(rows, metric, figure, series, criterion, cohort):
    assert len(rows) == 5 and sorted(int(r['seed']) for r in rows) == sorted(SEEDS), (series, metric, len(rows))
    rows = sorted(rows, key=lambda r: SEEDS.index(int(r['seed'])))
    a = np.array([float(r[metric]) for r in rows])
    assert np.isfinite(a).all()
    for r, v in zip(rows, a):
        exports.append(dict(figure=figure, series=series, cohort=cohort, criterion=criterion,
                            metric=metric, seed=r['seed'], value=v, mean=a.mean(), sample_sd=a.std(ddof=1)))
    return a

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', color='#DFE4E8', linewidth=.7)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, facecolor='white')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.titlesize':11, 'axes.labelsize':10,
                         'svg.fonttype':'none', 'pdf.fonttype':42})
    a = read('aggregate_phase11a/per_seed_metrics.csv')
    b = read('aggregate_phase11b/per_seed_metrics.csv')
    family = read('aggregate_phase11a/rf_ae_or_comparison.csv')
    pairs = read('aggregate_phase11a/paired_per_seed_differences.csv') + read('aggregate_phase11b/paired_component_differences.csv')
    conventional = ['logistic_regression','random_forest','xgboost','mlp']
    fig, axes = plt.subplots(2,2,figsize=(13,8.8))
    for col,(cohort,models,title) in enumerate([
        ('full_dataset',conventional+['autoencoder_B'],'Full dataset | 1,374,143 test flows'),
        ('phase5_matched',conventional+['transformer_L1','transformer_L64'],'Phase 5 matched | 171,753 test targets')]):
        for row,metric in enumerate(['f1','false_positive_rate']):
            ax=axes[row,col]
            for y,m in enumerate(models):
                criterion='benign_'+CAP if m=='autoencoder_B' else CAP
                v=values(select(a,model=m,cohort=cohort,partition='test',criterion=criterion),metric,'01',m,criterion,cohort)*100
                ax.errorbar(v.mean(),y,xerr=v.std(ddof=1),fmt='o',color=COLORS[y],capsize=4,ms=7,lw=1.7)
            ax.set_yticks(range(len(models)),[NAMES[m] for m in models])
            ax.set_ylim(len(models)-.5,-.5)
            ax.set_xlabel('F1 (%) — higher is better' if row==0 else 'Test FPR (%) — lower is better')
            if row==0: ax.set_title(title,pad=13,fontweight='bold')
            style(ax)
    fig.suptitle('Five-seed model comparison',fontsize=18,fontweight='bold',y=.98)
    fig.text(.5,.93,'Frozen validation FPR ≤ 1% thresholds; AE-B uses benign-only calibration',ha='center')
    fig.text(.06,.035,'Points: mean ± sample SD (ddof=1), n=5. Test FPR is not constrained to 1%.\nCohorts are separate: conventional controls in the right panels are sliced and recalibrated on matched validation targets.',fontsize=9)
    fig.subplots_adjust(left=.13,right=.98,top=.85,bottom=.14,wspace=.55,hspace=.32)
    save(fig,'01_five_seed_model_comparison')

    fig, axes=plt.subplots(1,2,figsize=(10,5.5),sharey=True)
    for ax,f in zip(axes,['Bot','Infilteration']):
        for y,m in enumerate(['random_forest','autoencoder_B','rf_or_ae']):
            selected=select(family,model=m,cohort='full_dataset',partition='test')
            v=values(selected,f+'_recall','02',m,selected[0]['criterion'],'full_dataset')*100
            ax.errorbar(v.mean(),y,xerr=v.std(ddof=1),fmt='o',color=COLORS[y],capsize=5,ms=8,lw=2)
            sd=v.std(ddof=1)
            sd_label=f'{sd:.2e}' if 0 < sd < .01 else f'{sd:.2f}'
            ax.annotate(f'{v.mean():.2f} ± {sd_label}%',(v.mean(),y),xytext=(3 if v.mean()<1 else 0,13),textcoords='offset points',ha='left' if v.mean()<1 else 'center',fontsize=9)
        ax.set_title(f,fontweight='bold')
        ax.set_yticks([0,1,2],['RF','AE-B','RF+AE OR'])
        ax.set_ylim(2.5,-.65)
        ax.set_xlabel('Test family recall (%)')
        style(ax)
        ax.margins(x=.28)
        ax.set_xlim(left=0)
    fig.suptitle('Attack families absent from training',fontsize=17,fontweight='bold',y=.98)
    fig.text(.5,.90,'Full dataset • frozen component 1% validation operating points • five seeds',ha='center',fontsize=10)
    fig.text(.07,.045,'Mean ± sample SD (ddof=1); panels use different x scales to show Infilteration differences.\nOR combines matching-seed RF and AE decisions; its union FPR is not capped at 1%.',fontsize=9)
    fig.subplots_adjust(left=.14,right=.97,top=.78,bottom=.21,wspace=.26)
    save(fig,'02_unseen_family_recall')

    comparisons=[('gatv2','edge_mlp','Edge MLP → GATv2\nFeb20 matched'),
                 ('gatv2','gatv2_self_only','Self-only → GATv2\nFeb20 matched'),
                 ('transformer_L64','transformer_L1','Transformer L1 → L64\nPhase 5 matched'),
                 ('gat_transformer_L8','gat_transformer_L1','GAT+Transformer L1 → L8\nFeb20 matched')]
    fig,axes=plt.subplots(3,2,figsize=(14,12))
    for row,(criterion,title) in enumerate([('fixed_0_5','Fixed threshold 0.5'),(CAP,'Validation FPR ≤ 1%'),('maximum_macro_f1','Validation maximum macro-F1')]):
        for col,metric in enumerate(['f1','false_positive_rate']):
            ax=axes[row,col]
            for y,(model,control,label) in enumerate(comparisons):
                selected=select(pairs,model=model,control=control,partition='test',criterion=criterion)
                v=values(selected,'delta_'+metric,'03',control+' → '+model,criterion,selected[0]['cohort'])*100
                # Independently verify stored signed differences from original per-seed metrics.
                for r,delta in zip(sorted(selected,key=lambda r:SEEDS.index(int(r['seed']))),v):
                    conditions=dict(seed=r['seed'],cohort=r['cohort'],partition='test',criterion=criterion)
                    mr=select(a+b,model=model,**conditions); cr=select(a+b,model=control,**conditions)
                    assert len(mr)==len(cr)==1
                    assert np.isclose(delta,100*(float(mr[0][metric])-float(cr[0][metric])),atol=1e-10,rtol=1e-10)
                for s,(delta,offset) in enumerate(zip(v,np.linspace(-.17,.17,5))):
                    ax.scatter(delta,y+offset,s=27,color=COLORS[s],alpha=.9,label=str(SEEDS[s]) if row==col==y==0 else None,zorder=3)
                ax.errorbar(v.mean(),y,xerr=v.std(ddof=1),fmt='D',color='#172B3A',ms=6,capsize=4,lw=1.8,zorder=4)
            ax.axvline(0,color='#697580',lw=1,ls='--')
            ax.set_yticks(range(4),[x[2] for x in comparisons] if col==0 else [])
            ax.set_ylim(3.55,-.55)
            ax.set_title(title+' | '+('ΔF1' if col==0 else 'ΔFPR'),fontweight='bold')
            ax.set_xlabel('Difference (percentage points) • '+('positive favors added component' if col==0 else 'positive means more false alarms'))
            style(ax)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,title='Individual seed',loc='upper center',bbox_to_anchor=(.57,.946),ncol=5,frameon=False)
    fig.suptitle('Paired component contributions across five seeds',fontsize=18,fontweight='bold',y=.985)
    fig.text(.07,.025,'Difference = model after arrow minus control before arrow, paired within seed and identical cohort.\nDiamonds and horizontal whiskers: mean ± sample SD (ddof=1). Colors: individual seeds; vertical offsets only prevent overlap.\nFixed, previously observed splits; SD reflects training stochasticity, not dataset uncertainty. No significance claims.',fontsize=9)
    fig.subplots_adjust(left=.22,right=.98,top=.865,bottom=.12,wspace=.16,hspace=.48)
    save(fig,'03_paired_component_ablations')
    with (OUT/'plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(exports[0]));w.writeheader();w.writerows(exports)
    assert all(digest(p)==h for p,h in inputs.items()), 'Input changed during plotting'
    report={'seeds':SEEDS,'std_ddof':1,'partition':'test','input_sha256':{str(p.relative_to(ROOT)):h for p,h in inputs.items()},
            'source_sha256':digest(Path(__file__)), 'checks':{'five_unique_seeds_every_series':True,'finite_values':True,'paired_deltas_recomputed':True,'input_files_unchanged':True},
            'notes':['No training, threshold recalibration, prediction or XAI executed.', 'Cohorts displayed separately; full dataset has no final Transformer predictions.', 'Symmetric SD whiskers are not confidence intervals and may cross the physical metric range.']}
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Created 3 figures in PNG/PDF/SVG; {len(exports)} plotted seed values. All checks passed. Output: {OUT}')

if __name__=='__main__':
    main()
