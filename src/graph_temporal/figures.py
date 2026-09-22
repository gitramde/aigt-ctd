"""Error-count plots expose differences hidden by near-perfect macro-F1."""
import csv
from .prepare import OUT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    with (OUT/'test_metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    labels={'edge_mlp':'Edge MLP','gatv2_w1_h128_p2':'GATv2','gat_transformer_L1':'GAT + L1','gat_transformer_L4':'GAT + L4','gat_transformer_L8':'GAT + L8'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    criteria=('fixed_0_5','maximum_macro_f1','fpr_at_most_0.01')
    for j,criterion in enumerate(criteria):
        subset=[r for r in rows if r['criterion']==criterion]
        for i,(metric,title) in enumerate((('false_positives','False positives'),('false_negatives','False negatives'))):
            ax=axes[i,j];values=[int(r[metric]) for r in subset]
            bars=ax.barh([labels[r['model']] for r in subset],values)
            ax.bar_label(bars,padding=3,fontsize=9);ax.set(xlabel='Test target edges',title=criterion+'\n'+title,xlim=(0,max(1,max(values))*1.2))
    fig.suptitle('Matched Phase 9 test cohort | 193,893 benign and 18,698 DDoS targets')
    fig.savefig(OUT/'figures/test_error_counts.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
