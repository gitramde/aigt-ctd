"""Display low-probability structure without choosing any test threshold."""
import numpy as np
import matplotlib.pyplot as plt
from .run import MODEL_NAMES,labels,scores,savefig

def run():
    for model in MODEL_NAMES:
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,part in zip(axes,('validation','test')):
            _,codes,book=labels(part);p=scores(model,part)
            for family in (['Benign','Infilteration'] if part=='validation' else ['Benign','Infilteration','Bot']):
                values=np.sort(p[codes==book.index(family)])
                indices=np.unique(np.linspace(0,len(values)-1,min(4000,len(values))).astype(int))
                ax.step(values[indices],(indices+1)/len(values),where='post',label=family)
            ax.set_xscale('symlog',linthresh=1e-12);ax.set_xlim(0,1);ax.axvline(.5,color='black',ls='--',lw=.8)
            ax.set(xlabel='Malicious score (symlog; linear below 1e-12)',ylabel='Empirical cumulative fraction',title=f'{model}: {part}')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        savefig(fig,'score_distributions',model+'_ecdf.png')

if __name__=='__main__': run()
