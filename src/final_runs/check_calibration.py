"""Independent brute-force threshold and metric tests, with no fitting."""
import numpy as np
from .common import *
from .metrics import thresholds,counts

def main():
    try:
        for seed in SEEDS:
            rng=np.random.default_rng(seed);y=np.r_[np.zeros(300,dtype=np.uint8),np.ones(100,dtype=np.uint8)];p=rng.integers(0,21,len(y))/20
            candidates=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)[::-1]]
            stats=[counts(y,p,t) for t in candidates]
            for row in thresholds(y,p,'temporal'):
                c=row['criterion'];t=row['threshold']
                if c=='fixed_0_5':require(t==.5,'Fixed threshold changed');continue
                if c=='maximum_macro_f1':best=max(range(len(stats)),key=lambda i:(stats[i]['macro_f1'],candidates[i]))
                else:
                    cap=float(c.rsplit('_',1)[1]);valid=[i for i,r in enumerate(stats) if r['false_positive_rate']<=cap]
                    best=max(valid,key=lambda i:(stats[i]['TP'],-stats[i]['FP'],candidates[i]))
                require(t==candidates[best],'Label-aware threshold algorithm mismatch')
            raw=p*1000
            for row in thresholds(y,raw,'anomaly'):
                cap=float(row['criterion'].rsplit('_',1)[1]);b=np.sort(raw[y==0]);k=int(np.floor(cap*len(b)))
                require(row['threshold']==np.nextafter(b[len(b)-k-1],np.inf),'Benign calibration algorithm mismatch')
                require(int((b>=row['threshold']).sum())<=k,'Benign FPR cap failed')
        write(OUT/'audit/calibration_check.json',dict(status='PASS',seeds=SEEDS,independent_brute_force=True,no_fitting=True))
        print('Frozen threshold algorithms PASS independent brute-force tests',flush=True)
    except Exception as exc:
        write(OUT/'audit/STOP_FAILURE.json',dict(status='STOPPED',stage='calibration_semantic_check',reason=str(exc),completed_final_fits=0));raise

if __name__=='__main__':main()
