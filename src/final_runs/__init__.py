"""Separate Phase11 execution package; development and specification are read-only."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for name in ('phase4_runtime','temporal_runtime'):
    sys.path.insert(0,str(ROOT/'data'/name))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
OUT=ROOT/'results/final_runs_v1'
