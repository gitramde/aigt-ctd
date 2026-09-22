"""Phase 4 isolated runtime; frozen baseline preparation is imported read-only."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
runtime=ROOT/'data/phase4_runtime'
if runtime.is_dir():
    sys.path.insert(0,str(runtime))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
