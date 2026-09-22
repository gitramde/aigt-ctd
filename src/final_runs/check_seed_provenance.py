"""Exercise selection audit provenance without fitting or reading test data."""
import traceback
from .common import ROOT,OUT,read,write,require

def main():
    require(not (OUT/'audit/STOP_FAILURE.json').exists(),'Prior stop record exists')
    import numpy as np
    import src.final_runs.vendor_supervised as training
    captures=[]
    # These are precisely the Selection implementation and cfg binding used by worker.supervised.
    cfg=read(ROOT/'configs/phase4_seed42.json');cfg['seed']=123
    training.diagnostic_labels=lambda partition:(np.array([0,1],dtype=np.uint8),None,None)
    training.predict_partition=lambda *args:(np.array([0.1,0.9]),{})
    training.save_model=lambda *args:None
    training.json_write=lambda path,value:captures.append((str(path),value))
    training.METRICS=OUT/'audit/semantic_scratch'
    try:
        selection=training.Selection('logistic_regression',cfg)
        selection.check(None,1)
        observed=captures[-1][1]['seed']
        require(observed==123,f'Selection audit seed mismatch: requested 123, recorded {observed}')
        write(OUT/'audit/seed_provenance_check.json',dict(status='PASS',requested_seed=123,observed_seed=observed,model_fitting_performed=False))
    except Exception as exc:
        record=dict(status='STOPPED',stage='runner_seed_provenance_semantic_check',reason=str(exc),
            requested_seed=123,observed_record=captures[-1][1] if captures else None,
            completed_final_fits=0,model_fitting_performed=False,traceback=traceback.format_exc(),
            cause='New runner reused a development Selection implementation that hard-codes seed 42 in audit metadata.')
        write(OUT/'audit/seed_provenance_check.json',record)
        write(OUT/'audit/STOP_FAILURE.json',record)
        (OUT/'FINAL_EXECUTION_SUMMARY.md').write_text('# Phase 11 stopped before the first fit\n\n'
            'Runner construction retried successfully after correcting UTF-8 BOM handling. Environment and hardware checks passed, and all 592 frozen artifact hashes passed.\n\n'
            'The required runner seed-provenance semantic check failed: a synthetic seed-123 selection check recorded seed 42. The new supervised runner reused the development Selection class, whose audit metadata hard-codes seed 42. This is a runner implementation defect, not a change to the frozen specification.\n\n'
            'No model was fitted: the semantic check mocked predictions and checkpoint saving. All five final seeds remain unstarted. No final metrics or five-seed aggregation exist. Remaining preflight checks and runner snapshots are incomplete.\n\n'
            'Execution stopped immediately under the approved failure rule. Failure details are preserved in audit/STOP_FAILURE.json and audit/seed_provenance_check.json. No automatic repair, retry, or training follows this failure.\n',encoding='utf-8')
        raise

if __name__=='__main__':main()
