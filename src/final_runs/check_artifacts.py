"""Read-only frozen artifact gate; preserve and stop on the first mismatch."""
import csv, traceback
from .common import ROOT, OUT, SPEC, read, write, sha, require

def main():
    checked=[]
    try:
        require(not (OUT/'audit/STOP_FAILURE.json').exists(), 'Prior stop record exists')
        from src.final_spec.verify import main as verify_spec
        verify_spec()
        manifest=ROOT/SPEC['artifact_hash_manifest']['path']
        require(sha(manifest)==SPEC['artifact_hash_manifest']['sha256'], 'Artifact manifest hash mismatch')
        with manifest.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                path=ROOT/row['path']
                require(path.is_file(), 'Missing frozen artifact: '+row['path'])
                actual=sha(path)
                require(actual==row['sha256'], 'Frozen artifact hash mismatch: '+row['path'])
                require(path.stat().st_size==int(row['bytes']), 'Frozen artifact size mismatch: '+row['path'])
                checked.append(row['path'])
                if len(checked)%50==0: print('Verified',len(checked),'frozen artifacts',flush=True)
        write(OUT/'audit/artifact_check.json',dict(status='PASS',checked=checked))
        print('Frozen artifact check PASS:',len(checked),flush=True)
    except Exception as exc:
        failure=dict(status='STOPPED',stage='artifact_preflight',reason=str(exc),traceback=traceback.format_exc(),
                     completed_final_fits=0,training_started=False,checked=checked)
        write(OUT/'audit/artifact_check.json',failure)
        write(OUT/'audit/STOP_FAILURE.json',failure)
        (OUT/'FINAL_EXECUTION_SUMMARY.md').write_text('# Phase 11 stopped before training\n\n'+str(exc)+
            '\n\nThe required artifact check failed. No final fit has started. Runner construction is incomplete; remaining preflight checks and final aggregation have not run. Execution stopped under the approved failure rule.\n',encoding='utf-8')
        raise

if __name__=='__main__': main()
