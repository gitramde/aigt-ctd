"""Run frozen stages sequentially; every failed subprocess permanently stops this attempt."""
import subprocess,sys,traceback
from .common import *

def main():
    check_lock()
    completed=0;stage='starting'
    try:
        for group,names in GROUPS.items():
            for seed in SEEDS:
                for name in names:
                    for command in ('worker','evaluate'):
                        stage=f'{command}: {seed} {group} {name}'
                        print(stage,flush=True)
                        path=OUT/'audit'/f'{command}_{seed}_{name}.log'
                        require(not path.exists(),'Refusing automatic retry of an existing stage log: '+str(path))
                        with path.open('w',encoding='utf-8') as log:
                            result=subprocess.run([sys.executable,'-B','-u','-m','src.final_runs.'+command,'--seed',str(seed),'--group',group,'--model',name],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                        require(result.returncode==0,f'{stage} failed with exit {result.returncode}; see {path}')
                        if command=='worker':completed+=1
                        print('Complete:',stage,flush=True)
        stage='independent final reporting'
        from .report import main as report
        report()
        stage='final frozen artifact verification'
        from .check_artifacts import main as artifacts
        artifacts()
        print('All final runs and verification complete. STOP.',flush=True)
    except Exception as exc:
        record=dict(status='STOPPED',stage=stage,completed_fits=completed,reason=str(exc),traceback=traceback.format_exc(),automatic_retry_allowed=False)
        write(OUT/'audit/STOP_FAILURE.json',record)
        (OUT/'FINAL_EXECUTION_SUMMARY.md').write_text('# Phase 11 execution stopped\n\n'+str(exc)+f'\n\nCompleted fits: {completed}. Existing outputs and failure logs are preserved. No failed seed is replaced or automatically retried. Final five-seed evaluation is incomplete.\n',encoding='utf-8')
        raise

if __name__=='__main__':main()
