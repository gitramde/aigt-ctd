"""Reviewed disk-full recovery for Phase11B. No training without --execute."""
import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from src.final_runs import phase11b as p

OLD = p.OUT / 'audit/phase11b/continuation_v2'
NEW = p.OUT / 'audit/phase11b/disk_recovery_v1'
SELF = Path(__file__).resolve()

def available_space():
    free=shutil.disk_usage(p.ROOT).free
    # Four remaining 637.5MB embedding arrays plus scores, audit files and headroom.
    p.require(free >= 6 * 1024**3, f'Need at least 6 GiB free for this continuation; available {free/1024**3:.2f} GiB')
    return free

def no_active_workers():
    import psutil
    for proc in psutil.process_iter(['pid','name']):
        if proc.pid == __import__('os').getpid() or 'python' not in (proc.info['name'] or '').lower():
            continue
        try: command=' '.join(proc.cmdline())
        except psutil.NoSuchProcess: continue
        except psutil.AccessDenied: raise RuntimeError(f'Cannot inspect Python process {proc.pid}; verify it has exited before recovery')
        p.require('src.final_runs' not in command and 'resume_phase11b_disk.py' not in command,
                  f'Another final-run process is active: {proc.pid}')

def completed():
    expected={(42,g,m) for g,m in p.ORDER}|{(123,'graph','edge_mlp'),(123,'graph','gatv2')}
    found=set()
    for seed in p.SEEDS:
        for group,name in p.ORDER:
            base=p.folder(seed,group)
            train=base/'metrics'/f'{name}_training.json'
            evaluation=base/'metrics'/f'{name}_evaluation.json'
            key=(seed,group,name)
            if key not in expected:
                p.require(not train.exists() and not evaluation.exists() and not (base/'models'/f'{name}.pt').exists(),
                          f'Unexpected later/partial fit needs review: {key}')
                continue
            t=p.read(train); e=p.read(evaluation)
            lock=base/'configs'/f'{name}_threshold_lock.json'
            p.require(t['status']==e['status']=='complete' and p.selection_seed(t)==seed and e['seed']==seed,'Completed provenance mismatch')
            p.require(p.sha(base/'models'/f'{name}.pt')==t['model_sha256']==e['checkpoint_sha256'],'Checkpoint changed')
            p.require(p.sha(lock)==e['threshold_lock_sha256'],'Threshold lock changed')
            for row in e['artifacts']:p.require(p.sha(p.ROOT/row['path'])==row['sha256'],'Saved score/ID changed')
            for action in ('fit','evaluate'):
                p.read(OLD/f'{action}_{seed}_{name}_runtime.json')
                p.read(OLD/f'{action}_{seed}_{name}_environment.json')
            found.add(key)
    p.require(found==expected,'Unexpected completed inventory')
    p.dependency(42);p.dependency(123)
    return sorted(found)

def prepare():
    no_active_workers()
    p.environment()
    p.AUDIT=OLD;p.verify_sources()
    free=available_space()
    log=(OLD/'fit_123_gat_transformer_L1.log').read_text(encoding='utf-8')
    p.require('No space left on device' in log and 'open_memmap' in log,'Failure is not the reviewed disk-full allocation failure')
    p.require((OLD/'STOP_FAILURE.json').exists(),'Original failure record missing')
    done=completed()
    partial=p.folder(123,'graph_temporal')/'metrics/representations.npy'
    p.require(partial.is_file() and partial.stat().st_size==128,'Partial representation differs from reviewed 128-byte header')
    p.require(not (partial.parent/'representation_manifest.json').exists(),'Completed representation exists; do not archive')
    NEW.mkdir(exist_ok=False)
    # Copy prior successful audit context; never remove or clear its STOP record.
    for name in ('prepared.json','execution_started.json','source_lock.json','protected_baseline.json',
                 'history_replay.json','semantic_checks.json','semantic_tests.txt','environment.json','authorization.json'):
        shutil.copy2(OLD/name,NEW/name)
    for seed,group,name in done:
        for action in ('fit','evaluate'):
            for suffix in ('.log','_runtime.json','_environment.json'):
                shutil.copy2(OLD/f'{action}_{seed}_{name}{suffix}',NEW/f'{action}_{seed}_{name}{suffix}')
    shutil.copytree(OLD/'source_snapshot',NEW/'source_snapshot')
    shutil.copy2(SELF,NEW/'source_snapshot'/SELF.name)
    lock=p.read(NEW/'source_lock.json')
    lock['files'].append(dict(path=str(SELF.relative_to(p.ROOT)),sha256=p.sha(SELF)))
    p.write(NEW/'source_lock.json',lock)
    archive=NEW/'partial_artifacts';archive.mkdir()
    target=archive/'seed123_incomplete_representations.npy'
    p.require(partial.resolve().is_relative_to(p.ROOT.resolve()) and target.resolve().is_relative_to(p.ROOT.resolve()),'Archive outside workspace')
    digest=p.sha(partial)
    p.write(NEW/'recovery.json',dict(reason='Errno 28 during embedding allocation before L1 fitting',
        original_audit=str(OLD),completed=done,free_bytes_before=free,partial_original=str(partial),
        partial_archive=str(target),partial_sha256=digest,source_sha256=p.sha(SELF),
        action='Rebuild incomplete seed123 representations using the same selected final GAT; retain all seven completed fits',
        training_started=False))
    shutil.move(str(partial),str(target))
    p.require(p.sha(target)==digest,'Partial archive verification failed')
    # Preserve all original failure evidence through final verification too.
    baseline=p.read(NEW/'protected_baseline.json')
    for file in OLD.rglob('*'):
        if file.is_file():baseline[str(file.resolve())]=p.sha(file)
    p.write(NEW/'protected_baseline.json',baseline)
    p.write(NEW/'recovery_prepared.json',dict(status='PASS',completed=done,training_started=False))
    print('Recovery prepared; seven fits retained, partial embedding archived. No training started.',flush=True)

def execute():
    no_active_workers();available_space()
    p.AUDIT=NEW;p.verify_sources();p.environment()
    p.require(p.read(NEW/'recovery_prepared.json')['status']=='PASS','Recovery not prepared')
    p.require(not (NEW/'resume_started.json').exists() and not (NEW/'STOP_FAILURE.json').exists(),'Recovery already attempted; review logs rather than retry')
    p.protected(recheck=True)
    done={tuple(row) for row in p.read(NEW/'recovery_prepared.json')['completed']}
    p.write(NEW/'resume_started.json',dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    try:
        for seed in p.SEEDS:
            for group,name in p.ORDER:
                if (seed,group,name) in done:continue
                for action in ('fit','evaluate'):
                    label=f'{action}_{seed}_{name}'
                    print('START',label,flush=True)
                    with (NEW/f'{label}.log').open('x',encoding='utf-8') as log:
                        result=subprocess.run([sys.executable,'-B','-u',str(SELF),'--stage',action,
                            '--seed',str(seed),'--group',group,'--model',name],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
                    p.require(result.returncode==0,'Failed: '+label+'; inspect preserved log')
                    print('PASS',label,flush=True)
        from src.final_runs.report11b import main as report
        report()
    except Exception:
        p.write(NEW/'STOP_FAILURE.json',dict(status='STOPPED',traceback=traceback.format_exc(),automatic_retry=False))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--stage',choices=['fit','evaluate'])
    parser.add_argument('--seed',type=int);parser.add_argument('--group');parser.add_argument('--model')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.execute:execute()
    else:
        p.AUDIT=NEW
        p.require((NEW/'resume_started.json').exists(),'Use the recovery controller')
        p.stage(args.stage,args.seed,args.group,args.model)
