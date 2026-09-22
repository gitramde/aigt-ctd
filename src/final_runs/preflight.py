"""Fail-closed Phase11 environment gate. No fitting or dependency installation."""
import importlib,json,platform,time,traceback,hashlib
from . import ROOT,OUT

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def environment():
    failure=OUT/'audit/STOP_FAILURE.json'
    if failure.exists():raise RuntimeError('Prior failure preserved; automatic retry forbidden: '+str(failure))
    spec=read(ROOT/'results/final_spec_v1/final_experiment_spec.json')
    write(OUT/'audit/execution_approval.json',dict(phase=11,approval='User explicitly APPROVED final_spec_v1 execution for42,123,456,789,1024',
        approved_seeds=spec['seeds'],spec_sha256=sha(ROOT/'results/final_spec_v1/final_experiment_spec.json'),
        frozen_spec_modified=False,recorded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    observed={};records=[]
    try:
        aliases={'scikit_learn':'sklearn'}
        for name,expected in spec['runtime']['software'].items():
            if name=='python':actual=platform.python_version();location='interpreter'
            else:
                module=importlib.import_module(aliases.get(name,name));actual=module.__version__;location=str(module.__file__)
            observed[name]=actual;records.append(dict(dependency=name,expected=expected,observed=actual,module_path=location,passed=actual==expected))
            if actual!=expected:raise RuntimeError(f'Required software version mismatch: {name}: frozen={expected}, observed={actual}')
        import os,psutil,torch,winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
        hardware=dict(cpu=cpu,physical_cores=psutil.cpu_count(logical=False),logical_cpus=os.cpu_count(),ram_bytes=psutil.virtual_memory().total,platform=platform.platform())
        if hardware!=spec['runtime']['hardware']:raise RuntimeError('Required hardware mismatch: '+json.dumps(hardware))
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        if torch.get_num_threads()!=2:raise RuntimeError('Required PyTorch thread count mismatch')
        write(OUT/'audit/environment_check.json',dict(status='PASS',software=records,hardware=hardware,threads=2,device='cpu'))
        print('Environment and hardware checks PASS; no fitting performed',flush=True)
    except Exception as exc:
        write(OUT/'audit/environment_check.json',dict(status='FAIL',software=records,training_started=False,error=str(exc)))
        write(failure,dict(status='STOPPED',stage='environment_preflight',mandatory_fit_started=False,completed_final_fits=0,
            reason=str(exc),traceback=traceback.format_exc(),automatic_retry_allowed=False,frozen_spec_modified=False))
        (OUT/'FINAL_EXECUTION_SUMMARY.md').write_text('# Phase11 execution stopped before first fit\n\nRequired environment preflight failed: '+str(exc)+
            '\n\nNo final fit or seed was executed. The failure is preserved in audit/STOP_FAILURE.json and audit/environment_check.json. '
            'No dependency, frozen specification or development artifact was changed. Final aggregation is unavailable. Execution stopped under the explicit Phase11 failure rule.\n',encoding='utf-8')
        print('STOP:',str(exc),flush=True)
        raise

if __name__=='__main__':environment()
