"""Scoped Phase 11A orchestration; original runner and freeze remain immutable."""
import subprocess
import sys
import shutil
import traceback
from .common import *

SCOPE = {'supervised': GROUPS['supervised'],
         'temporal': ['transformer_L1', 'transformer_L64'],
         'anomaly': ['autoencoder_B']}
AUDIT = OUT / 'audit/phase11a'
SUMMARY = OUT / 'PHASE_11A_EXECUTION_SUMMARY.md'

def inventory():
    rows = []
    for seed in SEEDS:
        for group, names in SCOPE.items():
            for name in names:
                p = folder(seed, group)
                train = p / 'metrics' / f'{name}_training.json'
                evaluation = p / 'metrics' / f'{name}_evaluation.json'
                status = 'not_started'
                if train.exists():
                    t = read(train)
                    require(t['status'] == 'complete', 'Incomplete training record')
                    checkpoint = p / 'models' / (name + ('.ubj' if name == 'xgboost' else '.joblib' if group == 'supervised' else '.pt'))
                    require(sha(checkpoint) == t['model_sha256'], 'Training checkpoint hash mismatch')
                    recorded_seed = t.get('seed', t.get('config', {}).get('seed'))
                    require(recorded_seed == seed, 'Training seed mismatch')
                    if group == 'supervised':
                        require(t['effective_configuration_sha256'] == sha(OUT / f'audit/effective/seed_{seed}/{name}.json'), 'Effective configuration mismatch')
                    status = 'fit_complete'
                elif (p / 'models' / f'{name}.pt').exists():
                    status = 'interrupted_without_completion_record'
                if evaluation.exists():
                    require(train.exists(), 'Evaluation without training')
                    e = read(evaluation)
                    require(e['checkpoint_sha256'] == t['model_sha256'], 'Evaluation checkpoint mismatch')
                    for artifact in e['artifacts']:
                        require(sha(ROOT / artifact['path']) == artifact['sha256'], 'Saved score/ID hash mismatch')
                    require(sha(p / 'configs' / f'{name}_threshold_lock.json') == e['threshold_lock_sha256'], 'Threshold lock mismatch')
                    status = 'fit_and_evaluation_complete'
                rows.append(dict(seed=seed, group=group, model=name, status=status))
    return rows

def summary(status, reason, rows):
    complete = sum(r['status'] in ('fit_complete', 'fit_and_evaluation_complete') for r in rows)
    lines = ['# Phase 11A execution summary', '', f'Status: **{status}**.', '', reason, '',
             f'Completed mandatory fits: **{complete}/35**. Scope: seven frozen models across seeds 42, 123, 456, 789, 1024.', '',
             '| Seed | Model | Status |', '| --- | --- | --- |']
    lines += [f"| {r['seed']} | {r['model']} | {r['status']} |" for r in rows]
    lines += ['', 'Existing final supervised fits are fresh final fits, not development substitutions. The prior interrupted seed-42 Transformer-L64 attempt is preserved separately if restarted. No seed is replaced and no checkpoint is selected using test outcomes.', '',
              'Graph models and Phase 11B are excluded. The frozen specification and development artifacts remain unchanged.', '',
              'Five-seed aggregation, independent metric recomputation, temporal paired differences, family diagnostics, RF/AE/OR comparisons, and complete runtime reporting are unavailable until all required fits and evaluations pass.', '',
              'Audit records and stage logs: `audit/phase11a/`.']
    SUMMARY.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run(module, *args):
    label = '_'.join((module, *args)).replace('--', '')
    path = AUDIT / (label + '.log')
    require(not path.exists(), 'Refusing to overwrite a prior Phase 11A stage log')
    print('START', label, flush=True)
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-B', '-u', '-m', 'src.final_runs.' + module, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f'{label} failed, exit {result.returncode}; see {path}')
    print('PASS', label, flush=True)

def main():
    AUDIT.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        check_lock()
        prior = AUDIT / 'prior_audit'
        prior.mkdir()
        for path in (OUT / 'audit').glob('*'):
            if path.is_file():
                shutil.copy2(path, prior / path.name)
        write(AUDIT / 'approval.json', dict(phase='11A', seeds=SEEDS, models=SCOPE,
              approval='User explicitly authorized Phase 11A only; preserve completed final fits after provenance checks and finish remaining same-seed configurations.',
              spec_sha256=sha(ROOT / 'results/final_spec_v1/final_experiment_spec.json')))
        rows = inventory()
        write(AUDIT / 'initial_inventory.json', rows)
        summary('PREFLIGHT', 'Checking all frozen prerequisites before additional fitting.', rows)
        for module in ('preflight', 'check_artifacts', 'check_seed_provenance', 'check_calibration', 'check_semantics'):
            run(module)
        # Preserve original approval; this invocation has the narrower explicit scope.
        shutil.copy2(prior / 'execution_approval.json', OUT / 'audit/execution_approval.json')
        check_lock()
        for path in (OUT / 'audit').glob('*check*.json'):
            shutil.copy2(path, AUDIT / path.name)
        write(AUDIT / 'source_lock.json', dict(sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in (ROOT / 'src/final_runs/phase11a.py', ROOT / 'src/final_runs/report11a.py')], original_runner_lock_sha256=sha(OUT / 'audit/runner_lock.json')))
        # Interrupted fit has no optimizer/RNG resume state. Preserve it in full;
        # rerun the identical seed/configuration from initialization, never warm-start.
        interrupted = [r for r in rows if r['status'] == 'interrupted_without_completion_record']
        for r in interrupted:
            require(r['seed'] == 42 and r['model'] == 'transformer_L64', 'Unexpected interrupted stage')
            source = folder(42, 'temporal').resolve()
            require(source == (OUT / 'seed_42/temporal').resolve(), 'Archive path mismatch')
            archive = AUDIT / 'interrupted_seed_42_temporal'
            shutil.move(str(source), str(archive))
            write(AUDIT / 'interruption.json', dict(reason='No active fitting process; epoch-one artifacts without completion record or optimizer/RNG resume state.', action='Preserve attempt and freshly initialize identical seed42/configuration; no development checkpoint reused', artifacts=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in archive.rglob('*') if p.is_file()]))
        for seed in SEEDS:
            for group, names in SCOPE.items():
                for name in names:
                    p = folder(seed, group)
                    args = ('--seed', str(seed), '--group', group, '--model', name)
                    if not (p / 'metrics' / f'{name}_training.json').exists():
                        run('worker', *args)
                    if not (p / 'metrics' / f'{name}_evaluation.json').exists():
                        run('evaluate', *args)
                    rows = inventory()
                    summary('RUNNING', f'Latest completed stage: seed {seed}, {name}.', rows)
        run('report11a')
        run('check_artifacts', '--final-phase11a')
        print('Phase 11A complete. STOP; no graph execution.', flush=True)
    except Exception as exc:
        write(AUDIT / 'STOP_FAILURE.json', dict(status='STOPPED', reason=str(exc), traceback=traceback.format_exc(), automatic_retry=False))
        summary('STOPPED / INCOMPLETE', str(exc), rows)
        raise

if __name__ == '__main__':
    main()
