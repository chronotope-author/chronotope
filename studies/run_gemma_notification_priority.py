"""Authorized priority handoff to Gemma; no qualification or inference probes."""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
import gemma_ledger_assets as assets
from core5_ledger_plan import validate_plan, load_cases
from legacy_d1_w8 import digest, now, write

OUT = ROOT / 'runs/notification_gemma4_26b_20261030'
EXP = OUT / 'experiment'
OLD_QWEN = ROOT / 'runs/legacy_detector_full_agents3_core5_qwen_20261029/experiment'
OLD_QUEUE = ROOT / 'runs/legacy_detector_full_agents3_gemma4_26b_20261029'
AGENT_PY = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
TOKEN_PY = (_CHRONOTOPE_HOME + '/qwen-serve/.venv/bin/python')
AUTHORIZATION = ('Can we pause whatever is currently running and do this with '
                 'nvidia/Gemma-4-26B-A4B-NVFP4 first, followed by '
                 'nvidia/Nemotron-3.5-Bolt-E4B-BF16?')

def read(path):
    return json.loads(Path(path).read_text())

def guard():
    for path in [OLD_QWEN / 'status.json', OLD_QUEUE / 'queue_status.json']:
        status = read(path)
        assert status['state'] == 'paused' and status['pid'] is None
        assert 'Gemma' in status['pause_authorization']
    plan = read(EXP / 'plan.json')
    assert validate_plan(plan) == 3000
    assert {j['task'] for j in plan['jobs']} == {'notification'}
    assert {j['window_s'] for j in plan['jobs']} == {16., 8., 4., 2., 1.}
    assert plan['qualification_trials'] == 0
    load_cases(EXP, plan)
    assert (plan['model'], plan['revision']) == (assets.MODEL, assets.REVISION)

def publish(state, **detail):
    write(OUT / 'priority_status.json', dict(state=state, updated_at=now(), pid=os.getpid(),
          model=assets.MODEL, revision=assets.REVISION, planned_trials=3000,
          qualification_trials=0, automatic_resume=False, **detail))

def run_command(name, argv, lock=None):
    folder = OUT / 'priority_phases'; folder.mkdir(exist_ok=True)
    report = folder / f'{name}.json'
    assert not report.exists(), 'No automatic phase retry'
    with (folder / f'{name}.log').open('xb') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=assets.environment(), stdin=subprocess.DEVNULL,
             stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),) if lock else ())
        proof = dict(state='running', command=argv, pid=proc.pid, started_at=now())
        write(report, proof); code = proc.wait()
        proof.update(state='complete' if code == 0 else 'failed', exit_code=code, completed_at=now())
        write(report, proof)
        if code: raise RuntimeError(f'{name} failed: {folder / (name + ".log")}')
    return folder / f'{name}.log'

def main():
    assets.OUT = OUT
    guard()
    with (OUT / 'priority.lock').open('a+') as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / 'priority_started.json').exists(), 'No automatic restart'
        write(OUT / 'priority_started.json', dict(at=now(), user_authorization=AUTHORIZATION,
              plan_sha256=digest(EXP / 'plan.json'), source_sha256=digest(__file__)))
        try:
            publish('waiting_for_checkpoint_download')
            while not (OUT / 'checkpoint_download.json').exists():
                launch = read(OUT / 'checkpoint_download_launch.json')
                path = Path(f"/proc/{launch['pid']}/stat")
                assert path.exists() and path.read_text().rsplit(')', 1)[1].split()[0] != 'Z', 'Checkpoint download exited without verification'
                time.sleep(5)
            assert read(OUT / 'checkpoint_download.json')['state'] == 'downloaded'
            shutil.copy2(assets.OLD / 'grounder_checkpoint_verification.json',
                         OUT / 'grounder_checkpoint_verification.json')
            with (ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for name, operation in [('stop_qwen_services', assets.stop_old_services),
                                        ('export', assets.export), ('build_engines', assets.build)]:
                    guard(); publish('preparing_model', phase=name); operation()
                for name, argv in [
                    ('verify_native', [TOKEN_PY, '-B', str(ROOT / 'studies/prepare_ledger_native.py'), '--root', str(OUT), '--verify-cpu', '--checkpoint', str(assets.CHECKPOINT)]),
                    ('build_native', [TOKEN_PY, '-B', str(ROOT / 'studies/prepare_ledger_native.py'), '--root', str(OUT), '--build', '--lock-fd', str(gpu.fileno())]),
                    ('verify_agents3', [AGENT_PY, '-B', str(ROOT / 'studies/verify_full_agents3.py'), '--config', str(EXP / 'configs/full_agent_components.json'), '--out', str(OUT / 'full_agents3_cpu_verification.json')])]:
                    guard(); publish('preparing_model', phase=name); run_command(name, argv, gpu)
                publish('preparing_model', phase='start_services'); assets.services()
                publish('preparing_model', phase='freeze')
                run_command('freeze', [AGENT_PY, '-B', str(ROOT / 'studies/run_core5_detector_full_agents3.py'),
                    '--out', str(EXP), '--freeze', '--services', str(OUT / 'services/launch.json')], gpu)
            guard(); publish('running_trials')
            run_command('evaluation', [AGENT_PY, '-B', str(EXP / 'code/run_core5_detector_full_agents3.py'), '--out', str(EXP), '--run'])
            status = read(EXP / 'status.json')
            assert status['state'] == 'complete' and status['completed_trials'] == status['planned_trials'] == 3000
            publish('auditing_completed_study')
            log = run_command('completion_audit', [AGENT_PY, '-B', str(EXP / 'code/run_core5_detector_full_agents3.py'), '--out', str(EXP), '--audit-completed'])
            proof = json.loads([x for x in log.read_text().splitlines() if x.strip()][-1])
            assert proof['state'] == 'passed' and proof['completed_trials'] == 3000
            write(OUT / 'completion_audit.json', proof)
            publish('complete', completed_trials=3000)
        except BaseException as exc:
            publish('failed', error=f'{type(exc).__name__}: {exc}')
            raise

if __name__ == '__main__':
    main()
