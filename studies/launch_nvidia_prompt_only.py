"""Launch the fresh prompt-only evaluation using the existing official NVIDIA service.

Never starts/restarts a model server, builds an engine, or submits a readiness
inference request. Only the reviewed 4800 evaluation jobs are authorized here.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from legacy_d1_w8 import now, write, digest
from launch_nvidia_agents3_policy import identity, stable_identity, environment, model_ready
from run_nvidia_prompt_only import ROOT, MODEL, REVISION, ASSETS, validate_plan, require_parent_stopped


def read(path):
    return json.loads(Path(path).read_text())


def launch(root):
    out = root / 'experiment'
    require_parent_stopped()
    validate_plan(read(out / 'plan.json'))
    execution = read(out / 'execution_manifest.json')
    assert (execution['model'], execution['revision']) == (MODEL, REVISION)
    assert execution['planned_trials'] == 4800 and execution['qualification_trials'] == 0
    assert read(out / 'status.json')['state'] == 'frozen_ready'
    assert not (out / 'launch.json').exists(), 'Do not launch a second copy'
    assert not list((out / 'records').glob('*.json')), 'Fresh run must not reuse rows'
    service = read(ASSETS / 'services/launch.json')
    current = identity(service['pid'])
    assert current and current['process_starttime_ticks'] == service['process_starttime_ticks']
    assert model_ready()
    from runtime_reversal.batch_control import service_idle
    assert service_idle(service['base_url']), 'The existing model service still has in-flight work'
    with (ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [(_CHRONOTOPE_HOME + '/agents3-venv/bin/python'), '-B',
               str(out / 'code/run_nvidia_prompt_only.py'), '--out', str(out), '--run']
    env = environment(out)
    env.pop('CHRONOTOPE_AGENT_S3_POLICY', None)
    with (out / 'runner.log').open('ab') as log:
        runner = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(out / 'launch.json', {**stable_identity(runner, command), 'launch_command': command,
          'started_at': now(), 'automatic_resume': False, 'helper_sha256': digest(Path(__file__)),
          'service_reused': str(ASSETS / 'services/launch.json'), 'readiness_inference_calls': 0})
    deadline = time.monotonic() + 1200
    while read(out / 'status.json')['state'] == 'frozen_ready':
        if runner.poll() is not None:
            raise RuntimeError('Runner exited before running state; inspect runner.log')
        if time.monotonic() > deadline:
            raise TimeoutError('Runner did not enter running state')
        time.sleep(1)
    assert read(out / 'status.json')['state'] == 'running'
    command = [str(ROOT / '.venv/bin/python'), '-B', str(out / 'code/watch_same_model_deadline.py'),
               '--out', str(out)]
    with (out / 'auditor.log').open('ab') as log:
        auditor = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(out / 'audit_launch.json', {**stable_identity(auditor, command), 'launch_command': command,
                                    'started_at': now()})
    print(json.dumps({'state': 'running', 'runner_pid': runner.pid, 'auditor_pid': auditor.pid,
                      'out': str(out), 'planned_trials': 4800, 'qualification_trials': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    launch(args.root.resolve())
