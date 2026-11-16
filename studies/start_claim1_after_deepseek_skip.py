"""Explicitly authorized dependency skip; keep the frozen evaluation unchanged."""
from contextlib import ExitStack
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/claim1_agents3_qwen08_remaining3_20261105'
OUT = RUN / 'experiment'
sys.path.insert(0, str(OUT / 'code'))
import run_claim1_agents3_remaining3 as study


def validate_authorization():
    authorization = study.read(RUN / 'dependency_skip_authorization.json')
    cfg = study.read(RUN / 'queue_manifest.json')
    assert authorization['action'] == 'skip_failed_deepseek_dependency_and_start_prepared_remaining3'
    assert authorization['user_request'] == 'skip that one?'
    assert authorization['supervisor_sha256'] == study.digest(Path(__file__))
    assert authorization['plan_sha256'] == study.digest(OUT / 'plan.json')
    assert authorization['execution_manifest_sha256'] == study.digest(OUT / 'execution_manifest.json')
    assert authorization['queue_manifest_sha256'] == study.digest(RUN / 'queue_manifest.json')
    predecessor = Path(cfg['predecessor'])
    assert authorization['predecessor_status_sha256'] == study.digest(predecessor / 'status.json')
    status = study.read(predecessor / 'status.json')
    assert status['state'] == 'failed' and status['completed_trials'] == 2576 and status['pid'] is None
    assert not study.same_process(cfg['predecessor_identity'])
    assert study.read(OUT / 'status.json')['state'] == 'prepared'
    assert not any((OUT / 'records').iterdir()) and not any((OUT / 'trials').iterdir())
    assert not (RUN / 'handoff.json').exists() and not (RUN / 'successor_launch.json').exists()
    study.validate(OUT)
    return authorization, cfg


def run():
    child = None

    def publish(state, **details):
        study.write(RUN / 'queue_status.json', dict(state=state, updated_at=study.now(),
            pid=os.getpid(), predecessor=str(study.PREDECESSOR), successor=str(OUT),
            predecessor_state='failed', predecessor_dependency='skipped_by_explicit_user_instruction',
            authorization=str(RUN / 'dependency_skip_authorization.json'), planned_trials=4500,
            automatic_resume=False, qualification_trials=0, **details))

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with ExitStack() as stack:
        queue_lock = stack.enter_context((RUN / 'queue.lock').open('a+'))
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authorization, cfg = validate_authorization()
        locks = []; inodes = set()
        for path in (ROOT / 'runs/legacy_d1_gpu.lock',
                     Path(cfg['predecessor_runner']).parents[1] / 'runs/legacy_d1_gpu.lock'):
            inode = (path.stat().st_dev, path.stat().st_ino)
            if inode in inodes:
                continue
            handle = stack.enter_context(path.open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle); inodes.add(inode)
        try:
            publish('checking_successor_configuration')
            study.check_services(OUT)
            study.write(RUN / 'handoff.json', dict(state='passed', checked_at=study.now(),
                predecessor_audit={'state': 'waived_by_user', 'reason': 'Explicitly skip failed DeepSeek dependency',
                                   'completed_trials': 2576, 'planned_trials': 3000},
                predecessor_dependency='skipped_by_explicit_user_instruction',
                authorization_sha256=study.digest(RUN / 'dependency_skip_authorization.json'),
                plan_sha256=authorization['plan_sha256'],
                execution_manifest_sha256=authorization['execution_manifest_sha256'],
                predecessor_process_exited=True, qualification_trials=0,
                queue_identity=study.process_identity(os.getpid())))
            command = [study.PYTHON, '-B', '-u', str(OUT / 'code/run_claim1_agents3_remaining3.py'),
                       'run', '--out', str(OUT)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH', None)
            with (RUN / 'evaluation.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(h.fileno() for h in locks))
                study.write(RUN / 'successor_launch.json', dict(pid=child.pid, command=command,
                    started_at=study.now(), planned_trials=4500, qualification_trials=0, successor_jobs=[],
                    dependency_skip_authorization=str(RUN / 'dependency_skip_authorization.json')))
                publish('running_successor', successor_pid=child.pid)
                returncode = child.wait()
            child = None
            status = study.read(OUT / 'status.json')
            if returncode != 0 or status['state'] != 'complete':
                raise RuntimeError(f'Successor stopped or failed (exit {returncode}); no automatic resume')
            publish('complete', completed_trials=4500)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate(); child.wait()
            publish('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate_authorization()
        print('Authorization and all 4,500 frozen evaluation specifications verified; no model calls.')
    else:
        run()
