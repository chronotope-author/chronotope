"""Frozen, serial reproduction of the historical C++ TensorRT D1 configurations.

Only Notification and Moving Balloon drift; all windows are eight seconds.
Preserves the old model/control code. Adds explicit process cleanup, immutable
trial records, scope validation and continuously refreshed progress summaries.
"""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASKS = {'closing_window': 'Disappearing Notification', 'moving_target': 'Moving Balloon'}
ARMS = ('detector', 'keep', 'update')

def evaluation_jobs():
    return [dict(id=f'evaluation_{family}_{arm}_{index:03d}_{mode}',phase='evaluation',
                 family=family,arm=arm,index=index,mode=mode)
            for index in range(100) for family in TASKS for arm in ARMS
            for mode in ('live','replay','paused')]


def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def setup(code):
    sys.path[:0] = [str(code / 'src'), str(code / 'chronotope/python')]
    os.environ['CHRONOTOPE_BIN'] = str(code / 'chronotope/build/chronotope')
    os.environ['CHRONOTOPE_LIB'] = str(code / 'chronotope/build/libchronotope.so')
    os.environ['EDGELLM_PLUGIN_PATH'] = (_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so')
    os.environ['LD_LIBRARY_PATH'] = (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu:') + os.environ.get('LD_LIBRARY_PATH', '')

def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    code = out / 'code'
    for sub in ('src', 'chronotope/python', 'chronotope/src'):
        shutil.copytree(ROOT / sub, code / sub, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (code / 'chronotope/build').mkdir(parents=True)
    for name in ('chronotope', 'libchronotope.so'):
        shutil.copy2(ROOT / 'chronotope/build' / name, code / 'chronotope/build' / name)
    shutil.copy2(__file__, code / 'legacy_d1_w8.py')
    shutil.copy2(ROOT / 'AGENTS.md', out / 'AGENTS.md')
    setup(code)
    from chronotope.distributional.manifest import GenConfig, generate, build_manifest, verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec, trial_spec_for
    (out / 'configs').mkdir()
    (out / 'manifests').mkdir()
    (out / 'records').mkdir()
    engines = Path((_CHRONOTOPE_HOME + '/tensorrt-edgellm-workspace/Qwen3.5-0.8B/engines'))
    base = json.loads((ROOT / 'runs/d1/closing_window/cpp_qwen35_0p8b_detector/controller.json').read_text())
    for family, prompt in [('closing_window', 4), ('moving_target', 5)]:
        configs = {}
        for arm in ARMS:
            cfg = dict(base, prompt_version=prompt, image_scale=2,
                       at_commit_policy='frozen' if arm == 'keep' else 'late_bind',
                       binder='detector' if arm == 'detector' else 'cuda')
            assert cfg['planner'] == 'trt' and cfg['server'] == 'cpp_socket'
            configs[arm] = cfg
            write(out / 'configs' / f'{family}_{arm}.json', cfg)
        for phase, n, seed, seed_base in [('evaluation', 100, 26102812, 7400000)]:
            cfg = GenConfig(task_family=family, n=n, master_seed=seed, seed_base=seed_base,
                            empirical_windows_s=(8.0,), generous_deadline_s=8.0,
                            canvas_w_px=640 if family == 'moving_target' else 320,
                            canvas_h_px=480 if family == 'moving_target' else 240)
            ms = generate(cfg)
            for row in ms.rows:
                assert row.task_family in TASKS and row.window_ns == 8_000_000_000
                verify_row_against_manifest(row, build_manifest(row))
                for arm, controller in configs.items():
                    for mode in ('paused', 'live', 'replay'):
                        trial_spec_for(row, ControllerSpec(arm, controller), mode, '0' * 64 if mode == 'replay' else None)
            ms.freeze(out / 'manifests' / f'{family}_{phase}.jsonl')
    jobs = evaluation_jobs()
    hashes = {str(p.relative_to(out)): digest(p) for p in code.rglob('*') if p.is_file()}
    for sub in ('configs', 'manifests'):
        hashes.update({str(p.relative_to(out)): digest(p) for p in (out/sub).iterdir() if p.is_file()})
    external = {str(p): digest(p) for p in [engines/'llm/llm.engine', engines/'visual/visual.engine',
                engines/'llm/config.json', Path(os.environ['EDGELLM_PLUGIN_PATH'])]}
    plan = dict(created_at=now(), label='Historical D1 C++ TensorRT implementation', window_seconds=8,
                model='Qwen3.5-0.8B', runtime='TensorRT-Edge-LLM 0.10.0 / TensorRT 10.16.1; C++ process',
                precision='Historical engine; exact weight precision not yet independently verified',
                image_scale=2, task_names=TASKS, arms=list(ARMS), jobs=jobs, planned_trials=len(jobs),
                hashes=hashes, external_hashes=external,
                evaluation_policy='No separate qualification; report competence failures in all evaluation cells',
                scope='Notification and drift only. No jump/select/Pending Effect. No vLLM inference.',
                limitations=['GPU event records, no history embeddings fed to model',
                             'Host gym/socket orchestration and CPU parsing remain',
                             'Detector-first is a combined model/detector diagnostic',
                             'Drift image_scale fixed at 2; historical cell mixed 1 and 2',
                             'No authenticator/button/trading or Agent S3 comparison in this reproduction'])
    write(out/'manifest.json', plan)
    write(out/'status.json', dict(state='prepared', planned_trials=len(jobs), completed_trials=0, pid=None))
    print(json.dumps({'out': str(out), 'jobs': len(jobs), 'scope_validated': True}), flush=True)

def run(out):
    setup(out/'code')
    from chronotope.distributional.manifest import ManifestSet, build_manifest
    from chronotope.distributional.episode import ControllerSpec, trial_spec_for, make_controller, verify_trace_evidence, score_trace
    from chronotope.distributional.metrics import percentiles
    from chronotope.runner import run_trial, PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    from chronotope.controller import CallBudget
    plan = json.loads((out/'manifest.json').read_text())
    for rel, want in plan['hashes'].items():
        assert digest(out/rel) == want, f'Frozen input changed: {rel}'
    for path, want in plan['external_hashes'].items():
        assert digest(path) == want, f'Engine/plugin changed: {path}'
    assert plan['jobs'] == evaluation_jobs(), 'Only the complete evaluation-only job list is permitted'
    assert all(j['family'] in TASKS and j['arm'] in ARMS and j['mode'] in ('live','replay','paused') for j in plan['jobs'])
    lock = (out.parent/'legacy_d1_gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ms = {(family, phase): ManifestSet.load(out/'manifests'/f'{family}_{phase}.jsonl')
          for family in TASKS for phase in ('evaluation',)}
    assert all(row.window_ns == 8_000_000_000 and row.task_family in TASKS for m in ms.values() for row in m.rows)
    records = {}
    for p in (out/'records').glob('*.json'):
        r = json.loads(p.read_text()); assert digest(out/r['trace']) == r['trace_sha256']
        records[r['job']['id']] = r
    started = now()
    payloads = PayloadStore(out/'payloads')
    def publish(state, current=None, error=None):
        groups = defaultdict(list)
        for r in records.values():
            j = r['job']; groups[(j['phase'],j['family'],j['arm'],j['mode'])].append(r)
        rows = []
        for (phase,family,arm,mode), rs in sorted(groups.items()):
            os_ = [r['outcome'] for r in rs]
            row = {'Phase':phase,'Task':TASKS[family],'Variant':'drift' if family=='moving_target' else 'notification',
                   'Implementation':plan['label'],'Model':plan['model'],'Precision':plan['precision'],'Runtime':plan['runtime'],
                   'Policy':arm,'Condition':mode,'Window (s)':8,'n':len(rs),'Successes':sum(o['Y'] for o in os_),
                   'SR':sum(o['Y'] for o in os_)/len(rs),'No receipt':sum(not o['R'] for o in os_),
                   'Invalid':sum(o['invalid_trial'] for o in os_),
                   'Replay pairs valid':sum(r.get('pair_valid') is True for r in rs)}
            for label,key in [('E-to-O','L_EO'),('O-to-B','L_OB'),('B-to-D','L_BD'),('D-to-A','L_DA'),('E-to-A','L_EA')]:
                for pc,val in percentiles([o[key] for o in os_]).items():
                    if pc in ('p50','p95'):row[f'{label} {pc} (ms)'] = None if val is None else val/1e6
            rows.append(row)
        if rows:
            tmp = out/'summary.csv.tmp'
            with tmp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            tmp.replace(out/'summary.csv')
        write(out/'status.json',dict(state=state,started_at=started,updated_at=now(),
              pid=os.getpid() if state=='running' else None, planned_trials=plan['planned_trials'],
              completed_trials=len(records),current_job=current,error=error,automatic_resume=False))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for job in plan['jobs']:
            if job['id'] in records: continue
            publish('running',job)
            manifest = ms[job['family'],job['phase']]
            row = manifest.rows[job['index']]
            cfg = json.loads((out/'configs'/f"{job['family']}_{job['arm']}.json").read_text())
            ctrl = ControllerSpec(job['arm'], cfg)
            folder = out/'trials'/job['phase']/job['family']/job['arm']
            folder.mkdir(parents=True,exist_ok=True)
            path = folder/f"{row.episode_id}_{job['mode']}.jsonl"
            assert not path.exists(), f'Incomplete existing trace: {path}; explicit recovery required'
            live = Trace.load(folder/f'{row.episode_id}_live.jsonl') if job['mode']=='replay' else None
            spec = trial_spec_for(row,ctrl,job['mode'],live.file_hash() if live else None)
            scenario = build_manifest(row)
            controller = make_controller(spec,scenario,ctrl,budget=CallBudget(100),vlm_adapter=None)
            t0=time.monotonic()
            try:
                run_trial(spec,path,payloads,live_trace=live,controller=controller,manifest=scenario)
            finally:
                controller.close()
            trace=Trace.load(path)
            verify_trace_evidence(row,scenario,trace,manifest.frozen_hash)
            assert trace.header()['spec']['controller']==cfg
            outcome=score_trace(row,trace,manifest.frozen_hash,controller=ctrl.name,repeat_index=0)
            pair=validate_pair(trace,live) if live else None
            record=dict(job=job,outcome=outcome.to_dict(),trace=str(path.relative_to(out)),trace_sha256=trace.file_hash(),
                        pair_valid=pair.valid if pair else None,pair_reason=pair.reason if pair else None,
                        wall_seconds=time.monotonic()-t0,completed_at=now())
            write(out/'records'/f"{job['id']}.json",record)
            records[job['id']]=record
            print(f"{len(records)}/{plan['planned_trials']} {job['id']} success={outcome.Y} wall={record['wall_seconds']:.2f}s",flush=True)
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):
                raise RuntimeError(f'Invalid trial/pair: {job["id"]}: {record}')
            publish('running',job)
        publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',job if 'job' in locals() else None,f'{type(exc).__name__}: {exc}')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare == args.run: parser.error('Choose exactly one of --prepare or --run')
    (prepare if args.prepare else run)(args.out.resolve())
