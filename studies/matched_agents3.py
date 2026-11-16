"""Serial, resumable Agent S3/shared-proposer Chronotope comparison.

Prepare with --prepare; run the snapshotted script with --run. All generation is
real native Agent S3 generation, through agents3_matched_service.py. No generated
Python is executed. The manifest distinguishes native keep from added policies.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import ast
import base64
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3.8-27B+ByteDance-Seed/UI-TARS-1.5-7B'
TASKS = {
 'notification': ('Disappearing Notification', 'runs/r1_qwen38_notification_100/eval_w4000000000_s1000_live.jsonl'),
 'drift': ('Moving Balloon', 'runs/c1_qwen38_nothink_100/live_v8_frozen_s1000.jsonl'),
 'jump': ('Moving Balloon', 'runs/c2_qwen38_nothink_100/live_jump2s_frozen_s1000.jsonl'),
 'select': ('Moving Balloon', 'runs/c3_qwen38_nothink_100/live_v8_frozen_s1000.jsonl'),
 'code': ('Expiring Authenticator Code', 'runs/c4_qwen38_nothink_100/live_ch2_frozen_s1000.jsonl'),
 'button': ('Displaced Button', 'runs/g1_reflow_qwen38_nothink_100/live_ch2_frozen_s1000.jsonl'),
 'trading': ('Reversing Trading Signal', 'runs/b1_qwen38_nothink_100/prl_r2000_s204_live.jsonl'),
}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def parse_action(codes):
    """Translate the supported native action subset; never evaluate Python.

    A native type command may include a focus click, select-all and enter. The
    benchmark's type action is atomic and assumes a focused field, identically
    in both arms. Digit strings keep leading zeros. Unsupported commands fail.
    """
    if isinstance(codes, str):
        codes = [codes]
    if not isinstance(codes, list):
        return None
    clicks, values, waits = [], [], False
    for code in codes:
        if code.strip() == 'WAIT':
            waits = True
            continue
        try:
            tree = ast.parse(code)
        except (SyntaxError, TypeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ''
            name = node.func.attr
            try:
                args = [ast.literal_eval(a) for a in node.args]
                kw = {a.arg: ast.literal_eval(a.value) for a in node.keywords}
            except (ValueError, TypeError):
                continue
            if owner == 'pyautogui' and name == 'click':
                x, y = (args[:2] if len(args) >= 2 else (kw.get('x'), kw.get('y')))
                if type(x) in (int, float) and type(y) in (int, float):
                    clicks.append({'action': 'click', 'x': round(x), 'y': round(y)})
            if (owner == 'pyautogui' and name in ('write', 'typewrite')) or (owner == 'pyperclip' and name == 'copy'):
                if args and isinstance(args[0], str):
                    values.append(args[0])
            if owner in ('time', 'pyautogui') and name == 'sleep':
                waits = True
    # Native Unicode handling contains both clipboard and write fallback paths.
    if values and len(set(values)) == 1:
        return {'action': 'type', 'value': values[0]}
    if len(clicks) == 1 and not values:
        return clicks[0]
    if waits and not clicks and not values:
        return {'action': 'wait'}
    return None


class Adapter:
    def __init__(self, url, log):
        self.url, self.log, self.fresh = url, Path(log), True

    def spec_fields(self):
        return {'server': 'native_agents3_instrumented', 'model_id': MODEL}

    def invoke(self, png, system, user_text, **kwargs):
        from chronotope.controller import RawInvocation
        body = {'png_b64': base64.b64encode(png).decode(), 'instruction': system, 'reset': self.fresh}
        req = urllib.request.Request(self.url + '/predict', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            raw = json.load(response)
        self.fresh = False
        if raw.get('model_reported') != MODEL or 'error' in raw:
            raise RuntimeError(f'Invalid model response: {raw}')
        if not raw.get('model_calls') or any(c.get('usage') is None for c in raw['model_calls']):
            raise RuntimeError('Missing native model call accounting')
        with self.log.open('a') as f:
            f.write(json.dumps(raw) + '\n')
        return RawInvocation(parse_action(raw.get('exec_code')), f'agents3-{time.monotonic_ns()}',
            {'model_reported': MODEL, 'finish_reason': 'stop', 'input_tokens': raw['input_tokens'],
             'output_tokens': raw['output_tokens'], 'model_calls': len(raw['model_calls']),
             'planner_calls': sum(c['role'] == 'planner' for c in raw['model_calls']),
             'grounder_calls': sum(c['role'] == 'grounder' for c in raw['model_calls']),
             'raw_text': '\n'.join(raw['exec_code']), 'agent_elapsed_ns': raw['elapsed_ns']})


def setup(code):
    sys.path[:0] = [str(code / 'src'), str(code / 'chronotope/python')]
    os.environ['CHRONOTOPE_LIB'] = str(code / 'chronotope/build/libchronotope.so')
    from chronotope import spec
    spec._CONTROLLER_SCHEMAS['matched_agents3'] = {
        'kind': (str, ('matched_agents3',)), 'implementation': (str, ('host', 'chronotope')),
        'policy': (str, ('keep', 'check', 'restart', 'update')), 'model_id': (str, None),
        'service_url': (str, None), 'image_scale': (int, None), 'task': (str, None)}


def controller(spec, log):
    from chronotope.controller import (CallBudget, Correction, Decision,
                                      VLMLateBindingController, UITARS_INSTRUCTION)
    from chronotope.hashing import hash_obj
    from chronotope_core.controller import KGUAController, _VLMPlanner
    from chronotope.png import encode_png

    class NativeType(_VLMPlanner):
        def decide(self):
            obs = self._newest
            self._baseline_hash = obs.payload_hash
            self.budget.take()
            r = self.adapter.invoke(encode_png(obs.payload, self.w, self.h, self.scale),
                                    self.system, 'Screenshot attached.')
            p = r.parsed or {}
            value = p.get('value')
            valid = p.get('action') == 'type' and isinstance(value, str) and len(value) == 4 and value.isdigit()
            if valid:
                self._done = True
            return Decision({'kind': 'type', 'value': value} if valid else None, None,
                invocation_id=r.invocation_id, invocation_detail={**r.detail, 'observation_id': obs.observation_id,
                    'outcome': 'type' if valid else 'wait' if p.get('action') == 'wait' else 'unparseable'})
    c, m = spec.raw['controller'], spec.manifest()
    policy, task = c['policy'], c['task']
    common = dict(adapter=Adapter(c['service_url'], log), budget=CallBudget(100), prompt_version=5,
                  image_scale=c['image_scale'], invoke_on='frame_change', canvas_w_px=m.canvas_w_px,
                  canvas_h_px=m.canvas_h_px, skin=m.skin,
                  instruction_color=getattr(m, 'instruction_color', lambda: None)(), target_color=m.target_color())
    if task == 'code':
        instruction = ('Type the four-digit code displayed in the yellow panel into the text field below it. '
                       'The field is already focused. Preserve leading zeros. If no code is visible, wait.')
    elif task == 'button':
        instruction = ('Click the center of the purple Continue button on this page. '
                       'A banner may move the button. If the button is absent, wait.')
    elif task == 'trading':
        instruction = ('When a buy signal is active the order button in the right-hand panel turns green; '
                       'otherwise it is grey. Click the center of the green order button when visible. '
                       'If no green order button is visible, wait.')
    else:
        instruction = UITARS_INSTRUCTION['balloon_select' if task == 'select' else m.skin].format(
            color=common['instruction_color'] or '')
    mapped = {'keep': 'frozen', 'check': 'late_bind', 'restart': 'restart_if_moved', 'update': 'late_bind'}[policy]
    if task == 'code':
        if policy != 'keep':
            raise ValueError('Code updates are outside this matched execution study')
        inner = NativeType(**common)
    elif c['implementation'] == 'host' and policy != 'keep':
        inner = VLMLateBindingController(**common, at_commit_policy=mapped, template_radius_px=12,
                                         min_match_score=0.8, restart_threshold_px=6)
    else:
        inner = _VLMPlanner(**common)
    inner.system = instruction
    if task != 'code':
        inner.prompt_version = 17  # native Agent S3 pixel coordinates
    resident = c['implementation'] == 'chronotope'
    if resident:
        if task == 'code' and policy != 'keep':
            raise ValueError('Resident code updates are unsupported; never substitute a host decoder silently')
        inner = KGUAController(canvas_w_px=m.canvas_w_px, canvas_h_px=m.canvas_h_px, planner='vlm',
            binder='cuda', at_commit_policy='restart_if_moved' if policy == 'check' else mapped,
            template_radius_px=12, min_match_score=0.8,
            restart_threshold_px=2_000_000_000 if policy == 'check' else 6,
            target_color=None, rgb_template=bool(common['instruction_color']), vlm=inner)

    class Measured:
        def __init__(self):
            self.last, self.obs_b, self.timings = None, None, []

        def __getattr__(self, key):
            return getattr(inner, key)

        def spec_hash(self):
            return hash_obj({'matched_spec': c, 'inner': inner.spec_hash(), 'instruction': instruction})

        def decide(self):
            self.obs_b = inner._newest.observation_id
            if resident and task == 'code':
                b = inner.core.begin_decision(self.obs_b)
                d = inner.vlm.decide()
                if d.action is None:
                    inner.core.abort_decision()
                else:
                    inner.core.propose('type', value=int(d.action['value']))
                    d = replace(d, invocation_detail={**d.invocation_detail, 'device_b_ns': b['b_device_ns']})
            else:
                d = inner.decide()
            self.last = d.action
            return d

        def correct_at_commit(self, at):
            start = time.perf_counter_ns()
            if resident and task == 'code':
                r = inner.core.resolve_and_commit()
                if r['outcome_name'] != 'committed' or r['value'] != int(self.last['value']):
                    raise RuntimeError(f'Typed device commit failed: {r}')
                corr = Correction(self.last, 'state_at_B', self.obs_b,
                                  {'track_ns': r['track_ns'], 'track_device_ns': r['track_device_ns'],
                                   'typed_commit': r, 'value_source': 'native planner; no decoder'})
            else:
                correct = getattr(inner, 'correct_at_commit', None)
                corr = correct(at) if correct else None
            if not resident and policy == 'check' and isinstance(corr, Correction):
                corr = replace(corr, action=self.last, binding='state_at_B', observation_id=self.obs_b,
                               detail={**corr.detail, 'check_only': True})
            # Charge checks even when they keep the proposal unchanged.
            if corr is None and self.last:
                corr = Correction(self.last, 'state_at_B', self.obs_b, {'no_argument_update': True})
            elapsed = time.perf_counter_ns() - start
            self.timings.append({'stage': 'commit_control', 'wall_ns': elapsed})
            if corr is not None:
                corr = replace(corr, detail={**corr.detail, 'control_wall_ns': elapsed,
                                             'implementation': c['implementation'], 'study_policy': policy})
            return corr

        def close(self):
            if resident:
                inner.close()
    return Measured()


def prepare(out, n):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Manifest already exists; use --run to resume')
    templates = {}
    for task, (name, source) in TASKS.items():
        path = ROOT / source
        templates[task] = {'task_name': name, 'source': source, 'sha256': digest(path),
                           'spec': json.loads(path.open().readline())['spec']}
    jobs = []
    def add(phase, task, impl, policy, seed, modes, window=None):
        d = copy.deepcopy(templates[task]['spec'])
        s = d['scenario']; s['scenario_seed'] = seed
        if task == 'select':
            s['named_color'] = ('red', 'blue', 'yellow')[seed % 3]
        if window is not None:
            s['response_window_ns'] = int(window * 1e9)
            s['episode_horizon_ns'] = s['event_time_ns'] + s['response_window_ns'] + 500_000_000
        d['controller'] = dict(kind='matched_agents3', implementation=impl, policy=policy,
            model_id=MODEL, service_url='http://127.0.0.1:8110', image_scale=1 if task == 'button' else 2, task=task)
        d['task_clock_policy'], d['replay_source_live_trace_sha256'] = 'live', None
        key = f'{phase}_{task}_{impl}_{policy}_w{s["response_window_ns"]}_s{seed}'
        jobs.append(dict(id=key, phase=phase, task=task, implementation=impl, policy=policy, seed=seed,
                         modes=modes, spec=d))
    # Independent qualification set; report low competence without suppressing results.
    for seed in range(100, 110):
        for task in TASKS:
            for impl in (('host', 'chronotope') if seed % 2 == 0 else ('chronotope', 'host')):
                add('qualification', task, impl, 'keep', seed, ['paused'])
    for seed in range(1000, 1000+n):
        arms = ('host', 'chronotope') if seed % 2 == 0 else ('chronotope', 'host')
        for task in TASKS:
            for impl in arms:
                add('prl', task, impl, 'keep', seed, ['live', 'replay', 'paused'])
        for task in ('drift', 'jump', 'select', 'button'):
            for policy in ('check', 'restart', 'update'):
                for impl in arms:
                    add('policy', task, impl, policy, seed, ['live'])
        # Predeclared common deadlines. Includes original 16 s Jump rows above.
        for window in (4, 8, 32, 64):
            for policy in ('keep', 'restart', 'update'):
                for impl in arms:
                    add('window', 'jump', impl, policy, seed, ['live'], window)
    code = out / 'code'
    for rel in ('src', 'chronotope/python'):
        shutil.copytree(ROOT / rel, code / rel, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'assets/skin_browser', code / 'assets/skin_browser')
    (code / 'chronotope/build').mkdir(parents=True)
    shutil.copy2(ROOT / 'chronotope/build/libchronotope.so', code / 'chronotope/build/libchronotope.so')
    shutil.copy2(__file__, code / 'matched_agents3.py')
    for rel in ('agents3_matched_service.py', 'agents3_service.py'):
        shutil.copy2(ROOT / 'studies' / rel, code / rel)
    # Preserve native S3 source for audit (execution uses the installed environment).
    native = Path((_CHRONOTOPE_HOME + '/agents3-venv/lib/python3.12/site-packages/gui_agents/s3'))
    shutil.copytree(native, code / 'native_s3', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'n_evaluation_seeds': n,
        'model_pair': MODEL, 'templates': templates, 'jobs': jobs,
        'code_sha256': {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()},
        'protocol': 'Serial real-model trials; alternate controller order by seed; shared native S3 proposal pipeline. Native S3 is host/keep; all other host policies are explicit extensions. Fresh Live, Replay validated against it, independent Paused.',
        'limitations': ['Software-track fixed publication/delivery and lazy frame catch-up, not physical desktop end-to-end latency.',
            'Model inference and native S3 orchestration use shared host/vLLM services in both arms; Chronotope residency applies to control/commit, not the complete S3 stack.',
            'Resident typed argument updates are not implemented: code is keep-only in both matched arms.',
            'Trading is keep-only; coordinate updates cannot correct a reversed operation.',
            'No claim that a grounder changes only coordinates without semantic differences; paired proposal replay is a separate ablation.',
            'All qualification scores reported; poor competence is not filtered out of evaluation.'],
        'planned_trials': sum(len(j['modes']) for j in jobs)}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'jobs': len(jobs), 'trials': manifest['planned_trials'], 'out': str(out)}), flush=True)


def run_job(out, job):
    from chronotope.evaluate import evaluate_trace
    from chronotope.pair import validate_pair
    from chronotope.runner import PayloadStore, run_trial
    from chronotope.spec import TrialSpec
    from chronotope.trace import Trace
    directory = out / 'trials' / job['id']; directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f'attempt_{len(list(directory.glob("attempt_*")))+1:04d}'; attempt.mkdir()
    base = TrialSpec.from_dict(job['spec']); traces, results = {}, {}
    for mode in job['modes']:
        live = traces.get('live')
        spec = base.derive(mode, live.file_hash() if mode == 'replay' else None)
        ctl = controller(spec, attempt / f'{mode}_model_calls.jsonl')
        start = time.monotonic()
        try:
            path = run_trial(spec, attempt / f'{mode}.jsonl', PayloadStore(out / 'payloads'),
                             live_trace=live if mode == 'replay' else None, controller=ctl)
        finally:
            ctl.close()
        trace = Trace.load(path); r = evaluate_trace(trace)
        if trace.all('decision_failed') or not r.valid_trial or r.contract_violations:
            raise RuntimeError(f'Invalid {job["id"]}/{mode}: {r.to_dict()} {trace.all("decision_failed")}')
        invocations = trace.all('decision_invocation')
        results[mode] = {'evaluation': r.to_dict(), 'wall_s': time.monotonic()-start,
            'model_calls': sum(v.get('model_calls', 0) for v in invocations),
            'input_tokens': sum(v.get('input_tokens', 0) for v in invocations),
            'output_tokens': sum(v.get('output_tokens', 0) for v in invocations),
            'control_wall_ns': sum(t['wall_ns'] for t in ctl.timings),
            'trace': str(path.relative_to(out)), 'sha256': trace.file_hash()}
        traces[mode] = trace
    pair = validate_pair(traces['replay'], traces['live']) if 'replay' in traces else None
    if pair is not None and not pair.valid:
        raise RuntimeError(f'Replay validation failed: {pair.reason}')
    record = {'job': job, 'results': results, 'pair_valid': pair.valid if pair else None}
    write_json(attempt / 'result.json', record); write_json(directory / 'result.json', record)
    return record


def report(out, manifest, records, state, active=None, error=None):
    status = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'completed_jobs': len(records), 'planned_jobs': len(manifest['jobs']),
              'completed_trials': sum(len(r['results']) for r in records.values()),
              'planned_trials': manifest['planned_trials'], 'active': active, 'error': error, 'pid': os.getpid()}
    write_json(out / 'status.json', status)
    groups = defaultdict(list)
    for r in records.values():
        j = r['job']
        for mode, values in r['results'].items():
            groups[(j['phase'], j['task'], j['implementation'], j['policy'],
                    j['spec']['scenario']['response_window_ns']/1e9, mode)].append(values)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(['Phase','Task','Variant','Controller','Policy','Window (s)','Condition','n','SR','Return',
                'Mean model calls','Mean input tokens','Mean output tokens','Mean control wall (ms)',
                'Mean E-to-A (ms)','Mean O-to-D (ms)','Mean trial wall (s)'])
    for (phase,task,impl,policy,window,mode), rows in sorted(groups.items()):
        ev = [r['evaluation'] for r in rows]
        vals = [e['intervals_ns']['E_A']/1e6 for e in ev if e['intervals_ns'].get('E_A') is not None]
        od = [(e['intervals_ns']['O_B']+e['intervals_ns']['B_D'])/1e6 for e in ev
              if e['intervals_ns'].get('O_B') is not None and e['intervals_ns'].get('B_D') is not None]
        returns = [e['return_G'] for e in ev if e.get('return_G') is not None]
        w.writerow([phase,TASKS[task][0],task,impl,policy,window,mode,len(rows),
            sum(bool(e['valid_at_receipt']) for e in ev)/len(rows), mean(returns) if returns else '',
            *[mean(r[k] for r in rows) for k in ('model_calls','input_tokens','output_tokens')],
            mean(r['control_wall_ns'] for r in rows)/1e6,mean(vals) if vals else '',mean(od) if od else '',
            mean(r['wall_s'] for r in rows)])
    tmp = out / 'summary.tmp'; tmp.write_text(buf.getvalue()); tmp.replace(out / 'summary.csv')


def run(out, limit=None):
    lock = (out / 'runner.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((out / 'manifest.json').read_text())
    for rel, sha in manifest['code_sha256'].items():
        if digest(out / 'code' / rel) != sha:
            raise RuntimeError(f'Snapshot changed: {rel}')
    setup(out / 'code')
    records = {p.parent.name: json.loads(p.read_text()) for p in (out / 'trials').glob('*/result.json')}
    completed = 0
    for job in manifest['jobs']:
        if job['id'] in records:
            continue
        report(out, manifest, records, 'qualifying' if job['phase']=='qualification' else 'running', job['id'])
        print(job['id'], flush=True)
        try:
            records[job['id']] = run_job(out, job)
        except Exception:
            error = traceback.format_exc(); print(error, flush=True)
            write_json(out / 'last_error.json', {'job': job, 'traceback': error})
            report(out, manifest, records, 'infrastructure_error', job['id'], error)
            raise
        report(out, manifest, records, 'qualifying' if job['phase']=='qualification' else 'running')
        completed += 1
        if limit and completed >= limit:
            report(out, manifest, records, 'smoke_complete')
            return
    report(out, manifest, records, 'complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--run', action='store_true')
    p.add_argument('--n', type=int, default=100); p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.prepare: prepare(a.out.resolve(), a.n)
    if a.run: run(a.out.resolve(), a.limit)
