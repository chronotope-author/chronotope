"""Run prompt-only legacy C++ and unchanged stripped Agent S3 on NVIDIA Qwen3.8-27B.
Evaluation only; no competence filtering, extra smoke episodes, or implicit resume.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import urllib.request
from legacy_d1_w8 import digest, now, write, setup as legacy_setup

ROOT = Path(__file__).resolve().parents[1]
MODEL='nvidia/Qwen3.8-27B-NVFP4'
REVISION='482ca0f3832238542f8f5295dde86b5f22711d80'
PRECISION='NVFP4 MLP/lm_head; FP8 attention; BF16 vision'
CHECKPOINT=ROOT/'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f'
PARENT=ROOT/'runs/agents3_policy_nvidia27b_20261029/experiment'
ASSETS=PARENT.parent
RUNTIME_PRECISION='NVFP4 MLP/lm_head; FP8 attention projections; FP16 vision and unquantized LLM tensors'
PLUGIN=ROOT/'runs/runtime_reversal_stock_20261029/build/libNvInfer_edgellm_plugin.so'
ARMS=('legacy_cpp','agents3_stripped')
WINDOWS=(.125,.25,.5,1.,2.,4.,8.,16.)


def setup(code):
    legacy_setup(code)
    sys.path.insert(0,str(code))
    os.environ['EDGELLM_PLUGIN_PATH']=str(PLUGIN)


def validate_plan(plan):
    assert set(plan['models'])=={'nvidia_27b'}, 'One identical model in both arms'
    model=plan['models']['nvidia_27b']
    assert (model['id'],model['revision'],model['precision'])==(MODEL,REVISION,PRECISION)
    assert plan['planned_trials']==4800 and plan['qualification_trials']==0
    jobs=plan['jobs'];assert len(jobs)==4800
    keys=set();groups=defaultdict(set)
    for j in jobs:
        assert j['phase']=='evaluation' and j['task']=='notification' and j['family']=='closing_window'
        assert j['model']=='nvidia_27b' and j['arm'] in ARMS and j['window_s'] in WINDOWS
        assert j['mode'] in ('live','replay','paused') and 0<=j['index']<100
        k=(j['arm'],j['window_s'],j['index'],j['mode']);assert k not in keys;keys.add(k)
        groups[j['arm'],j['window_s'],j['mode']].add(j['index'])
    assert len(groups)==48 and all(v==set(range(100)) for v in groups.values())


def job_id(j):
    return f"{j['arm']}_w{round(j['window_s']*1e9)}_{j['index']:03d}_{j['mode']}"


def run(out, resume=False):
    require_parent_stopped()
    setup(out/'code');sys.path.insert(0,str(out/'code'))
    from agents3_stripped import StrippedAgentS3Adapter
    from audit_nvidia_prompt_only import audit_trial_requests
    from chronotope.distributional.manifest import ManifestSet,build_manifest
    from chronotope.distributional.episode import ControllerSpec,trial_spec_for,make_controller,verify_trace_evidence,score_trace
    from chronotope.distributional.metrics import percentiles
    from chronotope.controller import CallBudget
    from chronotope.runner import run_trial,PayloadStore
    from chronotope.trace import Trace
    from chronotope.pair import validate_pair
    plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
    cfg=json.loads((out/'execution_manifest.json').read_text())
    os.environ.pop('CHRONOTOPE_AGENT_S3_POLICY',None)
    os.environ['CHRONOTOPE_AGENT_S3_PROMPT']=str(out/'configs/agents3_prompt.json')
    for rel,sha in cfg['hashes'].items():assert digest(out/rel)==sha, rel
    for path,sha in {**cfg['checkpoint_hashes'],**cfg['engine_hashes']}.items():assert digest(path)==sha,path
    assert digest(cfg['plugin']['path'])==cfg['plugin']['sha256']
    with urllib.request.urlopen(cfg['model_service']+'/models',timeout=15) as response:models=json.load(response)
    assert [m['id'] for m in models['data']]==[MODEL]
    lock=(out.parent.parent/'legacy_d1_gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifests={name:ManifestSet.load(out/'manifests'/name) for name in plan['manifest_hashes']}
    configs={arm:json.loads((out/'configs'/f'{arm}.json').read_text()) for arm in ARMS}
    previous=json.loads((out/'status.json').read_text())
    existing=list((out/'records').glob('*.json'))
    if existing and not resume:raise RuntimeError('Use explicit --resume after validating partial records')
    records=[json.loads(p.read_text()) for p in sorted(existing)]
    planned={job_id(j):j for j in plan['jobs']}
    completed=set()
    for record in records:
        ident=job_id(record['job'])
        assert ident in planned and record['job']==planned[ident] and ident not in completed
        assert digest(out/record['trace'])==record['trace_sha256']
        assert not record['outcome']['invalid_trial'] and not record['outcome']['contract_violations']
        completed.add(ident)
    started=previous.get('started_at',now()) if resume else now();current=None
    def publish(state,error=None):
        groups=defaultdict(list)
        for r in records:groups[r['job']['arm'],r['job']['window_s'],r['job']['mode']].append(r)
        rows=[]
        for (arm,w,mode),rs in sorted(groups.items()):
            outcomes=[r['outcome'] for r in rs]
            row={'Task':'Disappearing Notification','Controller':cfg['arm_labels'][arm],
                'Model':MODEL,'Revision':REVISION,'Source precision':PRECISION,'Precision':RUNTIME_PRECISION,
                'Runtime':'TensorRT-Edge-LLM' if arm=='legacy_cpp' else 'vLLM',
                'KV cache':'FP16',
                'History':'none' if arm=='legacy_cpp' else 'CPU RAM text',
                'Reflection':False if arm=='legacy_cpp' else True,
                'Max formatting attempts':1 if arm=='legacy_cpp' else 3,
                'Input match scope':'first planner request only',
                'Window (s)':w,'Condition':mode,'n':len(rs),'Successes':sum(o['Y'] for o in outcomes),
                'SR':sum(o['Y'] for o in outcomes)/len(rs),'No receipt':sum(not o['R'] for o in outcomes),
                'Invalid':sum(o['invalid_trial'] for o in outcomes),'Replay pairs valid':sum(r['pair_valid'] is True for r in rs)}
            for label,key in [('E-to-O','L_EO'),('O-to-B','L_OB'),('B-to-D','L_BD'),('D-to-A','L_DA'),('E-to-A','L_EA')]:
                for pc,val in percentiles([o[key] for o in outcomes]).items():
                    if pc in ('p50','p95'):row[f'{label} {pc} (ms)']=None if val is None else val/1e6
            od=[o['D']-o['O'] for o in outcomes if o['D'] is not None and o['O'] is not None]
            row['O-to-D p50 (ms)']=None if not od else percentiles(od)['p50']/1e6
            rows.append(row)
        if rows:
            with (out/'summary.csv.tmp').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
            (out/'summary.csv.tmp').replace(out/'summary.csv')
        write(out/'status.json',{'state':state,'started_at':started,'updated_at':now(),'completed_trials':len(records),
            'planned_trials':4800,'pid':os.getpid() if state=='running' else None,'current_job':current,
            'qualification_trials':0,'model':MODEL,'error':error,'automatic_resume':False})
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in plan['jobs']:
            ident=job_id(current)
            if ident in completed:continue
            publish('running');arm=current['arm'];mode=current['mode']
            m=manifests[current['manifest']];row=m.rows[current['index']];scenario=build_manifest(row)
            ctrl=ControllerSpec(arm,configs[arm])
            folder=out/'trials'/f"w{round(current['window_s']*1e9)}"/arm;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{row.episode_id}_{mode}.jsonl';assert not path.exists(),path
            live=Trace.load(folder/f'{row.episode_id}_live.jsonl') if mode=='replay' else None
            spec=trial_spec_for(row,ctrl,mode,live.file_hash() if live else None)
            adapter=StrippedAgentS3Adapter(cfg['model_service'],out/'calls'/f'{ident}.jsonl',model_id=MODEL,max_tokens=configs[arm]['max_tokens']) if arm=='agents3_stripped' else None
            os.environ['CHRONOTOPE_AGENT_S3_AUDIT']=str(out/'calls'/f'{ident}.jsonl')
            controller=make_controller(spec,scenario,ctrl,budget=CallBudget(100),vlm_adapter=adapter)
            t0=time.monotonic()
            try:run_trial(spec,path,PayloadStore(out/'payloads'),live_trace=live,controller=controller,manifest=scenario)
            finally:
                close=getattr(controller,'close',None)
                if close:close()
            trace=Trace.load(path)
            native_errors=[e for e in trace.all('decision_invocation') if e.get('planner')=='tensorrt_edgellm' and e.get('outcome')=='unparseable']
            if trace.all('decision_failed') or native_errors or (adapter and adapter.errors):raise RuntimeError(f'Controller/runtime error: {ident}')
            request_audit=audit_trial_requests(out,ident,arm,trace,scenario,cfg)
            verify_trace_evidence(row,scenario,trace,m.frozen_hash)
            assert trace.header()['spec']['controller']==configs[arm]
            outcome=score_trace(row,trace,m.frozen_hash,controller=arm,repeat_index=0)
            pair=validate_pair(trace,live) if live else None
            rec={'job':current,'request_audit':request_audit,'outcome':outcome.to_dict(),'trace':str(path.relative_to(out)),
                 'trace_sha256':trace.file_hash(),'pair_valid':pair.valid if pair else None,'pair_reason':pair.reason if pair else None,
                 'wall_seconds':time.monotonic()-t0,'completed_at':now()}
            write(out/'records'/f'{ident}.json',rec);records.append(rec)
            if outcome.invalid_trial or outcome.contract_violations or (pair and not pair.valid):raise RuntimeError(f'Invalid trace/pair: {ident}')
            print(f'{len(records)}/4800 {ident} success={outcome.Y}',flush=True);publish('running')
        current=None;publish('complete')
    except BaseException as exc:
        publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',f'{type(exc).__name__}: {exc}');raise


def read(path):
    return json.loads(Path(path).read_text())


def verify_hashes(mapping):
    assert mapping
    for path, expected in mapping.items():
        source = Path(path)
        if not source.is_absolute():
            source = ROOT / source
        assert digest(source) == expected, path


def require_parent_stopped():
    status = read(PARENT / 'status.json')
    assert status['state'] != 'running', 'Stop the superseded Worker-port batch first'
    if status.get('pid'):
        proc = Path('/proc') / str(status['pid']) / 'cmdline'
        assert not proc.exists() or str(PARENT).encode() not in proc.read_bytes(), 'Predecessor process is still active'


def same_cohort(plan):
    parent = read(PARENT / 'plan.json')
    assert plan['manifest_hashes'] == parent['manifest_hashes']
    assert plan['jobs'] == parent['jobs'], 'The original 4800 ordered evaluation cases must be retained'


def validate_specs(out):
    plan = read(out / 'plan.json')
    validate_plan(plan)
    same_cohort(plan)
    setup(out / 'code')
    from chronotope.distributional.manifest import ManifestSet, build_manifest, verify_row_against_manifest
    from chronotope.distributional.episode import ControllerSpec, trial_spec_for
    configs = {arm: read(out / 'configs' / f'{arm}.json') for arm in ARMS}
    for key in ('model_id', 'prompt_version', 'max_tokens', 'image_scale', 'temperature', 'enable_thinking', 'invoke_on'):
        assert configs[ARMS[0]][key] == configs[ARMS[1]][key], key
    assert configs['legacy_cpp']['planner'] == 'trt'
    assert configs['legacy_cpp']['server'] == 'cpp_socket'
    assert configs['legacy_cpp']['at_commit_policy'] == 'frozen'
    assert configs['agents3_stripped']['model_id'] == MODEL
    manifests = {name: ManifestSet.load(out / 'manifests' / name) for name in plan['manifest_hashes']}
    for name, manifest in manifests.items():
        assert manifest.frozen_hash == plan['manifest_hashes'][name] and len(manifest.rows) == 100
        for row in manifest.rows:
            verify_row_against_manifest(row, build_manifest(row))
    prior = set()
    for job in plan['jobs']:
        row = manifests[job['manifest']].rows[job['index']]
        assert row.task_family == 'closing_window' and row.deadline_s == job['window_s']
        assert row.episode_id == job['episode_id']
        key = (job['arm'], job['window_s'], job['index'], job['mode'])
        if job['mode'] == 'replay':
            assert key[:3] + ('live',) in prior
        prior.add(key)
        trial_spec_for(row, ControllerSpec(job['arm'], configs[job['arm']]), job['mode'],
                       '0' * 64 if job['mode'] == 'replay' else None)
    return len(plan['jobs'])


def prepare(out):
    assert not out.exists(), 'Never overwrite an experiment'
    plan = read(PARENT / 'plan.json')
    plan.update(created_at=now(), state='prepared', qualification_trials=0,
                supersedes=str(PARENT), predecessor_plan_sha256=digest(PARENT / 'plan.json'),
                comparison='Legacy C++ stateless prompt/parser only versus unchanged Python Agent S3 Worker; first planner request matched')
    plan['policy_difference'] = {
        'legacy_cpp': 'One stateless model request per decision; no Worker history, reflection, or retries',
        'agents3_stripped': 'Original Python Worker CPU text history, reflection, and retries retained',
        'matching_scope': 'First planner request only; later requests may differ by original controller behavior',
    }
    validate_plan(plan)
    same_cohort(plan)
    out.mkdir(parents=True)
    shutil.copytree(PARENT / 'manifests', out / 'manifests')
    (out / 'configs').mkdir()
    code = out / 'code'
    for sub in ('src', 'specs', 'gui_agents', 'chronotope/python', 'audit_deps'):
        source = PARENT / 'code' / sub
        if source.exists():
            shutil.copytree(source, code / sub, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('agents3_stripped.py', 'agents3_policy_contract.py', 'legacy_d1_w8.py', 'prepare_deadline_envelope.py'):
        shutil.copy2(PARENT / 'code' / name, code / name)
    shutil.copy2(PARENT / 'code/audit_agents3_policy.py', code / 'worker_policy_audit.py')
    for name in ('run_nvidia_prompt_only.py', 'audit_nvidia_prompt_only.py'):
        shutil.copy2(ROOT / 'studies' / name, code / name)
    watcher = (PARENT / 'code/watch_same_model_deadline.py').read_text()
    old = 'Matched policy and initial model request; generated outputs and later histories may differ across inference runtimes. Separate inference and control timing before overhead-only attribution'
    new = 'First planner request matched only. Native remains stateless; Python Worker history/reflection/retries remain enabled. This is not a matched-policy or overhead-only comparison.'
    assert old in watcher
    (code / 'watch_same_model_deadline.py').write_text(watcher.replace(old, new))
    for arm in ARMS:
        cfg = read(PARENT / 'configs' / f'{arm}.json')
        cfg['base_url'] = str(ASSETS / 'engines') if arm == 'legacy_cpp' else 'http://127.0.0.1:8012/v1'
        write(out / 'configs' / f'{arm}.json', cfg)
    worker = read(PARENT / 'configs/agents3_policy.json')
    write(out / 'configs/host_worker_policy.json', worker)
    prompt = {key: worker[key] for key in ('planner_system', 'initial_user_text', 'text_buffer_suffix',
                                         'screenshot_width', 'screenshot_height', 'generation_settings')}
    prompt.update(schema='agents3-prompt-only-1', model=MODEL, revision=REVISION,
                  source_worker_policy_sha256=worker['policy_sha256'])
    setup(code)
    from audit_nvidia_prompt_only import prompt_fingerprint, initial_messages, canonical
    from worker_policy_audit import model_input
    from agents3_policy_contract import PolicyReference, build_policy_contract
    assert build_policy_contract(model_id=MODEL) == worker, 'Unchanged host Worker contract must reproduce'
    prompt['prompt_sha256'] = prompt_fingerprint(prompt)
    write(out / 'configs/agents3_prompt.json', prompt)
    first = []
    def fixed_response(role, messages):
        first.append((role, messages))
        return '```python\nagent.click(x=270, y=225)\n```'
    PolicyReference(worker).run_turn(fixed_response)
    assert first == [('planner', initial_messages(prompt))]
    rendered = model_input(initial_messages(prompt), str(CHECKPOINT), 300)
    inputs = out.parent / 'inputs'
    inputs.mkdir(exist_ok=True)
    write(inputs / 'agents3_prompt.json', prompt)
    write(inputs / 'first_request_reference.json', {'messages': initial_messages(prompt), **rendered})
    write(out / 'plan.json', plan)
    checks = validate_specs(out)
    report = {'state': 'passed', 'created_at': now(), 'spec_checks': checks,
              'qualification_trials': 0, 'model_inference_calls': 0, 'experimental_trials': 0,
              'same_ordered_cohort': True, 'first_request_matches_unchanged_worker': True,
              'prompt_sha256': prompt['prompt_sha256'], 'expected_first_input_tokens': rendered['input_tokens'],
              'first_request_text_sha256': rendered['text_sha256'], 'matching_scope': 'first planner request only',
              'host_source_hashes': {str(code / name): digest(code / name) for name in ('agents3_stripped.py', 'agents3_policy_contract.py')},
              'parent_plan_sha256': digest(PARENT / 'plan.json')}
    write(out / 'preparation_report.json', report)
    write(out / 'status.json', {'state': 'prepared', 'completed_trials': 0, 'planned_trials': 4800,
                               'qualification_trials': 0, 'pid': None, 'model': MODEL, 'automatic_resume': False})
    print(json.dumps(report), flush=True)


def freeze(out):
    require_parent_stopped()
    assert not (out / 'execution_manifest.json').exists(), 'Already frozen'
    root = out.parent
    native = read(root / 'native_build_result.json')
    assert native['state'] == 'complete' and native['source_patches'] == []
    assert native['upstream_commit'] == 'bb291453bc10d9bd1142070c605e090d4e483ee2'
    assert native['model_inference_calls'] == native['qualification_trials'] == 0
    verify_hashes(native['source_hashes'])
    verify_hashes(native['binary_hashes'])
    verification = read(root / 'native_prompt_verification.json')
    assert verification['state'] == 'passed'
    assert verification['first_request_text_equal'] and verification['first_request_messages_equal']
    assert verification['action_parser_equal'] and verification['worker_history_reflection_retry_state_absent']
    assert verification['native_handle_request_calls_in_source'] == 1
    assert verification['model_inference_calls'] == verification['experimental_trials'] == 0
    verify_hashes(verification['source_file_sha256'])
    assets = read(ASSETS / 'engine_build_result.json')
    assert (assets['model'], assets['revision'], assets['state']) == (MODEL, REVISION, 'built')
    assert assets['source_patches'] == [] and assets['kv_cache']['dtype'] == 'FP16'
    verify_hashes(assets['hashes'])
    verify_hashes(assets['source_checkpoint_hashes'])
    service = read(ASSETS / 'services/launch.json')
    assert service['model'] == MODEL and service['revision'] == REVISION
    assert service['dtype'] == 'half' and service['effective_kv_cache_dtype'] == 'float16'
    assert service['quantization'] == 'modelopt_mixed' and not service['enable_prefix_caching']
    code = out / 'code'
    source = root / 'build_source/chronotope'
    assert not (source / 'src/agents3_policy.h').exists(), 'Do not compile the Worker port into the replacement'
    for path, expected in verification['source_file_sha256'].items():
        relative = Path(path)
        if relative.is_absolute():
            relative = relative.relative_to(ROOT)
        if str(relative).startswith('chronotope/src/'):
            assert digest(source / relative.relative_to('chronotope')) == expected, 'Built source differs from CPU-tested source'
    shutil.copytree(source / 'src', code / 'chronotope/src')
    shutil.copy2(source / 'CMakeLists.txt', code / 'chronotope/CMakeLists.txt')
    # The build manifest must bind the Python socket wrapper to the same snapshot.
    remote = ROOT / 'chronotope/python/chronotope_core/remote.py'
    assert str(remote) in native['source_hashes'], 'Native build provenance must include the socket wrapper'
    shutil.copy2(remote, code / 'chronotope/python/chronotope_core/remote.py')
    (code / 'chronotope/build').mkdir(parents=True)
    for name in ('chronotope', 'libchronotope.so'):
        shutil.copy2(root / 'native_build' / name, code / 'chronotope/build' / name)
    for name in ('run_nvidia_prompt_only.py', 'audit_nvidia_prompt_only.py'):
        shutil.copy2(ROOT / 'studies' / name, code / name)
    for sub in ('records', 'calls'):
        (out / sub).mkdir(exist_ok=True)
    setup(code)
    from agents3_policy_contract import build_policy_contract
    from audit_nvidia_prompt_only import prompt_fingerprint
    host = read(out / 'configs/host_worker_policy.json')
    assert build_policy_contract(model_id=MODEL) == host
    for name in ('agents3_stripped.py', 'agents3_policy_contract.py'):
        assert digest(code / name) == digest(PARENT / 'code' / name), 'Host Worker adapter changed'
    prompt = read(out / 'configs/agents3_prompt.json')
    assert prompt_fingerprint(prompt) == prompt['prompt_sha256']
    checks = validate_specs(out)
    shutil.copy2(ROOT / 'AGENTS.md', out / 'AGENTS.md')
    for name in ('native_build_result.json', 'native_prompt_verification.json'):
        shutil.copy2(root / name, out / name)
    shutil.copy2(ASSETS / 'services/launch.json', out / 'service_launch.json')
    artifacts = {str(p.relative_to(out)): digest(p) for sub in ('code', 'configs', 'manifests')
                 for p in (out / sub).rglob('*') if p.is_file()}
    for name in ('plan.json', 'preparation_report.json', 'AGENTS.md', 'native_build_result.json',
                 'native_prompt_verification.json', 'service_launch.json'):
        artifacts[name] = digest(out / name)
    execution = {
        'created_at': now(), 'model': MODEL, 'revision': REVISION,
        'precision': PRECISION, 'runtime_precision': RUNTIME_PRECISION, 'kv_cache': assets['kv_cache'],
        'arms': ARMS, 'planned_trials': 4800, 'qualification_trials': 0, 'spec_checks': checks,
        'checkpoint_path': str(CHECKPOINT), 'image_tokens': 300,
        'checkpoint_hashes': assets['source_checkpoint_hashes'], 'engine_hashes': assets['hashes'],
        'hashes': artifacts, 'plugin': {'path': str(PLUGIN), 'sha256': digest(PLUGIN)},
        'model_service': service['base_url'], 'reused_service_launch': str(ASSETS / 'services/launch.json'),
        'native_prompt_contract': prompt, 'host_worker_contract': host,
        # The watcher uses this only to choose the explicit first-request attribution text.
        'request_contract': {'scope': 'first planner request only', 'prompt_sha256': prompt['prompt_sha256']},
        'arm_labels': {'legacy_cpp': 'Legacy C++ (prompt/parser only; stock TensorRT-Edge-LLM)',
                       'agents3_stripped': 'Agent S3 (stripped; original Python Worker)'},
        'upstream_commit': native['upstream_commit'], 'source_patches': [],
        'native_build': native, 'engine_provenance': str(ASSETS / 'engine_build_result.json'),
        'interpretation': [
            'Fresh paired evaluation rows; no previous results reused.',
            'First planner prompt, screenshot and generation settings matched.',
            'Native remains stateless with one call per decision; no Worker history/reflection/format retries.',
            'Python stripped Agent S3 Worker remains unchanged, including CPU history/reflection/retries.',
            'Subsequent requests may differ by controller policy; not a matched-policy or overhead-only comparison.',
            'Official NVIDIA checkpoint retained; NVFP4/FP8 projections, FP16 floating runtime tensors and KV.',
            'Stock public image API requires an application-side host screenshot copy.',
            'This legacy path does not supply model-consumed GPU interaction-history embeddings.',
        ],
    }
    write(out / 'execution_manifest.json', execution)
    write(out / 'status.json', {'state': 'frozen_ready', 'completed_trials': 0, 'planned_trials': 4800,
                               'pid': None, 'qualification_trials': 0, 'model': MODEL, 'automatic_resume': False})
    print(json.dumps({'state': 'frozen_ready', 'spec_checks': checks, 'model': MODEL,
                      'matching_scope': 'first planner request only'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare', 'freeze', 'run'):
        group.add_argument('--' + action, action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.prepare:
        prepare(args.out.resolve())
    elif args.freeze:
        freeze(args.out.resolve())
    else:
        run(args.out.resolve(), resume=args.resume)

