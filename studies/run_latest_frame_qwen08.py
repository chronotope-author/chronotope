"""Fresh, interleaved Live evaluation: legacy detector, Agent S3, latest-frame.

No qualifications, retries of failed cases, automatic resume or successor jobs.
All model calls retain the frozen Agent S3 policy. CPU-only preparation/audit.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'runs/notification_qwen08_edgellm0111_20261031/experiment'
OTHER = ROOT/'runs/claim1_agents3_qwen08_remaining3_20261105/experiment'
PYTHON = (_CHRONOTOPE_HOME + '/agents3-venv/bin/python')
MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
TASKS = {'notification':'Disappearing Notification', 'button':'Displaced Button',
         'trading':'Reversing Trading Signal'}
ARMS = ('legacy_cpp_detector','agents3_full','agents3_latest_frame')
LABELS = {'legacy_cpp_detector':'Legacy C++ + detector', 'agents3_full':'Agent S3 (Keep)',
          'agents3_latest_frame':'Agent S3 + latest-frame updater'}
SETTINGS = (16,8,4,2,1)
TOTAL = 4500


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def verify(hashes,base=Path('/')):
    for path,h in hashes.items():
        if digest(base/path)!=h: raise ValueError('Frozen artifact changed: '+path)
def setup(out):
    code=out/'code'
    sys.path[:0]=[str(code),str(code/'src'),str(code/'chronotope/python')]
    from chronotope.latest_frame import register
    from chronotope_core.glue import register as native_register
    register();native_register()
def csv_write(path,rows):
    if not rows: return
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def prepare(out):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copytree(REFERENCE/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run_latest_frame_qwen08.py','run_claim1_agents3_remaining3.py','prepare_ledger_native.py','latest_frame_cuda.cu'):
        shutil.copy2(ROOT/'studies'/name,out/'code'/name)
    for name in ('latest_frame.py','runner.py'):
        shutil.copy2(ROOT/'src/chronotope'/name,out/'code/src/chronotope'/name)
    for folder in ('configs','cases','records','trials','calls','payloads','build'):
        (out/folder).mkdir()
    lib=out/'build/liblatest_frame.so'
    shutil.copy2(ROOT/'runs/latest_frame_qwen08_20261106/build/liblatest_frame.so',lib)
    # All model/native/scoring sources other than the small harness hook are frozen unchanged.
    assert digest(out/'code/src/chronotope/evaluate.py')==digest(REFERENCE/'code/src/chronotope/evaluate.py')
    setup(out)
    from chronotope.controller import SYSTEM_PROMPTS
    from prepare_ledger_native import TASK_VERSIONS
    from run_core5_detector_full_agents3 import task_instruction
    cfg=copy.deepcopy(read(REFERENCE/'execution_manifest.json'))
    cfg.update(created_at=now(),planned_trials=TOTAL,runnable_trials=TOTAL,deferred_trials=0,
        schema='latest-frame-comparison-1',arm_labels=LABELS,qualification_trials=0,
        source_reference=str(REFERENCE),reference_manifest_sha256=digest(REFERENCE/'execution_manifest.json'),
        grounder_checkpoint_hashes=cfg['grounder_checkpoint']['file_hashes'],
        interpretation='Accepted legacy C++ detector vs full Agent S3 vs Agent S3 latest-frame; not Full Chronotope',
        integration_limitations=[
            'Legacy native detector uses evaluator-provided target color, unlike the new pixel-only updater.',
            'Only the Agent S3 pair matches prompts, history, grounding and generation settings.',
            'Native algorithm and scorer unchanged; native is not full GPU-history Chronotope.',
            'Benchmark has a scheduled 20ms action route, not physical application ACK timestamps.',
            'Frames advance on wall-clock task time, but simulator catches up publications after blocking model calls.',
            'CUDA NCC kernels match the existing updater; identity check and Python orchestration add measured CPU work.',
            'Code is deferred pending context repair and native typing support; no Code trials in this plan.'])
    for name in ('full_agent_components.json','service_launch_reference.json'):
        source=REFERENCE/'configs'/name if name.startswith('full_') else OTHER/'configs'/name
        shutil.copy2(source,out/'configs'/name)
    write(out/'configs/task_instructions.json',{t:task_instruction(t) for t in TASKS})
    full=read(REFERENCE/'configs/notification_agents3_full.json')
    native=read(REFERENCE/'configs/notification_legacy_cpp_detector.json')
    original_prompt=read(REFERENCE/'configs/notification_compact_prompt.json')
    cases={};cells=[]
    for task in TASKS:
        scale=1 if task=='button' else 2
        host=dict(full,image_scale=scale)
        update=dict(host,kind='agents3_latest_frame',updater_backend='cuda_ncc',
                    updater_library=str(lib),template_radius_px=12,min_match_score=.8,
                    identity_max_mae=2.0,safe_policy='abstain')
        nc=dict(native,prompt_version=TASK_VERSIONS[task],image_scale=scale)
        for arm,c in zip(ARMS,(nc,host,update)): write(out/'configs'/f'{task}_{arm}.json',c)
        if task=='notification': prompt,ref=original_prompt,cfg['native_task_prompt_references'][task]
        else:
            base=dict(original_prompt,task=task,prompt_version=TASK_VERSIONS[task],
                      planner_system=SYSTEM_PROMPTS[TASK_VERSIONS[task]])
            # Tokenizer dependencies live in the existing serving environment;
            # this subprocess renders text on CPU and never loads model weights.
            command=[str(ROOT.parent/'qwen-serve/.venv/bin/python'),'-B','-c',
                'import sys,json; sys.path[:0]=[sys.argv[1],sys.argv[1]+"/src"]; '
                'from prepare_ledger_native import render_contract; '
                'print(json.dumps(render_contract(json.load(sys.stdin),sys.argv[2])))',
                str(out/'code'),str(ROOT/'runs/model_cache/qwen35_0p8b_legacy_source')]
            rendered=subprocess.run(command,input=json.dumps(base),text=True,capture_output=True,check=True)
            prompt,ref=json.loads(rendered.stdout)
        cfg['native_task_prompts'][task]=prompt;cfg['native_task_prompt_references'][task]=ref
        write(out/'configs'/f'{task}_compact_prompt.json',prompt)
        for setting in SETTINGS:
            source=(REFERENCE/'cases'/f'notification_w{setting*10**9}.jsonl' if task=='notification'
                    else OTHER/'cases'/f'{task}_change{setting}s.jsonl')
            cell=f'{task}_{setting}s'; dest=out/'cases'/f'{cell}.jsonl'
            shutil.copy2(source,dest)
            cases[cell]=[json.loads(line) for line in dest.read_text().splitlines()]
            cells.append(dict(task=task,setting_s=setting,cell_id=cell,manifest=str(dest.relative_to(out)),
                              source_cases=str(source),source_sha256=digest(source),trials_per_arm=100,
                              task_deadline_s=setting if task=='notification' else 32))
    jobs=[]
    # Round-robin all cells per seed; rotate controller order to avoid time/order confounding.
    for index in range(100):
        for k,cell in enumerate(cells):
            case=cases[cell['cell_id']][index];order=(index+k)%3
            for arm in ARMS[order:]+ARMS[:order]:
                jobs.append(dict(task=cell['task'],setting_s=cell['setting_s'],cell_id=cell['cell_id'],
                    task_deadline_s=cell['task_deadline_s'],manifest=cell['manifest'],index=index,
                    seed=case['seed'],episode_id=case['episode_id'],arm=arm,mode='live',phase='evaluation',
                    id=f"{arm}_{cell['cell_id']}_{index:03d}_live"))
    plan=dict(schema='latest-frame-live-plan-1',model=MODEL,revision=REVISION,cells=cells,jobs=jobs,
        planned_trials=TOTAL,qualification_trials=0,successor_jobs=[],automatic_resume=False,
        modes=['live'],settings_s=SETTINGS,trials_per_cell=100,
        action_policies={'agents3_full':'Keep','agents3_latest_frame':'Update arguments or abstain; never Restart',
                         'legacy_cpp_detector':'historical detector correction'},
        deferred={'code':'Context-limit repair pending; native typing unsupported'},
        authorization='User requested additional latest-frame controller alongside existing Qwen3.5-0.8B legacy C++ vs Agent S3; no questions; obtain measured results',
        previous_failed_runs_preserved=True)
    write(out/'plan.json',plan)
    cfg['hashes']={str(p.relative_to(out)):digest(p) for folder in ('code','configs','cases','build')
                   for p in sorted((out/folder).rglob('*')) if p.is_file()}
    cfg['hashes']['plan.json']=digest(out/'plan.json')
    write(out/'execution_manifest.json',cfg)
    write(out/'status.json',dict(state='prepared',completed_trials=0,infrastructure_errors=0,
                                attempted_trials=0,planned_trials=TOTAL,pid=None,updated_at=now()))
    validate(out)
    summarize(out,[])


def validate(out):
    setup(out)
    from core5_ledger_plan import make_trial_spec
    from agents3_full import source_fingerprints
    cfg=read(out/'execution_manifest.json');plan=read(out/'plan.json')
    verify(cfg['hashes'],out)
    assert (plan['model'],plan['revision'],plan['planned_trials'])==(MODEL,REVISION,TOTAL)
    assert plan['qualification_trials']==0 and not plan['successor_jobs'] and not plan['automatic_resume']
    assert len(plan['jobs'])==TOTAL and len({j['id'] for j in plan['jobs']})==TOTAL
    assert {j['task'] for j in plan['jobs']}==set(TASKS)
    assert {j['phase'] for j in plan['jobs']}=={'evaluation'}
    assert {j['mode'] for j in plan['jobs']}=={'live'}
    assert {j['arm'] for j in plan['jobs']}==set(ARMS)
    cases={};groups=defaultdict(set)
    for cell in plan['cells']:
        path=out/cell['manifest'];assert digest(path)==cell['source_sha256']
        own=[json.loads(line) for line in path.read_text().splitlines()];assert len(own)==100
        cases[cell['manifest']]=own
    for j in plan['jobs']:
        c=cases[j['manifest']][j['index']]
        assert (c['index'],c['seed'],c['episode_id'])==(j['index'],j['seed'],j['episode_id'])
        spec=make_trial_spec(c,read(out/'configs'/f"{j['task']}_{j['arm']}.json"),'live')
        scenario=spec.raw['scenario']
        assert scenario['response_window_ns']==j['task_deadline_s']*10**9
        if j['task']!='notification':
            field='change_delay_ns' if j['task']=='button' else 'reversal_delay_ns'
            assert scenario[field]==j['setting_s']*10**9
        groups[j['task'],j['setting_s'],j['arm']].add(j['seed'])
    assert len(groups)==45 and all(len(s)==100 for s in groups.values())
    for task in TASKS:
        for setting in SETTINGS:
            assert groups[task,setting,ARMS[0]]==groups[task,setting,ARMS[1]]==groups[task,setting,ARMS[2]]
    original=cfg['full_agent_source_verification']['source_file_sha256']
    for path,h in source_fingerprints().items():
        suffix=path[path.index('/gui_agents/'):] if '/gui_agents/' in path else '/agents3_full.py'
        assert [v for p,v in original.items() if p.endswith(suffix)]==[h]
    write(out/'validation.json',dict(state='passed',at=now(),planned_trials=TOTAL,
        task_families=list(TASKS),qualification_trials=0,matched_seed_sets=True,model_inference_calls=0,
        plan_sha256=digest(out/'plan.json'),scorer_unchanged=True))
    return cases


def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def summarize(out,records):
    import numpy as np
    groups=defaultdict(list)
    for r in records: groups[r['job']['task'],r['job']['setting_s'],r['job']['arm']].append(r)
    rows=[];flat=[];pairs=[]
    for task in TASKS:
        for setting in SETTINGS:
            for arm in ARMS:
                allr=groups[task,setting,arm];own=[r for r in allr if r['status']=='scored']
                n=len(own);k=sum(r['success'] for r in own);lo,hi=wilson(k,n)
                def pc(key,p):
                    v=[r['metrics'].get(key) for r in own if r['metrics'].get(key) is not None]
                    return float(np.percentile(v,p)) if v else None
                row=dict(task=TASKS[task],setting_s=setting,task_deadline_s=setting if task=='notification' else 32,
                    controller=LABELS[arm],arm=arm,mode='live',successes=k,trials=n,planned=100,
                    infrastructure_errors=len(allr)-n,success_rate=k/n if n else None,ci95_low=lo,ci95_high=hi,
                    receipt_trials=sum(r['metrics']['receipt_ns'] is not None for r in own),
                    e_to_receipt_p50_ms=pc('e_to_receipt_ms',50),e_to_receipt_p95_ms=pc('e_to_receipt_ms',95),
                    observed_handler_e_to_a_p50_ms=pc('observed_handler_e_to_a_ms',50),
                    observed_handler_e_to_a_p95_ms=pc('observed_handler_e_to_a_ms',95),
                    updater_p50_ms=pc('updater_ms',50),updater_p95_ms=pc('updater_ms',95),
                    model_calls=sum(r['metrics']['model_calls'] for r in own),
                    input_tokens=sum(r['metrics']['input_tokens'] for r in own),
                    output_tokens=sum(r['metrics']['output_tokens'] for r in own),
                    abstentions=sum(bool(r['metrics']['abstention_reason']) for r in own),
                    operation_changes_after_B=sum(r['metrics']['operation_changed_after_B'] is True for r in own),
                    coverage='complete' if n==100 else 'partial' if n else 'unmeasured',
                    model=MODEL,revision=REVISION,planner_precision='FP16',kv_cache='FP16',
                    runtime='TensorRT-Edge-LLM 0.11.1' if arm==ARMS[0] else 'vLLM',
                    grounder='task-color detector' if arm==ARMS[0] else 'UI-TARS-1.5-7B BF16',
                    seed_set=json.dumps(sorted(r['job']['seed'] for r in own)),source=str(out))
                rows.append(row)
            a={r['job']['seed']:r for r in groups[task,setting,ARMS[1]] if r['status']=='scored'}
            b={r['job']['seed']:r for r in groups[task,setting,ARMS[2]] if r['status']=='scored'}
            shared=sorted(a.keys()&b.keys())
            matched=[s for s in shared if a[s]['metrics']['initial_observation_sha256']==b[s]['metrics']['initial_observation_sha256']
                     and a[s]['metrics']['first_request_sha256']==b[s]['metrics']['first_request_sha256']
                     and a[s]['metrics']['first_request_sha256']]
            diffs=np.array([b[s]['success']-a[s]['success'] for s in matched],dtype=float)
            if len(diffs):
                boot=np.random.default_rng(20261108).choice(diffs,(2000,len(diffs)),replace=True).mean(axis=1)
                ci=np.quantile(boot,[.025,.975]);delta=float(diffs.mean())
            else:ci=[None,None];delta=None
            pairs.append(dict(task=TASKS[task],setting_s=setting,completed_seed_pairs=len(shared),
                matched_initial_input_pairs=len(matched),input_mismatches=len(shared)-len(matched),
                missing_keep_seeds=json.dumps(sorted(b.keys()-a.keys())),
                missing_updater_seeds=json.dumps(sorted(a.keys()-b.keys())),
                paired_success_delta=delta,bootstrap95_low=ci[0],bootstrap95_high=ci[1]))
    for r in records:
        j=r['job'];flat.append(dict(trial_id=j['id'],task=TASKS[j['task']],setting_s=j['setting_s'],
            task_deadline_s=j['task_deadline_s'],controller=LABELS[j['arm']],arm=j['arm'],seed=j['seed'],mode='live',
            status=r['status'],success=r.get('success'),error=r.get('error'),**r['metrics'],
            trace=str(out/r['trace']) if r.get('trace') else '',trace_sha256=r.get('trace_sha256'),
            request_log=str(out/'calls'/f"{j['id']}.jsonl"),model=MODEL,revision=REVISION,
            planner_precision='FP16',grounder_precision='BF16' if j['arm']!=ARMS[0] else '',
            execution_manifest=str(out/'execution_manifest.json')))
    csv_write(out/'summary.csv',rows);csv_write(out/'per_trial.csv',flat);csv_write(out/'paired.csv',pairs)
    lines=['# Qwen3.5-0.8B additional-controller comparison','',f'Updated {now()}. Fresh Live trials only; counts are scored successes/trials. Infrastructure errors are separate.',
        '', '| Task | Setting | Legacy C++ + detector | Agent S3 Keep | Agent S3 + latest-frame |',
        '|---|---:|---:|---:|---:|']
    for task in TASKS:
        for setting in SETTINGS:
            rr=[r for r in rows if r['task']==TASKS[task] and r['setting_s']==setting]
            lines.append(f'| {TASKS[task]} | {setting} s | '+' | '.join(f"{r['successes']}/{r['trials']}" if r['trials'] else '—' for r in rr)+' |')
    lines+=['','Each cell plans 100 cases. Notification settings are lifetimes; Button/Trading settings are time to state change, with a separate 32 s deadline.',
        '', 'Code remains deferred for context repair and native typing. Full Chronotope is unavailable in this accepted legacy comparison.',
        '', 'The Agent S3 pair preserves Worker/history/reflection, UI-TARS, prompts and generation settings; paired.csv reports initial-input checks and paired bootstrap intervals. summary.csv has Wilson 95% intervals, latency percentiles and resource totals.',
        '', 'Latency uses the benchmark receipt boundary (scheduled fixed 20 ms route). observed_handler timestamps are measured harness handling times, not physical application acknowledgements; no-action trials have no receipt latency.',
        '', 'The native arm retains its task-color detector and compact prompt. It is not an oracle-free or identical-prompt comparison with Agent S3. The new updater sees pixels only, reuses unchanged CUDA NCC kernels, adds a conservative RGB identity check, and abstains on uncertain matches.',
        '', 'An argument updater cannot restore an expired Notification or make a reversed Trading order appropriate. These post-hoc operation-change cases are counted separately; task state is used only in analysis/scoring, never by the updater.',
        '', 'No substantive comparison conclusion should be drawn from small partial counts. All original paper and historical measurements remain separate.']
    tmp=out/'comparison.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(out/'comparison.md')


def metrics(trace,calls,task):
    # This function is analysis only; neither it nor a manifest is passed to the updater.
    h=trace.header();origin=h['harness_origin_ns'];ev=origin+h['spec']['scenario']['event_time_ns']
    receipt=trace.first('receipt') or {};delivery=trace.first('delivery') or {};commit=trace.first('commit') or {}
    sup=trace.first('action_suppressed') or {};upd=trace.first('coordinate_correction') or sup
    decision=trace.committing_decision() or trace.first('decision_begin') or {}
    obs={o['observation_id']:o for o in trace.all('observation')}
    first=trace.first('decision_begin') or {};initial=obs.get(first.get('observation_id'),{})
    # decision_begin uses observation_id_b in older frozen traces.
    if not initial:initial=obs.get(first.get('observation_id_b'),{})
    used=obs.get(decision.get('observation_id',decision.get('observation_id_b')), {})
    end=trace.first('trial_end') or {};a=receipt.get('receipt_time_ns');observed=receipt.get('receipt_observed_time_ns')
    terminal=receipt.get('task_time_at_receipt_ns',end.get('task_time_ns'))
    threshold=h['spec']['scenario']['event_time_ns']+h['spec']['scenario'].get('reversal_delay_ns',h['spec']['scenario']['response_window_ns'])
    changed=(used.get('source_time_ns',threshold)>=0 and used.get('source_time_ns',threshold)<threshold<=terminal
             if terminal is not None and task in ('notification','trading') else False)
    inputs=outputs=0
    for c in calls:
        usage=c.get('response',{}).get('usage',{}) if isinstance(c.get('response'),dict) else {}
        inputs+=usage.get('prompt_tokens',c.get('input_tokens',0)) or 0
        outputs+=usage.get('completion_tokens',c.get('output_tokens',0)) or 0
    update_ns=upd.get('binder_end_ns',0)-upd.get('binder_start_ns',0) if upd else None
    return dict(harness_origin_ns=origin,event_ns=ev,first_decision_ns=first.get('begin_time_ns'),
        proposal_ns=(trace.first('action_proposed') or {}).get('proposal_time_ns'),
        commitment_ns=commit.get('commitment_time_ns'),submission_ns=delivery.get('submit_time_ns'),
        submission_observed_ns=delivery.get('submit_observed_time_ns'),receipt_ns=a,
        receipt_observed_ns=observed,physical_application_ack_ns=None,
        receipt_timestamp_kind='scheduled benchmark receipt; observed handler timestamp reported separately',
        e_to_receipt_ms=(a-ev)/1e6 if a else None,observed_handler_e_to_a_ms=(observed-ev)/1e6 if observed else None,
        updater_ms=update_ns/1e6 if update_ns is not None else None,
        updater_internal_ms=upd.get('track_ns',0)/1e6 if upd else None,
        model_calls=len(calls),input_tokens=inputs,output_tokens=outputs,
        planner_calls=sum(c.get('role',c.get('request_kind'))=='planner' for c in calls),
        grounder_calls=sum(c.get('role')=='grounder' for c in calls),
        first_request_sha256=calls[0].get('request_sha256') if calls else None,
        initial_observation_sha256=initial.get('payload_hash'),
        abstention_reason=sup.get('reason'),operation_changed_after_B=bool(changed),
        operation_changed_by_updater=False,updater_observation_id=upd.get('observation_id'),
        terminal_reason=end.get('reason'),proposed_action=json.dumps((trace.first('action_proposed') or {}).get('action')),
        committed_action=json.dumps(commit.get('action')))


def run(out):
    if read(out/'status.json')['state']!='prepared' or any((out/'records').iterdir()) or any((out/'trials').iterdir()):
        raise RuntimeError('Fresh prepared study required; no automatic resume')
    # One shared GPU study at a time. Services are reused with verified identity.
    locks=[];inodes=set()
    for p in (ROOT/'runs/legacy_d1_gpu.lock',ROOT/'runs/hosted_api_deepseek_nothinking_20261105/source/runs/legacy_d1_gpu.lock'):
        identity=(p.stat().st_dev,p.stat().st_ino)
        if identity in inodes: continue
        f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        inodes.add(identity)
    cases=validate(out);cfg=read(out/'execution_manifest.json');plan=read(out/'plan.json')
    from run_claim1_agents3_remaining3 import check_services
    check_services(out);verify(cfg['engine_hashes'])
    assert digest(cfg['plugin']['path'])==cfg['plugin']['sha256']
    from agents3_full import FullAgentS3Adapter
    from core5_ledger_plan import make_trial_spec
    from run_core5_detector_full_agents3 import make_host
    from audit_legacy_detector_full_agents3 import audit_trial_requests
    from chronotope.controller import CallBudget
    from chronotope.latest_frame import wrap_from_config
    from chronotope.runner import run_trial,PayloadStore
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    from chronotope_core.controller import build as build_native
    from chronotope.distributional.manifest import EpisodeRow,build_manifest
    from chronotope.distributional.episode import verify_trace_evidence
    for key in ('CHRONOTOPE_AGENT_S3_POLICY','CHRONOTOPE_AGENT_S3_PROMPT','CHRONOTOPE_AGENT_S3_AUDIT'):
        os.environ.pop(key,None)
    os.environ['CHRONOTOPE_BIN']=str(out/'code/chronotope/build/chronotope')
    os.environ['CHRONOTOPE_LIB']=str(out/'code/chronotope/build/libchronotope.so')
    os.environ['EDGELLM_PLUGIN_PATH']=cfg['plugin']['path']
    records=[];current=None;started=now()
    def publish(state,error=None):
        n=sum(r['status']=='scored' for r in records)
        write(out/'status.json',dict(state=state,updated_at=now(),started_at=started,
            pid=os.getpid() if state=='running' else None,completed_trials=n,
            infrastructure_errors=len(records)-n,attempted_trials=len(records),planned_trials=TOTAL,
            current_job=current,qualification_trials=0,automatic_resume=False,error=error))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    publish('running')
    try:
        for current in plan['jobs']:
            publish('running');j=current;ident=j['id'];task=j['task'];arm=j['arm']
            case=cases[j['manifest']][j['index']]
            config=read(out/'configs'/f'{task}_{arm}.json')
            spec=make_trial_spec(case,config,'live');scenario=spec.manifest()
            distributional=None
            if case.get('distributional_row'):
                distributional=EpisodeRow.from_dict(case['distributional_row']);scenario=build_manifest(distributional)
            path=out/'trials'/f'{ident}.jsonl';log=out/'calls'/f'{ident}.jsonl'
            assert not path.exists() and not log.exists()
            controller=adapter=None;start=time.monotonic();error=None;request_audit=[];evaluation=None;trace=None
            try:
                if arm==ARMS[0]:
                    os.environ['CHRONOTOPE_PLANNER_PROMPT']=str(out/'configs'/f'{task}_compact_prompt.json')
                    os.environ['CHRONOTOPE_PLANNER_AUDIT']=str(log)
                    controller=build_native(spec,budget=CallBudget(100))
                else:
                    adapter=FullAgentS3Adapter(cfg['agent_s3_components'],log)
                    base=make_host(spec,task,adapter,{'code_action':'not_needed'})
                    controller=wrap_from_config(base,config,scenario.canvas_w_px,scenario.canvas_h_px)
                run_trial(spec,path,PayloadStore(out/'payloads'),controller=controller,manifest=scenario)
                trace=Trace.load(path)
                if trace.all('decision_failed') or (adapter and adapter.errors):
                    raise RuntimeError('Model/runtime error; retained unscored')
                evaluated=evaluate_trace(trace)
                if not evaluated.valid_trial or evaluated.contract_violations:
                    raise RuntimeError('Trace contract failure: '+str(evaluated.contract_violations))
                evaluation=evaluated.to_dict()
                acfg=dict(cfg,native_prompt_contract=cfg['native_task_prompts'][task],
                          native_prompt_reference=cfg['native_task_prompt_references'][task])
                request_audit=audit_trial_requests(out,ident,ARMS[0] if arm==ARMS[0] else ARMS[1],trace,scenario,acfg)
                if distributional:
                    verify_trace_evidence(distributional,scenario,trace,case['distributional_manifest_hash'])
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                (out/'records'/f'{ident}.error.txt').write_text(traceback.format_exc())
            finally:
                try:
                    close=getattr(controller,'close',None)
                    if close:close()
                finally:
                    if adapter:adapter.close()
            if trace is None and path.exists():trace=Trace.load(path)
            calls=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
            # Construction errors have no trace; use a fixed metric schema with null timestamps.
            if trace is not None:met=metrics(trace,calls,task)
            else:
                met=empty_metrics();met.update(model_calls=len(calls),terminal_reason='construction_error')
            record=dict(job=j,status='infrastructure_error' if error else 'scored',error=error,
                success=None if error else int(bool(evaluation['valid_at_receipt'])),evaluation=evaluation,
                metrics=met,request_audit=request_audit,trace=str(path.relative_to(out)) if path.exists() else None,
                trace_sha256=digest(path) if path.exists() else None,completed_at=now(),wall_seconds=time.monotonic()-start)
            write(out/'records'/f'{ident}.json',record);records.append(record)
            print(f"{len(records)}/{TOTAL} {ident} {record['status']} success={record['success']} {error or ''}",flush=True)
            summarize(out,records);publish('running')
        current=None
        audit(out,records)
        publish('complete' if all(r['status']=='scored' for r in records) else 'complete_with_infrastructure_errors')
    except BaseException as exc:
        summarize(out,records);publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',f'{type(exc).__name__}: {exc}')
        raise


def empty_metrics():
    keys='harness_origin_ns event_ns first_decision_ns proposal_ns commitment_ns submission_ns submission_observed_ns receipt_ns receipt_observed_ns physical_application_ack_ns receipt_timestamp_kind e_to_receipt_ms observed_handler_e_to_a_ms updater_ms updater_internal_ms model_calls input_tokens output_tokens planner_calls grounder_calls first_request_sha256 initial_observation_sha256 abstention_reason operation_changed_after_B operation_changed_by_updater updater_observation_id terminal_reason proposed_action committed_action'.split()
    d=dict.fromkeys(keys)
    for key in ('model_calls','input_tokens','output_tokens','planner_calls','grounder_calls'):d[key]=0
    return d


def audit(out,records=None):
    setup(out)
    from chronotope.trace import Trace
    from chronotope.evaluate import evaluate_trace
    validate(out)
    records=records if records is not None else [read(p) for p in sorted((out/'records').glob('*.json'))]
    plan={j['id']:j for j in read(out/'plan.json')['jobs']}
    for r in records:
        assert r['job']==plan[r['job']['id']]
        if r['trace']:
            assert digest(out/r['trace'])==r['trace_sha256']
            if r['status']=='scored':assert evaluate_trace(Trace.load(out/r['trace'])).to_dict()==r['evaluation']
    summarize(out,records)
    result=dict(state='passed',audited_records=len(records),scored=sum(r['status']=='scored' for r in records),
                complete_plan=len(records)==TOTAL,at=now(),plan_sha256=digest(out/'plan.json'))
    write(out/'evidence_audit.json',result);return result


def launch(out):
    validate(out)
    if read(out/'status.json')['state']!='prepared':raise RuntimeError('Already launched or attempted')
    marker=out.parent/'launch.json'
    if marker.exists():raise FileExistsError('No automatic relaunch')
    cfg=read(out/'execution_manifest.json')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH=':'.join([str(Path(cfg['plugin']['path']).parent),
            (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),'/usr/local/cuda/targets/sbsa-linux/lib']))
    env.pop('PYTHONPATH',None)
    cmd=[PYTHON,'-B',str(out/'code/run_latest_frame_qwen08.py'),'run','--out',str(out)]
    with (out.parent/'run.log').open('xb') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(marker,dict(pid=child.pid,command=cmd,at=now(),plan_sha256=digest(out/'plan.json'),automatic_resume=False))
    print(json.dumps(read(marker)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','validate','launch','run','audit','report'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve()
    if args.action=='report':
        summarize(out,[read(p) for p in sorted((out/'records').glob('*.json'))])
    else:
        try:globals()[args.action](out)
        except BaseException as exc:
            if args.action=='run' and read(out/'status.json')['state']=='prepared':
                state=read(out/'status.json')
                state.update(state='setup_failed',updated_at=now(),pid=None,error=f'{type(exc).__name__}: {exc}')
                write(out/'status.json',state)
            raise
