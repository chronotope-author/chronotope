"""Audit completed records; produce paragraph inputs/figure without screening trials.
Stops on corrupt evidence or native runtime errors, never on low task success.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
from statistics import median
import time
from legacy_d1_w8 import now,write
from prepare_deadline_envelope import observed_threshold,WINDOWS

MODEL='RadixArk/Qwen3.8-27B-NVFP4'
MODELS={'radixark_27b':MODEL,'qwen35_0p8b':'Qwen/Qwen3.5-0.8B'}


def audit_record(out,r):
    assert r['job']['phase']=='evaluation' and r['job']['model'] in MODELS
    model=MODELS[r['job']['model']]
    raw=(out/r['trace']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==r['trace_sha256'], 'Trace hash changed'
    rows=[json.loads(line) for line in raw.splitlines()]
    header=next(x for x in rows if x['record']=='trial_header')
    assert header['spec']['controller']['model_id']==model, 'Model mismatch'
    for event in rows:
        if event['record']=='decision_failed':raise RuntimeError('Infrastructure/controller failure in trace')
        if event['record']=='decision_invocation':
            if event.get('planner')=='tensorrt_edgellm' and event.get('outcome')=='unparseable':
                # Legacy wrapper uses this label when TensorRT answer_ok=False;
                # ordinary non-click/malformed model answers have answer_ok=True.
                raise RuntimeError('Native TensorRT request failed, not a task competence failure')
    assert not r['outcome']['invalid_trial'] and not r['outcome']['contract_violations']
    if r['job']['mode']=='replay':assert r['pair_valid'] is True


def summarize(out,records,status):
    model=status.get('model',MODEL)
    execution=json.loads((out/'execution_manifest.json').read_text()) if (out/'execution_manifest.json').exists() else {}
    labels=execution.get('arm_labels',{})
    groups=defaultdict(list)
    for r in records:groups[r['job']['arm'],r['job']['window_s'],r['job']['mode']].append(r['outcome'])
    rows=[]
    for (arm,w,mode),os_ in sorted(groups.items()):
        vals=[o['L_EA']/1e6 for o in os_ if o['L_EA'] is not None]
        rows.append({'Controller':labels.get(arm,arm),'Model':model,'Window (s)':w,'Condition':mode,'n':len(os_),
                     'Successes':sum(o['Y'] for o in os_),'E-to-A p50 (ms)':median(vals) if vals else '',
                     'Receipts':sum(o['R'] for o in os_)})
    dest=out/'results';dest.mkdir(exist_ok=True)
    if rows:
        with (dest/'table.csv.tmp').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        (dest/'table.csv.tmp').replace(dest/'table.csv')
    values={'updated_at':now(),'state':status['state'],'audited_records':len(records),'model':model,
            'success_criterion':'At least 90/100 Live at this and all wider tested windows; descriptive only',
            'qualification_trials':0,'thresholds':{},'paragraph_ready':False,
            'model_scope':'This single-model pair does not alone fill a larger-versus-smaller model paragraph',
            'attribution':('Matched policy and initial model request; generated outputs and later histories may differ across inference runtimes. Separate inference and control timing before overhead-only attribution' if execution.get('request_contract') else 'Same checkpoint, different system runtimes/prompts/history; separate inference and control before overhead-only attribution')}
    for arm in ['legacy_cpp','agents3_stripped']:
        points=[{'window_s':w,'n':len(groups[arm,w,'live']),'successes':sum(o['Y'] for o in groups[arm,w,'live'])} for w in WINDOWS]
        if all(p['n']==100 for p in points):values['thresholds'][arm]=observed_threshold(points)
    if len(values['thresholds'])==2:
        w=values['thresholds']['legacy_cpp']['window_s']
        if w is not None:
            values['duration_s']=w
            values['x_legacy_out_of_100']=sum(o['Y'] for o in groups['legacy_cpp',w,'live'])
            values['y_agents3_out_of_100']=sum(o['Y'] for o in groups['agents3_stripped',w,'live'])
        values['receipt_time_at_16s_ms']={}
        for arm in ['legacy_cpp','agents3_stripped']:
            os_=groups[arm,16.,'live'];ts=[o['L_EA']/1e6 for o in os_ if o['L_EA'] is not None]
            values['receipt_time_at_16s_ms'][arm]={'n':len(ts),'p50':median(ts) if ts else None,'missing':100-len(ts)}
        values['same_model_pair_complete']=True
    write(dest/'paragraph_values.json',values)
    return groups


def plot(out,groups):
    execution=json.loads((out/'execution_manifest.json').read_text());model=execution['model']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for arm,label in [('legacy_cpp','Legacy C++ / TensorRT-Edge-LLM'),('agents3_stripped','Stripped Agent S3 / vLLM')]:
        label=execution.get('arm_labels',{}).get(arm,label)
        ys=[sum(o['Y'] for o in groups[arm,w,'live']) for w in WINDOWS]
        ax.plot(WINDOWS,ys,'o-',label=label)
    ax.set_xscale('log',base=2);ax.set_xticks(WINDOWS,[f'{w:g}' for w in WINDOWS])
    ax.set(xlabel='Response window (s)',ylabel='Live successes / 100',ylim=(-2,102),title=model)
    ax.axhline(90,color='.6',linestyle=':',linewidth=.8);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/'results'/f'deadline_envelope.{suffix}',dpi=180)
    plt.close(fig)


def main(out):
    seen={}
    while True:
        status=json.loads((out/'status.json').read_text())
        try:
            for path in sorted((out/'records').glob('*.json')):
                if path.name in seen:continue
                r=json.loads(path.read_text());audit_record(out,r);seen[path.name]=r
            groups=summarize(out,list(seen.values()),status)
            write(out/'audit_status.json',{'state':'auditing' if status['state']=='running' else 'finished','updated_at':now(),'records':len(seen),'infrastructure_errors':0,'task_failures_retained':True})
        except Exception as exc:
            write(out/'audit_status.json',{'state':'failed','updated_at':now(),'records':len(seen),'error':f'{type(exc).__name__}: {exc}'})
            if status['state']=='running' and status.get('pid'):
                proc=Path(f"/proc/{status['pid']}/cmdline")
                if proc.exists() and str(out).encode() in proc.read_bytes():os.kill(status['pid'],signal.SIGTERM)
            raise
        if status['state']!='running':
            if status['state']=='complete':plot(out,groups)
            return
        time.sleep(5)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.out.resolve())
