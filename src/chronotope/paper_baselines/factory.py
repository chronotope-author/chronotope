"""Explicit construction and static scope checks. Never starts model services."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import ChatModel, ModelIdentity, SetupError, TASK_NAMES
from .upstream import LOCK_PATH, locked_file

BASELINES = ('aoi', 'mobiledreamer', 'aapt')
LABELS = {'aoi': 'AOI visual (buffered processing)',
          'mobiledreamer': 'MobileDreamer released one-step selection (adapted)',
          'aapt': 'AAPT (paper-derived harness adaptation)'}
SCENARIOS = {'notification': ('revealed_target', 'notification'),
    'drift': ('moving_target', 'balloon'), 'code': ('rotating_code', 'notification'),
    'button': ('reflow_page', 'browser'), 'trading': ('live_trading', 'chart')}


def register():
    from ..spec import _CONTROLLER_SCHEMAS
    _CONTROLLER_SCHEMAS['paper_baseline'] = {
        'kind': (str, ('paper_baseline',)), 'baseline': (str, BASELINES),
        'task': (str, tuple(TASK_NAMES)), 'instruction': (str, None),
        'config_path': (str, None), 'config_sha256': (str, None)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(controller):
    path = Path(controller['config_path'])
    if digest(path) != controller['config_sha256']:
        raise SetupError('Baseline config changed since preparation')
    config = json.loads(path.read_text())
    if config.get('schema') != 'paper-baseline-config-1' or config.get('baseline') != controller['baseline']:
        raise SetupError('Baseline configuration mismatch')
    return config


def check_scope(raw):
    c, s = raw['controller'], raw['scenario']
    if c.get('kind') != 'paper_baseline' or c.get('task') not in SCENARIOS:
        raise SetupError('Expected a core-five paper baseline')
    if (s['scenario_id'], s['skin']) != SCENARIOS[c['task']]:
        raise SetupError('Task/scenario mismatch; jump, select and non-core tasks are excluded')
    if c['task'] == 'code' and s.get('kind_change', False):
        raise SetupError('Only the established authenticator task is supported')
    if raw['timestamp_source'] != 'monotonic_ns':
        raise SetupError('Real model trials must measure wall-clock latency')
    if raw['delivery']['route'] not in ('simulated_fixed_delay', 'simulated_seeded_jitter', 'simulated_empirical'):
        raise SetupError('This adapter supports software harness routes only')
    if raw.get('observation', {}).get('source', 'environment') != 'environment':
        raise SetupError('This runner requires rendered core-five screenshots')
    if not c['instruction'].strip():
        raise SetupError('Missing public task instruction')
    config = read_config(c)
    # Preserve the standing large-model sub-second restriction. Never infer total
    # parameters from an active-parameter marketing suffix.
    total = config.get('planner_total_parameters')
    if type(total) is not int or total <= 0:
        raise SetupError('Explicit planner_total_parameters is required')
    if c['task'] == 'notification' and s['response_window_ns'] < 1_000_000_000 and total > 4_000_000_000:
        raise SetupError('Sub-second Notification windows are reserved for models <=4B total')
    return config


def make_model(config):
    return ChatModel(ModelIdentity(**config['identity']), config['base_url'],
                     timeout_s=config.get('timeout_s', 120), api_key_env=config.get('api_key_env'),
                     extra_body=config.get('extra_body'))


def check_config(config):
    baseline = config['baseline']
    if baseline not in BASELINES:
        raise SetupError('Unknown baseline')
    make_model(config['model'])  # constructor performs no network requests
    if baseline in ('aoi', 'mobiledreamer'):
        lock = json.loads(LOCK_PATH.read_text())[baseline]
        for relative in lock['files']:
            locked_file(config['source_root'], baseline, relative)
    if baseline == 'mobiledreamer':
        world = config.get('world_model')
        if not world:
            raise SetupError('MobileDreamer requires its learned world model; no planner fallback')
        make_model(world)
        identity = world['identity']
        if identity['model'] != 'UITron-hub/MobileDreamer' or identity['revision'] != '7d229e206cb4dd7aaff4c801174d2292fd545d57':
            raise SetupError('Expected the pinned official MobileDreamer world-model checkpoint')
        if world.get('checkpoint_subfolder') != 'merge_model':
            raise SetupError('Use the released merged world-model weights in merge_model')
    if baseline == 'aapt':
        params = config.get('parameters', {})
        budget = params.get('planner_p95_ms')
        if type(budget) is not int or budget <= 0 or not params.get('planner_latency_evidence'):
            raise SetupError('AAPT needs a planner latency budget and its existing evidence reference')


def build(spec):
    config = check_scope(spec.raw)
    check_config(config)
    c, s = spec.raw['controller'], spec.raw['scenario']
    kwargs = dict(task=c['task'], instruction=c['instruction'], width=s['canvas_w_px'], height=s['canvas_h_px'],
                  config=config.get('parameters', {}), model=make_model(config['model']))
    if c['baseline'] == 'aapt':
        from .aapt import AAPTController
        return AAPTController(**kwargs)
    if c['baseline'] == 'aoi':
        from .aoi import AOIController
        return AOIController(source_root=config['source_root'], **kwargs)
    from .mobiledreamer import MobileDreamerController
    return MobileDreamerController(source_root=config['source_root'],
                                  world_model=make_model(config['world_model']), **kwargs)
