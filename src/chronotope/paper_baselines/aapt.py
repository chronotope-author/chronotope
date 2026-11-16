"""Paper-derived AAPT, arXiv:2607.28399v1 §§3, 9, 10, 16.3.

Flat bounded trees; same frozen model for planning and 64-token routing;
pixel-change/periodic-probe gate; strict action, eligibility, prior-confidence,
and observation-age checks. All preparation consumes the trial clock.
"""
from __future__ import annotations

import math
import numpy as np

from ..controller import Restart
from .common import (InvalidOutput, PixelController, SetupError, parse_command,
                     pil_image, strict_json, user_message)


def tree_schema(max_nodes=10, max_depth=2, max_children=3, *, required_cover_ms=1, max_cover_ms=64000):
    node = {'id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
        'depth': {'type': 'integer', 'minimum': 0, 'maximum': max_depth},
        'guard': {'type': 'string', 'minLength': 1, 'maxLength': 96},
        'action': {'type': 'string', 'minLength': 1, 'maxLength': 96,
                   'pattern': r'^(WAIT|wait\(\)|click\([0-9]{1,4}, ?[0-9]{1,4}\)|type\("[^"\n]{1,64}"\))$'},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'deadline_ms': {'type': 'integer', 'minimum': 1, 'maximum': max_cover_ms},
        'risk': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'expected_state': {'type': 'string', 'minLength': 1, 'maxLength': 64},
        'children': {'type': 'array', 'maxItems': max_children,
                     'items': {'type': 'string', 'minLength': 1, 'maxLength': 12}}}
    properties = {
        'cover_ms': {'type': 'integer', 'minimum': required_cover_ms, 'maximum': max_cover_ms},
        'root_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
        'nodes': {'type': 'array', 'minItems': 1, 'maxItems': max_nodes,
            'items': {'type': 'object', 'properties': node,
                      'required': list(node), 'additionalProperties': False}},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 96}}
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def probability(x):
    return type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1


def validate_tree(tree, *, width, height, required_cover_ms, max_cover_ms=60000,
                  max_nodes=10, max_depth=2, max_children=3, risk_horizons_ms=None):
    def require(condition, message):
        if not condition:
            raise InvalidOutput(message)
    require(isinstance(tree, dict) and set(tree) == {'reason', 'cover_ms', 'root_id', 'nodes'},
            'Invalid tree fields')
    require(isinstance(tree['reason'], str) and bool(tree['reason'].strip()), 'Missing reason')
    require(type(tree['cover_ms']) is int and required_cover_ms <= tree['cover_ms'] <= max_cover_ms,
            'Insufficient or excessive coverage')
    require(isinstance(tree['root_id'], str), 'Invalid root id')
    require(isinstance(tree['nodes'], list) and 1 <= len(tree['nodes']) <= max_nodes, 'Node budget')
    nodes, actions = {}, {}
    fields = set(tree_schema()['properties']['nodes']['items']['required'])
    for node in tree['nodes']:
        require(isinstance(node, dict) and set(node) == fields, 'Invalid node fields')
        for key in ('id', 'guard', 'expected_state', 'action'):
            require(isinstance(node[key], str) and bool(node[key].strip()), f'Invalid {key}')
        require(node['id'] != 'NONE' and node['id'] not in nodes, 'Duplicate or reserved node id')
        require(type(node['depth']) is int and 0 <= node['depth'] <= max_depth, 'Depth budget')
        require(probability(node['confidence']), 'Invalid planner confidence')
        require(type(node['deadline_ms']) is int and 0 < node['deadline_ms'] <= tree['cover_ms'],
                'Invalid deadline')
        require(isinstance(node['risk'], str) and node['risk'] in ('low', 'medium', 'high'), 'Invalid risk')
        children = node['children']
        require(isinstance(children, list) and len(children) <= max_children
                and all(isinstance(c, str) for c in children), 'Branch budget')
        require(len(set(children)) == len(children), 'Duplicate children')
        action = parse_command(node['action'], width, height)
        if action is not None and risk_horizons_ms is not None:
            require(node['deadline_ms'] <= risk_horizons_ms[node['risk']], 'Risk horizon exceeded')
        nodes[node['id']], actions[node['id']] = dict(node), action
    require(tree['root_id'] in nodes, 'Missing root')
    root = nodes[tree['root_id']]
    require(root['depth'] == 0 and root['action'] in ('WAIT', 'wait', 'wait()'), 'Root must wait at depth zero')
    seen = set()
    def visit(ident, depth):
        require(ident in nodes and ident not in seen, 'Unresolved child, cycle, or multiple parents')
        seen.add(ident)
        require(nodes[ident]['depth'] == depth, 'Inconsistent depth')
        for child in nodes[ident]['children']:
            visit(child, depth + 1)
    visit(root['id'], 0)
    require(len(seen) == len(nodes), 'Unreachable nodes')
    return nodes, actions


class ChangeGate:
    def __init__(self, *, pixel_threshold=20, min_changed_fraction=0.005, static_probe_ns=500_000_000):
        self.pixel_threshold, self.fraction = pixel_threshold, min_changed_fraction
        self.probe_ns = static_probe_ns
        self.previous = None
        self.last_ns = None

    def accepts(self, image, timestamp_ns):
        gray = np.asarray(image.convert('L'), dtype=np.int16)
        changed = (self.previous is None or
                   np.mean(np.abs(gray - self.previous) >= self.pixel_threshold) >= self.fraction)
        probe = self.last_ns is None or timestamp_ns - self.last_ns >= self.probe_ns
        if changed or probe:
            self.previous, self.last_ns = gray, timestamp_ns
            return True
        return False


class AAPTController(PixelController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planner_p95_ms = self.config.get('planner_p95_ms')
        if type(self.planner_p95_ms) is not int or self.planner_p95_ms <= 0:
            raise SetupError('AAPT needs an explicit planner_p95_ms budget from existing evidence')
        self.required_cover_ms = self.planner_p95_ms + 500
        self.max_cover_ms = self.config.get('max_cover_ms', 60000)
        if type(self.max_cover_ms) is not int or self.max_cover_ms < self.required_cover_ms:
            raise SetupError('Invalid AAPT coverage budget')
        limits = self.config.get('risk_horizons_ms')
        if limits is not None and (not isinstance(limits, dict) or set(limits) != {'low', 'medium', 'high'}
                or any(type(v) is not int or v < 0 for v in limits.values())):
            raise SetupError('risk_horizons_ms must specify nonnegative low/medium/high bounds')
        self.schema = tree_schema(required_cover_ms=self.required_cover_ms, max_cover_ms=self.max_cover_ms)
        self.gate = ChangeGate()
        self.tree = None
        self.nodes, self.actions = {}, {}
        self.eligible = []
        self.armed_ns = None
        self.gated = None
        self.last_routed_id = None
        self.selected = None

    def admit(self, obs):
        update = super().admit(obs)
        if self.gate.accepts(pil_image(obs, self.width, self.height), obs.publication_time_ns):
            self.gated = obs
        return update

    def ready_to_decide(self):
        if self.newest is None:
            return False
        if self.tree is None:
            return self.newest.observation_id != self.last_decided_id
        return self.gated is not None and self.gated.observation_id != self.last_routed_id

    def _plan(self, calls_start):
        obs = self.newest
        prompt = (
            f'Task: {self.instruction}\nPrepare a bounded conditional policy before acting. '
            f'Image geometry is {self.width}x{self.height}. Use only visible evidence and the task; '
            'do not assume unseen coordinates, values, or event times. '
            'Return a flat JSON tree with a depth-zero WAIT root and string child ids. '
            'Each executable branch needs a specific visual guard, expected_state, confidence, '
            'risk, positive deadline_ms, and a scalar click(x,y) or type("value") action. '
            'WAIT is also allowed. Never emit code or a catch-all executable branch. '
            f'cover_ms must be at least {self.required_cover_ms} and at most {self.max_cover_ms}. '
            'Use at most 10 nodes, depth 2, and 3 children per node. '
            'For confident low-risk outcomes use one successor; for uncertain outcomes branch '
            'over the most likely possibilities up to the budget; leave uncovered outcomes '
            'to abstention. Do not invent confidence calibration. '
            'Use the smallest useful tree and short strings: guards at most 96 characters, '
            'expected_state at most 64, reason at most 96, and ids at most 12. '
            'Do not repeat prose. Return all fields required by the supplied schema.')
        risk_limits = self.config.get('risk_horizons_ms')
        if risk_limits is not None:
            prompt += '\nConfigured maximum action deadline by risk: ' + str(risk_limits)
        messages = [user_message(prompt, [pil_image(obs, self.width, self.height)])]
        errors = []
        for _ in range(2):  # exactly one bounded retry of invalid planner output
            message = self.model.complete(messages,
                max_tokens=self.config.get('planner_max_tokens', 2048), temperature=0.2, schema=self.schema)
            try:
                tree = strict_json(message.get('content'))
                nodes, actions = validate_tree(tree, width=self.width, height=self.height,
                    required_cover_ms=self.required_cover_ms, max_cover_ms=self.max_cover_ms,
                    risk_horizons_ms=risk_limits)
                self.tree, self.nodes, self.actions = tree, nodes, actions
                self.armed_ns = self.clock()
                self.eligible = nodes[tree['root_id']]['children'][:]
                self.gated = obs if self.gated is None else self.gated
                self.last_routed_id = None
                return self.result(None, outcome='tree_armed', calls_start=calls_start,
                    tree=tree, planner_validation_errors=errors, preparation_observation_id=obs.observation_id,
                    armed_ns=self.armed_ns, required_cover_ms=self.required_cover_ms)
            except InvalidOutput as exc:
                errors.append(str(exc))
                messages.append({'role': 'user', 'content':
                    'The previous output was invalid: ' + str(exc)[:180] +
                    '. Return a complete compact JSON tree within the token budget, '
                    'using only the necessary nodes and short strings.'})
        self.tree = None
        return self.result(None, outcome='invalid_tree', calls_start=calls_start,
                           planner_validation_errors=errors)

    def _expired(self, now):
        return self.tree is not None and now - self.armed_ns > self.tree['cover_ms'] * 1_000_000

    def decide(self):
        calls_start = len(self.model.calls)
        self.last_decided_id = self.newest.observation_id
        if self.tree is None or self._expired(self.clock()):
            self.tree = None
            return self._plan(calls_start)
        obs = self.gated
        self.last_routed_id = obs.observation_id
        eligible = self.eligible[:]
        guards = [{'id': i, 'guard': self.nodes[i]['guard'],
                   'deadline_ms': self.nodes[i]['deadline_ms']} for i in eligible]
        properties = {'branch_id': {'type': 'string', 'enum': eligible + ['NONE']},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'reason': {'type': 'string'}}
        schema = {'type': 'object', 'properties': properties,
                  'required': list(properties), 'additionalProperties': False}
        # The observer gets guards, never permission to construct a new action.
        import json
        prompt = ('Classify the screenshot against the eligible visual guards. Return only '
                  'branch_id, confidence, and a short evidence reason as JSON. Choose NONE '
                  'when ambiguous or no guard is visible. Do not generate an action.\n' + json.dumps(guards))
        message = self.model.complete([user_message(prompt, [pil_image(obs, self.width, self.height)])],
                                     max_tokens=64, temperature=0.0, schema=schema)
        action, outcome, branch = None, 'abstain', None
        try:
            result = strict_json(message.get('content'))
            if (not isinstance(result, dict) or set(result) != set(properties)
                    or not probability(result['confidence']) or not isinstance(result['reason'], str)
                    or not isinstance(result['branch_id'], str)):
                raise InvalidOutput('Invalid observer response')
            branch = result['branch_id']
            if branch not in eligible:
                outcome = 'no_match' if branch == 'NONE' else 'ineligible_branch'
            elif self.nodes[branch]['confidence'] < 0.45:
                outcome = 'planner_confidence_below_floor'
            elif self._expired(self.clock()):
                outcome = 'expired_tree'
            elif self.clock() - obs.publication_time_ns > self.nodes[branch]['deadline_ms'] * 1_000_000:
                outcome = 'late_observer'
            else:
                action = self.actions[branch]
                outcome = 'prepared_action' if action else 'prepared_wait'
                self.eligible = self.nodes[branch]['children'][:]
                self.selected = (branch, obs)
        except InvalidOutput as exc:
            outcome = f'invalid_observer: {exc}'
        if action is None:
            self.tree = None  # replan on the next admitted frame; no free spin on stale pixels
        return self.result(action, outcome=outcome, calls_start=calls_start,
                           branch_id=branch, observation_id=obs.observation_id,
                           decision_age_ns=self.clock() - obs.publication_time_ns)

    def correct_at_commit(self, commit_time_ns):
        if self.selected is None:
            return None
        branch, obs = self.selected
        if (self._expired(commit_time_ns) or
                commit_time_ns - obs.publication_time_ns > self.nodes[branch]['deadline_ms'] * 1_000_000):
            return Restart('aapt_deadline_expired', obs.observation_id, {'branch_id': branch})
        return None

    def restart(self):
        self.tree, self.selected = None, None
        self.last_decided_id = None
