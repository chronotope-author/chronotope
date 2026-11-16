"""Stateless latest-frame argument updater around an unchanged GUI controller.

Only the original model observation, pending action and newest admitted pixels
are retained. No event ledger, dependency matrix, restart or extra model call.
The target's RGB patch is its visual identity; uncertain matches abstain.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from pathlib import Path
import time

import numpy as np

from .controller import Correction
from .hashing import hash_obj
from .png import to_rgb
from .tracker import AMBIGUITY_MARGIN, PLATEAU_MAX, Match, cut_template, track


@dataclass(frozen=True)
class ActionAbstention:
    reason: str
    observation_id: int | None
    detail: dict


def register():
    from .spec import _CONTROLLER_SCHEMAS
    _CONTROLLER_SCHEMAS['agents3_latest_frame'] = dict(
        _CONTROLLER_SCHEMAS['vlm_direct'],
        kind=(str, ('agents3_latest_frame',)),
        updater_backend=(str, ('cuda_ncc', 'numpy_ncc')),
        updater_library=(str, None), template_radius_px=(int, None),
        min_match_score=(float, None), identity_max_mae=(float, None),
        safe_policy=(str, ('abstain',)))


class CudaNCC:
    """Only the existing vision kernels; no Core/control/ledger is linked."""
    def __init__(self, library, w, h, radius):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.lfu_create.argtypes = [ctypes.c_int] * 3
        self.lib.lfu_create.restype = ctypes.c_void_p
        self.lib.lfu_destroy.argtypes = [ctypes.c_void_p]
        self.lib.lfu_track.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.lib.lfu_track.restype = ctypes.c_int
        self.handle = self.lib.lfu_create(w, h, radius)
        if not self.handle:
            raise RuntimeError('Standalone CUDA updater allocation failed')

    def match(self, before, newest, x, y):
        out = (ctypes.c_double * 5)()
        if self.lib.lfu_track(self.handle, before, newest, x, y, out):
            raise RuntimeError('Standalone CUDA updater failed')
        return Match(int(out[0]), int(out[1]), out[2], out[3], int(out[4]))

    def close(self):
        if self.handle:
            self.lib.lfu_destroy(self.handle)
            self.handle = None


class LatestFrameController:
    """Wrap any existing click/type controller; preserve its complete policy.

NCC uses the existing grayscale v2 matcher. An additional RGB reconstruction
check refuses a different-looking target, including a recolored operation.
Code uses the existing fixed-font visual decoder (a declared benchmark prior).
Neither path receives a manifest, task identity, clock deadline or target color.
"""
    def __init__(self, base, *, width, height, backend='numpy_ncc', library='',
                 radius=12, min_score=0.8, identity_max_mae=2.0):
        self.base, self.w, self.h = base, width, height
        self.radius, self.min_score = radius, min_score
        self.identity_max_mae = identity_max_mae
        self.backend = backend
        if backend not in ('cuda_ncc', 'numpy_ncc'):
            raise ValueError(backend)
        self.matcher = CudaNCC(library, width, height, radius) if backend == 'cuda_ncc' else None
        self.newest = self.original = self.pending = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def spec_hash(self):
        return hash_obj(dict(kind='agents3_latest_frame', base=self.base.spec_hash(),
            backend=self.backend, radius=self.radius, min_score=self.min_score,
            identity_max_mae=self.identity_max_mae, safe_policy='abstain',
            version=1))

    def admit(self, obs):
        self.newest = obs
        return self.base.admit(obs)

    def decide(self):
        self.original = self.newest
        decision = self.base.decide()
        self.pending = dict(decision.action) if decision.action else None
        return decision

    def correct_at_commit(self, commit_time_ns):
        start = time.perf_counter_ns()
        if self.pending is None:
            return None
        old, new, action = self.original, self.newest, dict(self.pending)
        detail = dict(updater='latest_frame_v1', updater_backend=self.backend,
            original_observation_id=old.observation_id,
            original_payload_sha256=old.payload_hash, newest_payload_sha256=new.payload_hash,
            original_action=dict(action), additional_model_calls=0,
            target_identity='original RGB patch' if action['kind'] == 'click' else 'original visual code panel',
            operation_changed_by_updater=False)

        def abstain(reason):
            return ActionAbstention(reason, new.observation_id,
                dict(detail, track_ns=time.perf_counter_ns()-start, safe_policy='abstain'))

        if action['kind'] in ('click', 'type') and old.payload_hash == new.payload_hash:
            # The complete visual state is unchanged. Preserve the original
            # model action (including competence errors) without re-grounding.
            detail.update(unchanged_frame=True, matched=True, match_score=1.0,
                          moved_px=0.0, track_ns=time.perf_counter_ns()-start)
            return Correction(action, 'state_at_B', new.observation_id, detail)

        if action['kind'] == 'type':
            from .decoder import decode
            a = decode(old.payload, self.w, self.h)
            b = decode(new.payload, self.w, self.h)
            detail.update(decoder='existing fixed-font decoder', old_decoder_reason=a.reason,
                          new_decoder_reason=b.reason)
            if not a.value or not b.value or a.panel != b.panel:
                return abstain('code_panel_unreliable')
            # Preserve ordinary model errors when the displayed value did not change.
            if a.value != b.value:
                action['value'] = b.value
            detail.update(matched=True, match_score=1.0, moved_px=0.0,
                          value_changed=action['value'] != self.pending['value'])
            binding = 'decoded_at_D'
        elif action['kind'] == 'click':
            x, y = action['x'], action['y']
            if not (0 <= x < self.w and 0 <= y < self.h):
                return abstain('proposal_out_of_bounds')
            before = to_rgb(old.payload, self.w, self.h)
            newest = to_rgb(new.payload, self.w, self.h)
            template = cut_template(before, self.w, self.h, x, y, self.radius)
            match = (self.matcher.match(before, newest, x, y) if self.matcher else
                     track(template, newest, self.w, self.h))
            ambiguous = match.second >= match.score - AMBIGUITY_MARGIN or match.plateau > PLATEAU_MAX
            detail.update(match_score=match.score, second_score=match.second,
                plateau=match.plateau, ambiguous=ambiguous, matched=False, moved_px=0.0)
            if match.score < self.min_score or ambiguous:
                return abstain('target_missing_or_ambiguous')
            # Same target appearance, not merely another high-correlation color/label.
            original_patch = cut_template(before, self.w, self.h, x, y, self.radius, rgb=True).pixels
            yy, xx = match.y-template.dy, match.x-template.dx
            side = 2*self.radius+1
            current_patch = np.frombuffer(newest, np.uint8).reshape(self.h,self.w,3)[yy:yy+side,xx:xx+side]
            if current_patch.shape != original_patch.shape:
                return abstain('target_patch_out_of_bounds')
            mae = float(np.abs(original_patch.astype(np.float32)-current_patch).mean())
            detail['identity_rgb_mae'] = mae
            if mae > self.identity_max_mae:
                return abstain('target_appearance_changed_operation_unverified')
            action.update(x=match.x, y=match.y)
            detail.update(matched=True, moved_px=float(np.hypot(match.x-x,match.y-y)))
            binding = 'tracked_at_D'
        else:
            return abstain('unsupported_operation')
        detail['track_ns'] = time.perf_counter_ns()-start
        return Correction(action, binding, new.observation_id, detail)

    def close(self):
        if self.matcher:
            self.matcher.close()
        close = getattr(self.base, 'close', None)
        if close:
            close()


def wrap_from_config(base, config, width, height):
    if config['kind'] != 'agents3_latest_frame':
        return base
    if config['safe_policy'] != 'abstain':
        raise ValueError('Only the predefined safe no-action policy is supported')
    return LatestFrameController(base, width=width, height=height,
        backend=config['updater_backend'], library=config['updater_library'],
        radius=config['template_radius_px'], min_score=config['min_match_score'],
        identity_max_mae=config['identity_max_mae'])
