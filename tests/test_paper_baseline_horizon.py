"""In-memory Notification fixtures; no services or experimental model trials."""
from chronotope.controller import ControlUpdate, Decision
from chronotope.runner import run_trial, PayloadStore
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict


class SlowNoAction:
    def __init__(self):
        self.latest = self.decided = None
    def spec_hash(self): return '0' * 64
    def admit(self, obs):
        self.latest = obs.observation_id
        return ControlUpdate('action', str(self.latest))
    def ready_to_decide(self): return self.latest != self.decided
    def decide(self):
        self.decided = self.latest
        return Decision(None, 100_000_000_000)


def test_opt_in_horizon_prevents_another_decision_after_slow_no_action(tmp_path):
    spec = TrialSpec.from_dict(base_spec_dict())
    store = PayloadStore(tmp_path/'payloads')
    old = Trace.load(run_trial(spec,tmp_path/'old.jsonl',store,controller=SlowNoAction()))
    new = Trace.load(run_trial(spec,tmp_path/'new.jsonl',store,controller=SlowNoAction(),enforce_horizon=True))
    assert len(old.all('decision_begin')) == 2
    assert len(new.all('decision_begin')) == 1
    assert new.first('trial_end')['reason'] == 'horizon'
