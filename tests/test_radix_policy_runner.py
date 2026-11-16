"""CPU-only guards against accidental cohort/model/policy changes."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies'))
import run_radix_agents3_policy as runner
from agents3_policy_contract import policy_fingerprint


class RadixPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=json.loads((runner.PREDECESSOR/'plan.json').read_text())
        cls.plan=json.loads((ROOT/'runs/agents3_policy_radixark27b_20261028/experiment/plan.json').read_text())

    def test_exact_evaluation_cohort(self):
        runner.validate_plan(self.plan);runner.same_cohort(self.plan,self.parent)
        self.assertEqual(len(self.plan['jobs']),4800)
        self.assertEqual({j['task'] for j in self.plan['jobs']},{'notification'})
        self.assertEqual({j['phase'] for j in self.plan['jobs']},{'evaluation'})

    def test_changed_cohort_rejected(self):
        changed=copy.deepcopy(self.plan);changed['jobs'][0]['episode_id']='different'
        with self.assertRaises(AssertionError):runner.same_cohort(changed,self.parent)

    def test_extra_phase_and_mixed_checkpoint_rejected(self):
        for field,value in [('phase','qualification'),('task','select'),('model','qwen35_0p8b'),('window_s',64.)]:
            changed=copy.deepcopy(self.plan);changed['jobs'][0][field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError):runner.validate_plan(changed)

    def test_duplicate_and_missing_case_rejected(self):
        changed=copy.deepcopy(self.plan);changed['jobs'][1]=changed['jobs'][0]
        with self.assertRaises(AssertionError):runner.validate_plan(changed)

    def test_policy_only_model_and_template_change(self):
        old=json.loads((runner.PREDECESSOR/'configs/agents3_policy.json').read_text())
        new=json.loads((ROOT/'runs/agents3_policy_radixark27b_20261028/policy_inputs/agents3_policy.json').read_text())
        self.assertEqual(old['policy_sha256'],policy_fingerprint(old))
        self.assertEqual(new['policy_sha256'],policy_fingerprint(new))
        for key in set(old)-{'model','policy_sha256'}:self.assertEqual(old[key],new[key],key)
        self.assertFalse(new['native_chat_template']['strip_historical_thinking'])

    def test_explicit_service_precision_required(self):
        good={'model':runner.MODEL,'revision':runner.REVISION,'dtype':'bfloat16','quantization':'modelopt_mixed',
            'kv_cache_dtype':'fp8_e4m3','calculate_kv_scales':False,'max_model_len':16384,
            'max_num_seqs':1,'enable_prefix_caching':False}
        runner.validate_service(good)
        for key,value in [('dtype','float16'),('quantization',None),('kv_cache_dtype','auto'),('calculate_kv_scales',True),('revision','main')]:
            with self.subTest(field=key),self.assertRaises(AssertionError):runner.validate_service({**good,key:value})


if __name__=='__main__':unittest.main()
