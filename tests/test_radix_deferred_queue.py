"""No GPU, service, inference, or benchmark calls; test queue handoff guards."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies'))
import queue_radix_after_qwen08 as q


def plan(model='radixark_27b'):
    jobs=[]
    for window in sorted(q.WINDOWS,reverse=True):
        for index in range(100):
            for arm in ['legacy_cpp','agents3_stripped']:
                for mode in ['live','replay','paused']:
                    jobs.append({'phase':'evaluation','task':'notification','family':'closing_window',
                        'model':model,'arm':arm,'mode':mode,'window_s':window,'index':index})
    return {'jobs':jobs}


def completed():
    return {'state':'complete','completed_trials':4800,'planned_trials':4800,'qualification_trials':0,'error':None}


def audited():
    return {'state':'finished','records':4800,'infrastructure_errors':0,'task_failures_retained':True}


class QueueTests(unittest.TestCase):
    def test_requires_actual_completion_and_final_audit_never_success_rate(self):
        self.assertEqual(q.predecessor_gate(completed(),audited()),(True,'predecessor_complete_and_audited'))
        for state in ['running','failed','stopped','paused','frozen_ready']:
            self.assertFalse(q.predecessor_gate({**completed(),'state':state},audited())[0])
        self.assertFalse(q.predecessor_gate({**completed(),'completed_trials':4799},audited())[0])
        self.assertFalse(q.predecessor_gate(completed(),{**audited(),'state':'auditing'})[0])
        self.assertFalse(q.predecessor_gate(completed(),{**audited(),'infrastructure_errors':1})[0])
        self.assertFalse(q.predecessor_gate(completed(),None)[0])
        self.assertTrue(q.predecessor_gate({**completed(),'successes':0},audited())[0])

    def test_entire_chain_exactly_planned_notification_evaluations(self):
        p=plan();self.assertEqual(len(q.validate_jobs(p,'radixark_27b')),4800)
        p['jobs'][2]['task']='select'
        with self.assertRaises(AssertionError):q.validate_jobs(p,'radixark_27b')
        p=plan();p['jobs'][2]['phase']='qualification'
        with self.assertRaises(AssertionError):q.validate_jobs(p,'radixark_27b')
        p=plan();p['jobs'][0],p['jobs'][1]=p['jobs'][1],p['jobs'][0]
        with self.assertRaises(AssertionError):q.validate_jobs(p,'radixark_27b')
        p=plan();p['jobs'][-1]=p['jobs'][0]
        with self.assertRaises(AssertionError):q.validate_jobs(p,'radixark_27b')

    def test_process_identity_checks_allow_interpreter_only(self):
        expected=['/venv/vllm','serve','/model','--port','8013']
        with patch.object(q,'command_line',return_value=[x.encode() for x in ['/venv/python',*expected]]):
            self.assertTrue(q.process_matches(12,expected))
        with patch.object(q,'command_line',return_value=[x.encode() for x in ['/venv/python',*expected[:-1],'8012']]):
            self.assertFalse(q.process_matches(12,expected))
        with patch.object(q,'command_line',return_value=[x.encode() for x in ['/tmp/unrelated',*expected]]):
            self.assertFalse(q.process_matches(12,expected))

    def test_running_or_failed_predecessor_does_not_touch_gpu_or_services(self):
        for state in ['running','failed','paused']:
            with self.subTest(state=state),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);pred=root/'predecessor';pred.mkdir()
                q.write(root/'queue_manifest.json',{'root':str(root),'predecessor':str(pred)})
                q.write(pred/'status.json',{**completed(),'state':state})
                q.write(pred/'audit_status.json',audited())
                with patch.object(q,'validate_pins'),patch.object(q.time,'sleep',side_effect=KeyboardInterrupt),patch.object(q.signal,'signal'),patch.object(q,'stop_previous_service') as stop,patch.object(q,'run_command') as command,patch.object(q,'start_service') as service:
                    with self.assertRaises(KeyboardInterrupt):q.run(root,1)
                    stop.assert_not_called();command.assert_not_called();service.assert_not_called()
                    self.assertFalse((root/'handoff_started.json').exists())

    def test_service_stop_refuses_wrong_pid_busy_or_shared_group(self):
        launch={'model':q.PREVIOUS_MODEL,'base_url':'http://127.0.0.1:8013/v1','pid':123,
                'command':['vllm','serve','model','--served-model-name',q.PREVIOUS_MODEL,'--port','8013']}
        with tempfile.TemporaryDirectory() as folder,patch.object(q,'process_alive',return_value=True),patch.object(q.os,'killpg') as kill:
            with patch.object(q,'process_matches',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'PID/command'):q.stop_previous_service(launch,Path(folder)/'stop.json')
            with patch.object(q,'process_matches',return_value=True),patch.object(q.os,'getpgid',return_value=1):
                with self.assertRaisesRegex(RuntimeError,'process group'):q.stop_previous_service(launch,Path(folder)/'stop.json')
            with patch.object(q,'process_matches',return_value=True),patch.object(q.os,'getpgid',return_value=123),patch.object(q,'idle_service',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'active or waiting'):q.stop_previous_service(launch,Path(folder)/'stop.json')
            with patch.object(q,'process_matches',return_value=True),patch.object(q,'process_starttime',return_value=300):
                with self.assertRaisesRegex(RuntimeError,'reused PID'):
                    q.stop_previous_service(launch,Path(folder)/'stop.json',{'pid':123,'starttime_ticks':200})
            kill.assert_not_called()

    def test_duplicate_handoff_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);q.write(root/'queue_manifest.json',{'root':str(root)})
            q.write(root/'handoff_started.json',{'pid':11})
            with self.assertRaisesRegex(RuntimeError,'already attempted'):q.run(root)

    def test_observed_service_precision_must_match_requested_pair(self):
        config=('Initializing a V1 LLM engine dtype=torch.bfloat16, quantization=modelopt_mixed, '
                'kv_cache_dtype=fp8_e4m3, max_seq_len=16384, served_model_name='+q.MODEL+', enable_prefix_caching=False')
        self.assertEqual(q.service_runtime_configuration(config),config)
        for expected in ['torch.bfloat16','modelopt_mixed','fp8_e4m3','16384',q.MODEL]:
            with self.assertRaises(RuntimeError):q.service_runtime_configuration(config.replace(expected,'unexpected'))
        with self.assertRaises(RuntimeError):q.service_runtime_configuration('API server ready')

    def test_completion_handoff_order_without_real_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pred=root/'pred';out=root/'experiment';pred.mkdir();out.mkdir()
            q.write(pred/'status.json',completed());q.write(pred/'audit_status.json',audited());q.write(root/'old_service.json',{})
            commands={x:[x] for x in ['build','verify','freeze','service','run','audit']}
            manifest={'root':str(root),'predecessor':str(pred),'gpu_lock':str(root/'gpu.lock'),
                      'old_service_launch':str(root/'old_service.json'),'commands':commands}
            q.write(root/'queue_manifest.json',manifest)
            events=[]
            def command(args,log):
                events.append(args[0])
                if args[0]=='freeze':
                    q.write(out/'execution_manifest.json',{'model':q.MODEL,'revision':q.REVISION,'planned_trials':4800,'qualification_trials':0})
                    q.write(out/'plan.json',{})
                    q.write(out/'status.json',{'state':'frozen_ready'})
            def popen(args,**kwargs):
                events.append(args[0]);p=Mock();p.pid=14 if args[0]=='run' else 15;p.poll.return_value=None
                q.write(out/'status.json',{'state':'running' if args[0]=='run' else 'complete','completed_trials':4800})
                return p
            with patch.object(q,'validate_pins'),patch.object(q,'validate_jobs'),patch.object(q,'predecessor_processes',return_value=[]),patch.object(q,'full_predecessor_audit',return_value={'state':'passed'}),patch.object(q,'stop_previous_service',side_effect=lambda *a:events.append('stop')),patch.object(q,'run_command',side_effect=command),patch.object(q,'start_service',side_effect=lambda *a:events.append('service')),patch.object(q.subprocess,'Popen',side_effect=popen),patch.object(q.signal,'signal'):
                q.run(root,1)
            self.assertEqual(events,['stop','build','verify','service','freeze','run','audit'])
            self.assertEqual(q.read(root/'queue_status.json')['state'],'successor_complete')
            self.assertTrue((root/'predecessor_completion_verification.json').exists())

if __name__=='__main__':unittest.main()
