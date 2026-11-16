"""Instrumented native Agent S3 service for the matched-controller study.

Uses the installed Agent S3 unchanged. A local forwarding proxy records every
planner/grounder request, including retries, and disables Qwen thinking for both
arms. Agent-generated executable strings are returned, never executed here.
"""
import argparse
import base64
import hashlib
import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from agents3_service import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8110)
    ap.add_argument('--proxy-port', type=int, default=8111)
    ap.add_argument('--planner-url', default='http://127.0.0.1:8010/v1')
    ap.add_argument('--grounder-url', default='http://127.0.0.1:8011/v1')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    planner, grounder = 'Qwen/Qwen3.8-27B', 'ByteDance-Seed/UI-TARS-1.5-7B'
    sessions, calls, lock = {}, [], threading.Lock()
    endpoints = {'planner': a.planner_url, 'grounder': a.grounder_url}

    class Base(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read(self):
            return json.loads(self.rfile.read(int(self.headers['Content-Length'])))

    class Proxy(Base):
        def do_POST(self):
            role = self.path.strip('/').split('/')[0]
            if role not in endpoints:
                return self.send(404, {'error': 'unknown model role'})
            body = self.read()
            body['temperature'] = 0.0
            if role == 'planner':
                body['chat_template_kwargs'] = {'enable_thinking': False}
            request_bytes = json.dumps(body).encode()
            start = time.monotonic_ns()
            record = {'role': role, 'model': body['model'], 'start_ns': start,
                      'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
                      'max_tokens': body.get('max_tokens'), 'temperature': body['temperature'],
                      'top_p': body.get('top_p'), 'repetition_penalty': body.get('repetition_penalty')}
            try:
                req = urllib.request.Request(endpoints[role] + '/chat/completions',
                      data=request_bytes, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=600) as response:
                    result = json.load(response)
                code = 200
            except urllib.error.HTTPError as exc:
                code = exc.code
                result = {'error': exc.read().decode(errors='replace')[:2000]}
            except Exception as exc:
                code, result = 502, {'error': f'{type(exc).__name__}: {exc}'}
            record.update(end_ns=time.monotonic_ns(), http_status=code, usage=result.get('usage'),
                          response=result)
            calls.append(record)
            with (a.out / 'model_calls.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            self.send(code, result)

    class Service(Base):
        def do_GET(self):
            self.send(200, {'data': [{'id': planner + '+' + grounder}],
                           'planner': planner, 'grounder': grounder,
                           'thinking': False, 'native_agent_s3': True})

        def do_POST(self):
            req = self.read()
            png = base64.b64decode(req['png_b64'])
            w, h = Image.open(io.BytesIO(png)).size
            session_id = req.get('session_id', 'default')
            with lock:
                start, before = time.monotonic_ns(), len(calls)
                try:
                    if req.get('reset') or session_id not in sessions:
                        params = SimpleNamespace(planner=planner, grounder=grounder,
                            planner_url=f'http://127.0.0.1:{a.proxy_port}/planner/v1',
                            grounder_url=f'http://127.0.0.1:{a.proxy_port}/grounder/v1')
                        sessions[session_id] = Session(params, w, h)
                    session = sessions[session_id]
                    info, actions = session.agent.predict(instruction=req['instruction'],
                                                         observation={'screenshot': png})
                    own_calls = calls[before:]
                    if any(c['http_status'] != 200 for c in own_calls):
                        raise RuntimeError('Model endpoint error; inspect model_calls.jsonl')
                    if any(c['usage'] is None for c in own_calls):
                        raise RuntimeError('Model endpoint omitted usage accounting')
                    result = {'exec_code': list(actions), 'plan_code': (info or {}).get('plan_code'),
                        'model_reported': planner + '+' + grounder, 'elapsed_ns': time.monotonic_ns()-start,
                        'grounding_wh': list(session.grounding_wh), 'screen_wh': [w,h],
                        'model_calls': own_calls,
                        'input_tokens': sum(c['usage']['prompt_tokens'] for c in own_calls),
                        'output_tokens': sum(c['usage']['completion_tokens'] for c in own_calls)}
                    self.send(200, result)
                except Exception as exc:
                    self.send(500, {'error': f'{type(exc).__name__}: {exc}', 'model_calls': calls[before:]})
                finally:
                    calls.clear()
                    # Trials explicitly reset; retain only the active trajectory to bound memory.
                    for key in list(sessions):
                        if key != session_id:
                            del sessions[key]

    proxy = ThreadingHTTPServer(('127.0.0.1', a.proxy_port), Proxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    print(f'Agent S3 matched service :{a.port}; proxy :{a.proxy_port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Service).serve_forever()


if __name__ == '__main__':
    main()
