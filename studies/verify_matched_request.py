"""CPU-only parity checks against the actual frozen native planner source.

Compiles only planner_vlm.cpp (no CUDA or model runtime), then verifies the
prompt, representative parser behavior, canvas conversion and PNG RGB pixels.
These are implementation checks, not experimental or qualification trials.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = [
    '{"action":"click","x":350,"y":200}',
    '{ "y": 200, "action" : "click", "x" : 350 }',
    '```json\n[{"action":"click","coordinate":[350,200]}]\n```',
    '```json\n[{"action":"click","coordinate":[350,200]',
    '{"action":"click","coordinate":[350.49,200.51]}',
    '{"action":"click","x":-10,"y":-3}',
    '{"action":"click","coordinate":[-10.6,-3.6]}',
    '{"action":"click","x":350.7,"y":200.9}',
    '{"action":"click","coordinate":[3.5e2, 2e2]}',
    '{"action":"click","coordinate":[350\t,200]}',
    '{"action":"click","coordinate":[350,\n200]}',
    '{"action":"cl\\u0069ck","x":350,"y":200}',
    '{"action":"click","x":350,"coordinate":[123,456]}',
    '{"action":"click","x":"350","y":200}',
    '{"action":"click","x":- ,"y":200}',
    '{"action":"click","x":0,"y":1000}',
    '{"action":"wait"}', '{"action":"done"}', 'DONE', '',
    '{"action":"click"}', '{"action":"click","coordinate":[350]}',
    '{"action":"click","coordinate":[350,"bad"]}',
    '{"x":350,"y":200}',
]
WRAPPER = r'''
#include "planner_vlm.h"
#include <fstream>
#include <string>
#include <vector>
extern "C" const char* matched_prompt() {
    static std::string s=chronotope::vlm_system_prompt(4,320,240,2);
    return s.c_str();
}
extern "C" int matched_parse(const char* s, int* out) {
    chronotope::VlmAnswer a;
    bool click=chronotope::vlm_parse_action(s,&a,4,320,240,2);
    out[0]=a.image_x;out[1]=a.image_y;out[2]=a.x;out[3]=a.y;
    return click ? 1 : 0;
}
extern "C" int matched_png(const unsigned char* p, int w, int h, int scale, const char* path) {
    std::vector<uint8_t> rgb(p,p+(size_t)w*h*3);
    auto png=chronotope::VlmPlanner::encode_png(rgb,w,h,scale);
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(png.data()),png.size());
    return f.good() && !png.empty();
}
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(code, adapter_path=None, native_config=None):
    from PIL import Image
    code = Path(code).resolve()
    adapter_path = Path(adapter_path or ROOT/'studies/agents3_matched_request.py').resolve()
    native_dir = code/'chronotope/src'
    config = Path(native_config).resolve() if native_config else code.parent/'configs/legacy_cpp.json'
    if not config.exists():
        config = code.parent/'configs/closing_window_keep.json'
    cfg = json.loads(config.read_text())
    adapter = load_module(adapter_path, '_matched_request_adapter_verification')
    png_module = load_module(code/'src/chronotope/png.py', '_matched_request_png_verification')
    contract = adapter.REQUEST_CONTRACT
    for key, want in {'prompt_version':4, 'image_scale':2, 'max_tokens':64,
                      'temperature':0.0, 'enable_thinking':False,
                      'planner':'trt', 'server':'cpp_socket'}.items():
        assert cfg[key] == want, f'Native configuration differs: {key}'
    for key, want in {'max_tokens':64, 'temperature':0.0, 'top_p':1.0, 'top_k':1,
                      'repetition_penalty':1.0, 'stop':['}'], 'enable_thinking':False,
                      'coordinate_space':'norm1000', 'history':'none', 'reflection':False,
                      'format_retries':0,
                      'message_order':['system:text','user:image','user:text']}.items():
        assert contract[key] == want, f'Host request contract differs: {key}'
    trt_source = (native_dir/'planner_trt.cpp').read_text()
    required_native = [
        'impl_->system = vlm_system_prompt(impl_->cfg.prompt_version, impl_->cfg.w, impl_->cfg.h, impl_->cfg.image_scale);',
        'req.temperature = (float)impl_->cfg.temperature;',
        'req.topP = 1.0f;', 'req.topK = 1;',
        'req.maxGenerateLength = impl_->cfg.max_tokens;',
        'req.enableThinking = impl_->cfg.enable_thinking;',
        'sys.role = "system"; sys.contents.push_back({"text", impl_->system});',
        'usr.contents.push_back({"image", ""});',
        'usr.contents.push_back({"text", '+json.dumps(contract['user_text'])+'});',
        'r.messages = {sys, usr};', 'r.stopStrings = {"}"};',
    ]
    for fragment in required_native:
        assert fragment in trt_source, f'Native request source changed: {fragment}'
    assert trt_source.index(required_native[7]) < trt_source.index(required_native[8]), 'Native image/text order changed'
    compiler = shutil.which('g++') or shutil.which('c++')
    assert compiler, 'A C++ compiler is required for native implementation verification'
    fixtures = []
    with tempfile.TemporaryDirectory(prefix='chronotope_request_parity_') as tmp:
        tmp = Path(tmp); wrapper=tmp/'wrapper.cpp'; wrapper.write_text(WRAPPER)
        lib_path=tmp/'native_contract.so'
        command=[compiler,'-std=c++17','-O2','-shared','-fPIC','-I',str(native_dir),
                 str(wrapper),str(native_dir/'planner_vlm.cpp'),'-lz','-pthread','-o',str(lib_path)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        native=ctypes.CDLL(str(lib_path))
        native.matched_prompt.restype=ctypes.c_char_p
        native.matched_parse.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
        native.matched_parse.restype=ctypes.c_int
        native.matched_png.argtypes=[ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p]
        native.matched_png.restype=ctypes.c_int
        prompt=native.matched_prompt().decode()
        assert prompt == contract['system'], 'Host/native system prompt text differs'
        for text in FIXTURES:
            values=(ctypes.c_int*4)()
            clicked=bool(native.matched_parse(text.encode(),values))
            actual=adapter.parse_native_action(text)
            expected={'action':'click','x':values[0],'y':values[1]} if clicked else {'action':'wait'}
            assert actual==expected, f'Parser mismatch for {text!r}: host={actual}, native={expected}'
            canvas=None
            if clicked:
                canvas=adapter.native_canvas_coordinates(actual['x'],actual['y'],320,240)
                assert canvas=={'kind':'click','x':values[2],'y':values[3]}, f'Canvas conversion mismatch for {text!r}'
            fixtures.append({'input':text,'normalized_action':expected,'canvas_action':canvas})
        # Deterministic RGB test pattern exercises all channels, edges and row order.
        w,h,scale=320,240,2
        rgb=bytes((x*7+y*11+c*83)%256 for y in range(h) for x in range(w) for c in range(3))
        data=(ctypes.c_ubyte*len(rgb)).from_buffer_copy(rgb); png_path=tmp/'native.png'
        assert native.matched_png(data,w,h,scale,str(png_path).encode())
        host_png=png_module.encode_png(rgb,w,h,scale)
        native_im=Image.open(png_path).convert('RGB')
        host_im=Image.open(io.BytesIO(host_png)).convert('RGB')
        assert native_im.size==host_im.size==(640,480)
        assert native_im.tobytes()==host_im.tobytes(), 'Decoded PNG RGB pixels differ'
        expected_rgb=b''.join(b''.join(rgb[(y//2*w+x//2)*3:(y//2*w+x//2)*3+3] for x in range(w*2)) for y in range(h*2))
        assert host_im.tobytes()==expected_rgb, 'Upscaling is not integer nearest-neighbour'
        pixel_sha=hashlib.sha256(host_im.tobytes()).hexdigest()
    files=[native_dir/'planner_vlm.cpp',native_dir/'planner_vlm.h',native_dir/'planner_trt.cpp',
           code/'src/chronotope/png.py',adapter_path,config]
    return {'state':'passed','created_at':datetime.now(timezone.utc).isoformat(),
            'verification':'Compiled actual frozen C++ prompt/parser/PNG implementation; no CUDA or model execution',
            'code_root':str(code),'native_config':str(config),'adapter':str(adapter_path),
            'request_contract_sha256':adapter.contract_fingerprint(),
            'source_hashes':{str(p):digest(p) for p in files},
            'system_prompt_exact_match':True,'system_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'native_request_source_checks':len(required_native),'parser_cases_matched':len(fixtures),
            'parser_cases':fixtures,'decoded_rgb_exact_match':True,
            'image_wh':[640,480],'image_rgb_sha256':pixel_sha,
            'generation_settings':{k:contract[k] for k in ['max_tokens','temperature','top_p','top_k','repetition_penalty','stop','enable_thinking']},
            'experimental_trials':0,'inference_calls':0,
            'limitations':['Parser checks cover the listed fixtures, not all malformed byte sequences',
                'PNG check verifies C++/Python CPU encoder pixels; GPU image preprocessing is audited separately',
                'No assertion of identical backend numerical kernels or generated outputs']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,default=ROOT/'studies/agents3_matched_request.py')
    parser.add_argument('--native-config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=verify(args.code,args.adapter,args.native_config)
    except Exception as exc:
        result={'state':'failed','error':f'{type(exc).__name__}: {exc}','experimental_trials':0,'inference_calls':0}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        raise
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['state','parser_cases_matched','system_prompt_exact_match','decoded_rgb_exact_match','experimental_trials','inference_calls']}))

if __name__=='__main__':main()
