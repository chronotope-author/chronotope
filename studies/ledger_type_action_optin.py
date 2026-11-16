#!/usr/bin/env python3
"""Prepare (or explicitly apply) the minimal legacy four-digit type transport.

Default execution writes a reviewable patch only. Application requires an
explicit --apply-after-user-approval flag and never builds or launches anything.
It changes no detector, GPU control, history, reflection, or model policy.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def replace(text, before, after):
    assert text.count(before) == 1, before[:80]
    return text.replace(before, after)


def proposed_files(source):
    files = {}
    p = source / 'src/planner_vlm.h'; t = p.read_text()
    files[p] = replace(t, '    bool click = false;           // false = wait / unparseable',
                      '    bool type4 = false;           // opt-in transport for existing type(value) action\n'
                      '    int type_value = 0;           // numeric core payload; preserve width with type_text\n'
                      '    std::string type_text;\n'
                      '    bool click = false;           // false = wait / unparseable')
    p = source / 'src/agents3_prompt.h'; t = p.read_text()
    helper = '''// Parse a model-provided four-digit string only; no OCR or state refresh.
inline std::string parse_type4(const std::string& response) {
    const auto begin = response.find('{'), end = response.rfind('}');
    if (begin == std::string::npos || end == std::string::npos || end < begin) return {};
    const auto value = Json::parse(response.substr(begin, end - begin + 1), nullptr, false);
    if (value.is_discarded() || !value.is_object() || !value.contains("action") ||
        !value.at("action").is_string() || value.at("action").get<std::string>() != "type" ||
        !value.contains("value") || !value.at("value").is_string()) return {};
    const auto digits = value.at("value").get<std::string>();
    if (digits.size() != 4 || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) return {};
    return digits;
}

'''
    files[p] = replace(t, 'inline Json parse_action_once(', helper + 'inline Json parse_action_once(')
    p = source / 'src/planner_trt.cpp'; t = p.read_text()
    old = '        } else vlm_parse_action(text, &a, impl_->cfg.prompt_version, impl_->cfg.w, impl_->cfg.h, impl_->cfg.image_scale);'
    new = '''        } else {
            vlm_parse_action(text, &a, impl_->cfg.prompt_version, impl_->cfg.w, impl_->cfg.h, impl_->cfg.image_scale);
            if (!a.click && fixed_prompt && impl_->cfg.prompt_version == 11 &&
                impl_->prompt_contract.value("allow_type4", false)) {
                a.type_text = agents3_prompt::parse_type4(text);
                a.type4 = !a.type_text.empty();
                if (a.type4) a.type_value = std::stoi(a.type_text);
            }
        }'''
    t = replace(t, old, new)
    old = '        audit["ttft_ms"] = impl_->prefill_ms; audit["decode_ms"] = impl_->decode_ms;'
    new = '''        audit["parsed_action"]["type4"] = a.type4;
        audit["parsed_action"]["type_text"] = a.type_text;
        audit["ttft_ms"] = impl_->prefill_ms; audit["decode_ms"] = impl_->decode_ms;'''
    files[p] = replace(t, old, new)
    p = source / 'src/serve.cpp'; t = p.read_text()
    old = '            if (ans.ok && ans.click) {\n                chronotope_proposal p{}; int rc = chronotope_propose(core, CHRONOTOPE_ACT_CLICK, ans.x, ans.y, 0, &p);'
    new = '''            j.b("type4", ans.type4).s("type_text", ans.type_text).i("type_value", ans.type_value);
            if (ans.ok && (ans.click || ans.type4)) {
                chronotope_proposal p{}; int rc = chronotope_propose(core,
                    ans.type4 ? CHRONOTOPE_ACT_TYPE : CHRONOTOPE_ACT_CLICK, ans.x, ans.y, ans.type_value, &p);'''
    files[p] = replace(t, old, new)
    p = source / 'python/chronotope_core/controller.py'; t = p.read_text()
    old = '            if not (r.get("answer_ok") and r.get("click") and r.get("proposal")):'
    new = '''            if r.get("answer_ok") and r.get("type4") and r.get("proposal"):
                value = r["type_text"]
                assert isinstance(value, str) and len(value) == 4 and value.isascii() and value.isdigit()
                assert int(value) == r["type_value"] == r["proposal"]["value"]
                self._decided = True
                self._proposal = {"kind": "type", "value": value}
                return Decision(action=dict(self._proposal), declared_compute_ns=None,
                                invocation_id=str(r.get("invocation_id")), invocation_detail={**detail, "outcome": "type"})
            if not (r.get("answer_ok") and r.get("click") and r.get("proposal")):'''
    t = replace(t, old, new)
    t = replace(t, '        px, py = self._proposal["x"], self._proposal["y"]',
                '        px, py = self._proposal.get("x", 0), self._proposal.get("y", 0)')
    old = '            return Correction({"kind": "click", "x": int(r["x"]), "y": int(r["y"])}, binding,'
    new = '''            if r.get("kind_name") == "type":
                value = self._proposal["value"]
                assert self._proposal["kind"] == "type" and int(value) == int(r["value"])
                return Correction({"kind": "type", "value": value}, "state_at_B",
                                  self._proposal_observation_id(), {**detail, "text_correction": "none; model value at B"})
            return Correction({"kind": "click", "x": int(r["x"]), "y": int(r["y"])}, binding,'''
    files[p] = replace(t, old, new)
    return files


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--apply-after-user-approval', action='store_true'); args = ap.parse_args()
    root = args.root.resolve(); source = root / 'build_source/chronotope'
    proposed = proposed_files(source)
    diff = ''.join(''.join(difflib.unified_diff(p.read_text().splitlines(True), text.splitlines(True),
                   fromfile='a/' + str(p.relative_to(source)), tofile='b/' + str(p.relative_to(source))))
                   for p, text in proposed.items())
    (root / 'inputs/native_type_action_optin.patch').write_text(diff)
    if args.apply_after_user_approval:
        assert not (root / 'native_build_result.json').exists(), 'Do not modify built experiment sources'
        for p, text in proposed.items(): p.write_text(text)
        prep_path = root / 'native_source_preparation.json'; prep = json.loads(prep_path.read_text())
        for p in proposed: prep['source_hashes'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        prep['application_changed_files'] = sorted(set(prep['application_changed_files']) | {str(p.relative_to(source)) for p in proposed})
        prep['action_transport_extension'] = 'Explicitly authorized type(value), four digits; no updater'
        prep_path.write_text(json.dumps(prep, indent=2) + '\n')
    print(json.dumps({'patch': str(root / 'inputs/native_type_action_optin.patch'),
                      'applied': args.apply_after_user_approval, 'files': len(proposed), 'inference_calls': 0}))


if __name__ == '__main__': main()
