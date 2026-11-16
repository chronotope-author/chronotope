// K4b (2026-08-27): the agent as a server process. The referee (the Python gym) connects on a UNIX socket and drives
// the SAME operations the C ABI exposes (admit / begin_decision / decide / propose / resolve_and_commit / restart /
// audit / ledger ...), one request per line of flat JSON, a raw payload after "admit" and "track" requests, one JSON
// reply line per request ("read_frame" replies with a header line then the raw RGB). The model call ("decide_trt")
// happens inside this process on the frame at B, which stays on the device (planner_trt). Field names of the replies
// are those of the C structs, so the Python client returns the same dicts as the ctypes wrapper (chronotope/python/chronotope_core).
#include "serve.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "chronotope_api.h"
#include "planner_trt.h"
#ifdef CHRONOTOPE_HOSTED_API
#include "planner_api.h"
#include <cuda_runtime_api.h>
#endif
#include "trace.h"

namespace chronotope {
namespace {

int64_t mono_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

// ---- flat JSON request parsing (keys with integer, floating or string values; no nesting needed) ----------------
bool jkey(const std::string& s, const char* key, size_t* pos) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return false;
    p = s.find(':', p + k.size());
    if (p == std::string::npos) return false;
    *pos = p + 1;
    return true;
}
int64_t jint(const std::string& s, const char* key, int64_t dflt = 0) {
    size_t p; if (!jkey(s, key, &p)) return dflt;
    return std::strtoll(s.c_str() + p, nullptr, 10);
}
double jdouble(const std::string& s, const char* key, double dflt = 0) {
    size_t p; if (!jkey(s, key, &p)) return dflt;
    return std::strtod(s.c_str() + p, nullptr);
}
std::string jstr(const std::string& s, const char* key, const std::string& dflt = "") {
    size_t p; if (!jkey(s, key, &p)) return dflt;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= s.size() || s[p] != '"') return dflt;
    size_t e = s.find('"', p + 1);
    return e == std::string::npos ? dflt : s.substr(p + 1, e - p - 1);
}

// ---- I/O ------------------------------------------------------------------------------------------------------------
bool read_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n) { ssize_t r = ::read(fd, p, n); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return false; } p += r; n -= (size_t)r; }
    return true;
}
bool write_all(int fd, const void* buf, size_t n) {
    auto* p = static_cast<const uint8_t*>(buf);
    while (n) { ssize_t w = ::write(fd, p, n); if (w <= 0) { if (w < 0 && errno == EINTR) continue; return false; } p += w; n -= (size_t)w; }
    return true;
}
bool read_line(int fd, std::string* line) {
    line->clear(); char c;
    for (;;) { ssize_t r = ::read(fd, &c, 1); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return !line->empty(); } if (c == '\n') return true; line->push_back(c); }
}
bool reply(int fd, const Json& j) { std::string s = j.str(); s.push_back('\n'); return write_all(fd, s.data(), s.size()); }
Json err(const char* what, const char* msg) { return Json().b("ok", false).s("op", what).s("error", msg ? msg : ""); }

// ---- struct serializers (field names = the C structs = the ctypes structs) ----------------------------------------
Json json_admission(const chronotope_admission& a, int rc) {
    return Json().b("ok", true).i("rc", rc).b("stale", rc == CHRONOTOPE_STALE).u("status", a.status).u("relevance", a.relevance)
        .u("control_state_version", a.control_state_version).u("observation_version", a.observation_version).u("payload_hash", a.payload_hash)
        .u("controller_state_hash", a.controller_state_hash).u("pending_actions_hash", a.pending_actions_hash).d("change_mean", a.change_mean)
        .raw("box", "[" + std::to_string(a.box[0]) + "," + std::to_string(a.box[1]) + "," + std::to_string(a.box[2]) + "," + std::to_string(a.box[3]) + "]")
        .u("box_present", a.box_present).i("tracked_x", a.tracked_x).i("tracked_y", a.tracked_y).d("tracked_score", a.tracked_score)
        .u("tracked_valid", a.tracked_valid).u("pending_active", a.pending_active).u("pending_event", a.pending_event)
        .i("admit_device_ns", a.admit_device_ns).i("admit_host_ns", a.admit_host_ns);
}
Json json_proposal(const chronotope_proposal& p, int rc) {
    return Json().b("ok", true).i("rc", rc).u("status", p.status).u("kind", p.kind).i("x", p.x).i("y", p.y).i("value", p.value)
        .u("observation_id", p.observation_id).u("decision_id", p.decision_id).u("supporting_control_state_version", p.supporting_control_state_version)
        .i("b_device_ns", p.b_device_ns).i("b_host_ns", p.b_host_ns);
}
Json json_commit(const chronotope_commit_result& r, int rc) {
    return Json().b("ok", true).i("rc", rc).u("outcome", r.outcome).u("kind", r.kind).i("x", r.x).i("y", r.y).i("value", r.value)
        .i("proposed_x", r.proposed_x).i("proposed_y", r.proposed_y).u("action_id", r.action_id).u("decision_id", r.decision_id)
        .u("serialized_action_hash", r.serialized_action_hash).u("supporting_control_state_version", r.supporting_control_state_version)
        .u("control_state_version_at_commit", r.control_state_version_at_commit).u("observation_id_at_d", r.observation_id_at_d)
        .u("proposal_observation_id", r.proposal_observation_id).i("d_device_ns", r.d_device_ns).i("d_device_mapped_ns", r.d_device_mapped_ns)
        .i("d_host_ns", r.d_host_ns).d("match_score", r.match_score).d("second_best", r.second_best).d("moved_px", r.moved_px)
        .i("plateau", r.plateau).u("matched", r.matched).u("ambiguous", r.ambiguous).u("bind_valid", r.bind_valid).u("binding", r.binding)
        .i("track_ns", r.track_ns).i("track_device_ns", r.track_device_ns).u("reject_reason", r.reject_reason).u("repetition_count", r.repetition_count)
        .i("bind_x", r.bind_x).i("bind_y", r.bind_y).u("pending_created", r.pending_created).i("evidence_age_ns", r.evidence_age_ns)
        .i("evidence_bracket_ns", r.evidence_bracket_ns).u("ledger_row", r.ledger_row);
}
Json json_audit(const chronotope_audit_report& a) {
    return Json().b("ok", true).u("h2d_publications", a.h2d_publications).u("h2d_proposals", a.h2d_proposals).u("d2h_log_in_window", a.d2h_log_in_window)
        .u("d2h_state_in_window", a.d2h_state_in_window).u("host_syncs_log_in_window", a.host_syncs_log_in_window)
        .u("host_syncs_action_in_window", a.host_syncs_action_in_window).u("declared_commit_waits", a.declared_commit_waits)
        .u("host_action_computations", a.host_action_computations).u("mallocs_after_warmup", a.mallocs_after_warmup)
        .u("kernel_launches_in_window", a.kernel_launches_in_window).u("windows_opened", a.windows_opened).u("window_open", a.window_open)
        .u("residency_ok", a.residency_ok).i("clock_offset_ns", a.clock_offset_ns).i("clock_calibration_latency_ns", a.clock_calibration_latency_ns)
        .u("device_bytes_allocated", a.device_bytes_allocated).u("planner_host_steps", a.planner_host_steps).u("planner_d2h_pixels", a.planner_d2h_pixels)
        .u("planner_requests", a.planner_requests).u("binder_host_steps", a.binder_host_steps);
}
std::string json_ledger_rows(const std::vector<chronotope_ledger_row>& rows) {
    std::string s = "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        s += Json().u("observation_id", r.observation_id).i("o_ns", r.o_ns).i("o_device_ns", r.o_device_ns).i("source_time_ns", r.source_time_ns)
                 .i("e_bracket_lo_ns", r.e_bracket_lo_ns).u("frames_since_prev", r.frames_since_prev).u("relevance", r.relevance)
                 .u("control_state_version", r.control_state_version).u("decision_proposed", r.decision_proposed).u("decision_bound", r.decision_bound)
                 .i("d_device_ns", r.d_device_ns).i("effect_observed_ns", r.effect_observed_ns).i("receipt_observed_ns", r.receipt_observed_ns).str();
        if (i + 1 < rows.size()) s += ",";
    }
    return s + "]";
}
std::string json_trace_rows(const std::vector<chronotope_trace_entry>& rows) {
    std::string s = "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        s += Json().u("observation_id", r.observation_id).u("decision_id", r.decision_id).u("kind", r.kind).i("x", r.x).i("y", r.y)
                 .u("accepted", r.accepted).u("reason", r.reason).d("screen_change", r.screen_change).u("repetition_count", r.repetition_count)
                 .i("device_ns", r.device_ns).i("device_mapped_ns", r.device_mapped_ns).u("control_state_version", r.control_state_version).str();
        if (i + 1 < rows.size()) s += ",";
    }
    return s + "]";
}

}  // namespace

int run_server(chronotope_core* core, const std::string& path, TrtPlanner* trt, const ServeOptions& opt, ApiPlanner* api) {
    ::unlink(path.c_str());
    int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { std::perror("socket"); return 1; }
    sockaddr_un addr{}; addr.sun_family = AF_UNIX; std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || ::listen(srv, 1) < 0) { std::perror("bind/listen"); return 1; }
    std::printf("chronotope serve: %s (planner %s)\n", path.c_str(), api ? "hosted API; CPU image staging" : trt ? "trt in-process" : "detector"); std::fflush(stdout);
    int fd = ::accept(srv, nullptr, nullptr);
    if (fd < 0) { std::perror("accept"); return 1; }
    std::string line; std::vector<uint8_t> payload, payload2;
    uint64_t trt_requests = 0;
    for (;;) {
        if (!read_line(fd, &line)) break;
        std::string op = jstr(line, "op");
        if (op == "hello") { reply(fd, Json().b("ok", true).s("version", chronotope_version()).b("trt", trt != nullptr).i("width", opt.width).i("height", opt.height)); continue; }
        if (op == "close") { reply(fd, Json().b("ok", true)); break; }
        if (op == "admit") {
            size_t n = (size_t)jint(line, "len"); payload.resize(n);
            if (n && !read_exact(fd, payload.data(), n)) break;
            chronotope_admission a{};
            int rc = chronotope_admit(core, payload.data(), n, (uint64_t)jint(line, "observation_id"), (uint64_t)jint(line, "source_event_id"),
                                jint(line, "source_time_ns"), jint(line, "publication_time_ns"), &a);
            if (rc != CHRONOTOPE_OK && rc != CHRONOTOPE_STALE) { reply(fd, err("admit", chronotope_last_error(core))); continue; }
            reply(fd, json_admission(a, rc)); continue;
        }
        if (op == "begin_decision") { chronotope_proposal p{}; int rc = chronotope_begin_decision(core, (uint64_t)jint(line, "observation_id"), &p); if (rc != CHRONOTOPE_OK) { reply(fd, err("begin_decision", chronotope_last_error(core))); continue; } reply(fd, json_proposal(p, rc)); continue; }
        if (op == "decide_detector") { chronotope_proposal p{}; int rc = chronotope_decide_detector(core, &p); if (rc != CHRONOTOPE_OK) { reply(fd, err("decide_detector", chronotope_last_error(core))); continue; } reply(fd, json_proposal(p, rc)); continue; }
        if (op == "propose") { chronotope_proposal p{}; int rc = chronotope_propose(core, (uint32_t)jint(line, "kind"), (int32_t)jint(line, "x"), (int32_t)jint(line, "y"), (int32_t)jint(line, "value"), &p); if (rc != CHRONOTOPE_OK) { reply(fd, err("propose", chronotope_last_error(core))); continue; } reply(fd, json_proposal(p, rc)); continue; }
        if (op == "decide_api") {
#ifdef CHRONOTOPE_HOSTED_API
            if (!api) { reply(fd, err("decide_api", "no hosted planner")); continue; }
            const auto start = mono_ns();
            const uint8_t* device = nullptr; uint32_t w = 0, h = 0;
            if (chronotope_frame_b_device(core, 1, &device, &w, &h) != CHRONOTOPE_OK) {
                reply(fd, err("decide_api", chronotope_last_error(core))); continue;
            }
            std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
            if (cudaMemcpy(rgb.data(), device, rgb.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
                reply(fd, err("decide_api", "GPU-to-CPU image staging failed")); continue;
            }
            const auto ans = api->invoke(rgb);
            chronotope_note_planner(core, 1, static_cast<uint64_t>(w) * h);
            if (!ans.ok) { chronotope_abort_decision(core); reply(fd, err("decide_api", ans.error.c_str())); continue; }
            auto j = nlohmann::json{{"ok", true}, {"answer_ok", true}, {"click", ans.click},
                {"x", ans.x}, {"y", ans.y}, {"image_x", ans.image_x}, {"image_y", ans.image_y},
                {"raw_text", ans.raw}, {"invocation_id", ans.request_id}, {"model_reported", ans.model_reported},
                {"input_tokens", ans.input_tokens}, {"output_tokens", ans.output_tokens},
                {"t_start_ns", start}, {"t_end_ns", mono_ns()},
                {"deliberation_ms", (mono_ns() - start) / 1e6}, {"planner_host_steps", 1},
                {"planner_d2h_pixels", static_cast<uint64_t>(w) * h}, {"proposal", nullptr}};
            if (ans.click) {
                chronotope_proposal p{}; int rc = chronotope_propose(core, CHRONOTOPE_ACT_CLICK, ans.x, ans.y, 0, &p);
                j["propose_rc"] = rc; j["proposal"] = nlohmann::json::parse(json_proposal(p, rc).str());
            } else chronotope_abort_decision(core);
            const auto line = j.dump() + "\n";
            write_all(fd, line.data(), line.size()); continue;
#else
            reply(fd, err("decide_api", "build with CHRONOTOPE_HOSTED_API=ON")); continue;
#endif
        }
        if (op == "decide_trt") {
            // the model, in this process, on the frame at B (device pointer; no host copy of the pixels); blocks until the answer
            if (!trt) { reply(fd, err("decide_trt", "no trt planner in this process")); continue; }
            const uint8_t* dptr = nullptr; uint32_t pw = 0, ph = 0;
            if (chronotope_frame_b_device(core, (uint32_t)opt.image_scale, &dptr, &pw, &ph) != CHRONOTOPE_OK) { reply(fd, err("decide_trt", chronotope_last_error(core))); continue; }
            int64_t t0 = mono_ns();
            trt->ask(DeviceFrame{dptr, (int)pw, (int)ph});
            while (!trt->done()) { timespec ts{0, 200000}; nanosleep(&ts, nullptr); }
            VlmAnswer ans = trt->take();
            ++trt_requests;
            chronotope_note_planner(core, trt->host_steps(), 0);
            Json j = Json().b("ok", true).b("answer_ok", ans.ok).b("click", ans.click).i("x", ans.x).i("y", ans.y).i("image_x", ans.image_x).i("image_y", ans.image_y)
                         .s("raw_text", ans.raw.substr(0, 500)).s("error", ans.error).s("invocation_id", ans.request_id).s("model_reported", "tensorrt_edgellm")
                         .i("input_tokens", ans.input_tokens).i("output_tokens", ans.output_tokens).i("t_start_ns", ans.t_start_ns).i("t_end_ns", ans.t_end_ns)
                         .d("deliberation_ms", (ans.t_end_ns - ans.t_start_ns) / 1e6).d("prefill_ms", trt->prefill_ms()).d("decode_ms", trt->decode_ms())
                         .u("planner_host_steps", trt->host_steps()).u("planner_requests", trt_requests).i("wall_ns", mono_ns() - t0);
            if (ans.ok && ans.click) {
                chronotope_proposal p{}; int rc = chronotope_propose(core, CHRONOTOPE_ACT_CLICK, ans.x, ans.y, 0, &p);
                j.i("propose_rc", rc).raw("proposal", json_proposal(p, rc).str());
            } else {
                chronotope_abort_decision(core); j.raw("proposal", "null");
            }
            reply(fd, j); continue;
        }
        if (op == "abort_decision") { int rc = chronotope_abort_decision(core); reply(fd, Json().b("ok", rc == CHRONOTOPE_OK).i("rc", rc).s("error", rc == CHRONOTOPE_OK ? "" : chronotope_last_error(core))); continue; }
        if (op == "restart") { int rc = chronotope_restart(core); reply(fd, Json().b("ok", rc == CHRONOTOPE_OK).i("rc", rc).s("error", rc == CHRONOTOPE_OK ? "" : chronotope_last_error(core))); continue; }
        if (op == "decision_open") { reply(fd, Json().b("ok", true).b("open", chronotope_decision_open(core) != 0)); continue; }
        if (op == "resolve_and_commit" || op == "resolve" || op == "commit_only" || op == "test_recommit") {
            chronotope_commit_result r{}; int rc;
            if (op == "resolve_and_commit") rc = chronotope_resolve_and_commit(core, &r);
            else if (op == "resolve") rc = chronotope_resolve(core, &r);
            else if (op == "commit_only") rc = chronotope_commit_only(core, &r);
            else rc = chronotope_test_recommit(core, &r);
            if (rc != CHRONOTOPE_OK && rc != CHRONOTOPE_NO_DECISION) { reply(fd, err(op.c_str(), chronotope_last_error(core))); continue; }
            reply(fd, json_commit(r, rc)); continue;
        }
        if (op == "read_frame") {
            size_t n = (size_t)opt.width * opt.height * 3; payload.resize(n);
            int rc = chronotope_read_frame(core, payload.data(), n);
            if (rc != CHRONOTOPE_OK) { reply(fd, err("read_frame", chronotope_last_error(core))); continue; }
            reply(fd, Json().b("ok", true).u("len", n)); if (!write_all(fd, payload.data(), n)) break; continue;
        }
        if (op == "override_binding") { int rc = chronotope_override_binding(core, (int32_t)jint(line, "x"), (int32_t)jint(line, "y")); reply(fd, Json().b("ok", rc == CHRONOTOPE_OK).i("rc", rc).s("error", rc == CHRONOTOPE_OK ? "" : chronotope_last_error(core))); continue; }
        if (op == "bind_external") { int rc = chronotope_bind_external(core, (int32_t)jint(line, "x"), (int32_t)jint(line, "y"), (uint32_t)jint(line, "binding")); reply(fd, Json().b("ok", rc == CHRONOTOPE_OK).i("rc", rc).s("error", rc == CHRONOTOPE_OK ? "" : chronotope_last_error(core))); continue; }
        if (op == "observe_receipt") { int rc = chronotope_observe_receipt(core, (uint64_t)jint(line, "decision_id"), jint(line, "observation_publication_ns")); reply(fd, Json().b("ok", rc == CHRONOTOPE_OK).i("rc", rc).s("error", rc == CHRONOTOPE_OK ? "" : chronotope_last_error(core))); continue; }
        if (op == "audit") { chronotope_audit_report a{}; int rc = chronotope_audit(core, &a); if (rc != CHRONOTOPE_OK) { reply(fd, err("audit", chronotope_last_error(core))); continue; } reply(fd, json_audit(a)); continue; }
        if (op == "state_hash") { uint64_t h = 0, v = 0; int rc = chronotope_state_hash(core, &h, &v); if (rc != CHRONOTOPE_OK) { reply(fd, err("state_hash", chronotope_last_error(core))); continue; } reply(fd, Json().b("ok", true).u("hash", h).u("version", v)); continue; }
        if (op == "ledger") { uint32_t n = (uint32_t)jint(line, "n", 256), cnt = 0; std::vector<chronotope_ledger_row> rows(n); int rc = chronotope_ledger(core, rows.data(), n, &cnt); if (rc != CHRONOTOPE_OK) { reply(fd, err("ledger", chronotope_last_error(core))); continue; } rows.resize(cnt); reply(fd, Json().b("ok", true).raw("rows", json_ledger_rows(rows))); continue; }
        if (op == "trace") { uint32_t n = (uint32_t)jint(line, "n", 256), cnt = 0; std::vector<chronotope_trace_entry> rows(n); int rc = chronotope_trace(core, rows.data(), n, &cnt); if (rc != CHRONOTOPE_OK) { reply(fd, err("trace", chronotope_last_error(core))); continue; } rows.resize(cnt); reply(fd, Json().b("ok", true).raw("rows", json_trace_rows(rows))); continue; }
        if (op == "track") {
            size_t n = (size_t)jint(line, "len"); payload.resize(n); payload2.resize(n);
            if (n && (!read_exact(fd, payload.data(), n) || !read_exact(fd, payload2.data(), n))) break;
            chronotope_match m{};
            int rc = chronotope_track(core, payload.data(), payload2.data(), n, (int32_t)jint(line, "x"), (int32_t)jint(line, "y"), (uint32_t)jint(line, "radius", 12), (uint8_t)jint(line, "rgb", 0), &m);
            if (rc != CHRONOTOPE_OK) { reply(fd, err("track", chronotope_last_error(core))); continue; }
            reply(fd, Json().b("ok", true).i("x", m.x).i("y", m.y).d("score", m.score).d("second", m.second).i("plateau", m.plateau).u("matched", m.matched)
                          .u("ambiguous", m.ambiguous).u("flat", m.flat).d("moved_px", m.moved_px));
            continue;
        }
        reply(fd, err(op.c_str(), "unknown op"));
    }
    ::close(fd); ::close(srv); ::unlink(path.c_str());
    return 0;
}

}  // namespace chronotope
