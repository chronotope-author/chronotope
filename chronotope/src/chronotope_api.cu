// chronotope_api.cu — the C ABI over Core.
#include "chronotope_api.h"
#include "core.h"
#include <cstdio>
#include <cstring>

#include <cstring>
#include <ctime>
#include <memory>
#ifdef CHRONOTOPE_WITH_EDGELLM
#include "planner_trt.h"
#endif

struct chronotope_core {
    chronotope::Core core;
#ifdef CHRONOTOPE_WITH_EDGELLM
    std::unique_ptr<chronotope::TrtPlanner> trt;   // K4d: the in-process planner owned by the handle
    uint32_t trt_scale = 2;
    uint64_t trt_requests = 0;
#endif
    explicit chronotope_core(const chronotope_config& c) : core(c) {}
};

static int64_t chronotope_mono_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

extern "C" {

const char* chronotope_version(void) { return chronotope::CORE_VERSION; }

chronotope_core* chronotope_create(const chronotope_config* cfg, char* err, size_t errlen) {
    if (!cfg) { if (err && errlen) std::snprintf(err, errlen, "null config"); return nullptr; }
    auto* c = new chronotope_core(*cfg);
    if (!c->core.ok()) {
        if (err && errlen) std::snprintf(err, errlen, "%s", c->core.error().c_str());
        delete c;
        return nullptr;
    }
    return c;
}

void chronotope_destroy(chronotope_core* core) { delete core; }

const char* chronotope_last_error(chronotope_core* core) { return core ? core->core.error().c_str() : "null core"; }

int chronotope_admit(chronotope_core* core, const uint8_t* payload, size_t len, uint64_t observation_id, uint64_t source_event_id,
               int64_t source_time_ns, int64_t publication_time_ns, chronotope_admission* out) {
    if (!core || !payload || !out) return CHRONOTOPE_ERR;
    chronotope::ObservationEnvelope env{observation_id, source_event_id, source_time_ns, publication_time_ns, 0, 0};
    return core->core.admit(payload, len, env, out);
}

int chronotope_begin_decision(chronotope_core* core, uint64_t observation_id, chronotope_proposal* out) {
    return core ? core->core.begin_decision(observation_id, out) : CHRONOTOPE_ERR;
}

int chronotope_decide_detector(chronotope_core* core, chronotope_proposal* out) { return core && out ? core->core.decide_detector(out) : CHRONOTOPE_ERR; }

int chronotope_propose(chronotope_core* core, uint32_t kind, int32_t x, int32_t y, int32_t value, chronotope_proposal* out) {
    return core && out ? core->core.propose(kind, x, y, value, out) : CHRONOTOPE_ERR;
}

int chronotope_abort_decision(chronotope_core* core) { return core ? core->core.abort_decision() : CHRONOTOPE_ERR; }

int chronotope_resolve_and_commit(chronotope_core* core, chronotope_commit_result* out) { return core && out ? core->core.resolve(true, false, out) : CHRONOTOPE_ERR; }

int chronotope_resolve(chronotope_core* core, chronotope_commit_result* out) { return core && out ? core->core.resolve(false, false, out) : CHRONOTOPE_ERR; }

int chronotope_commit_only(chronotope_core* core, chronotope_commit_result* out) { return core && out ? core->core.resolve(true, true, out) : CHRONOTOPE_ERR; }

int chronotope_restart(chronotope_core* core) { return core ? core->core.restart() : CHRONOTOPE_ERR; }

int chronotope_decision_open(chronotope_core* core) { return core ? core->core.decision_open() : 0; }

int chronotope_read_frame(chronotope_core* core, uint8_t* buf, size_t len) { return core && buf ? core->core.read_frame(buf, len) : CHRONOTOPE_ERR; }

int chronotope_override_binding(chronotope_core* core, int32_t x, int32_t y) { return core ? core->core.override_binding(x, y) : CHRONOTOPE_ERR; }

int chronotope_audit(chronotope_core* core, chronotope_audit_report* out) {
    if (!core || !out) return CHRONOTOPE_ERR;
    core->core.audit(out);
    return CHRONOTOPE_OK;
}

int chronotope_trace(chronotope_core* core, chronotope_trace_entry* buf, uint32_t n, uint32_t* count) {
    return core && buf && count ? core->core.trace(buf, n, count) : CHRONOTOPE_ERR;
}

int chronotope_state_hash(chronotope_core* core, uint64_t* h, uint64_t* v) { return core && h && v ? core->core.state_hash(h, v) : CHRONOTOPE_ERR; }

int chronotope_track(chronotope_core* core, const uint8_t* frame_b, const uint8_t* frame_d, size_t len, int32_t x, int32_t y,
               uint32_t radius, uint8_t rgb, chronotope_match* out) {
    return core && frame_b && frame_d && out ? core->core.track(frame_b, frame_d, len, x, y, radius, rgb, out) : CHRONOTOPE_ERR;
}

int chronotope_planner_trt_open(chronotope_core* core, const chronotope_trt_config* cfg) {
#ifdef CHRONOTOPE_WITH_EDGELLM
    if (!core || !cfg) return CHRONOTOPE_ERR;
    try {
        chronotope::TrtPlannerConfig tc;
        tc.engine_dir = cfg->engine_dir ? cfg->engine_dir : ""; tc.multimodal_engine_dir = cfg->multimodal_engine_dir ? cfg->multimodal_engine_dir : "";
        tc.prompt_version = cfg->prompt_version; tc.image_scale = cfg->image_scale > 0 ? cfg->image_scale : 2; tc.max_tokens = cfg->max_tokens > 0 ? cfg->max_tokens : 64;
        tc.temperature = cfg->temperature; tc.enable_thinking = cfg->enable_thinking != 0; tc.capture_cuda_graph = cfg->capture_cuda_graph != 0;
        tc.w = (int)core->core.width(); tc.h = (int)core->core.height();
        core->trt_scale = (uint32_t)tc.image_scale;
        core->trt = std::make_unique<chronotope::TrtPlanner>(tc, core->core.planner_stream());
        return CHRONOTOPE_OK;
    } catch (std::exception const& e) { core->core.set_error(std::string("planner trt: ") + e.what()); return CHRONOTOPE_ERR; }
#else
    (void)cfg; if (core) core->core.set_error("planner trt: not compiled (configure with CHRONOTOPE_EDGELLM_BUILD)"); return CHRONOTOPE_ERR;
#endif
}

int chronotope_planner_trt_close(chronotope_core* core) {
#ifdef CHRONOTOPE_WITH_EDGELLM
    if (!core) return CHRONOTOPE_ERR; core->trt.reset(); return CHRONOTOPE_OK;
#else
    return core ? CHRONOTOPE_OK : CHRONOTOPE_ERR;
#endif
}

int chronotope_decide_trt(chronotope_core* core, chronotope_trt_answer* out) {
#ifdef CHRONOTOPE_WITH_EDGELLM
    if (!core || !out) return CHRONOTOPE_ERR;
    std::memset(out, 0, sizeof *out); out->propose_rc = -1;
    if (!core->trt) { core->core.set_error("planner trt: not opened"); return CHRONOTOPE_ERR; }
    const uint8_t* dptr = nullptr; uint32_t pw = 0, ph = 0;
    int rc = chronotope_frame_b_device(core, core->trt_scale, &dptr, &pw, &ph);
    if (rc != CHRONOTOPE_OK) return rc;
    chronotope::VlmAnswer ans = core->trt->invoke(chronotope::DeviceFrame{dptr, (int)pw, (int)ph});   // blocking, this thread
    ++core->trt_requests;
    chronotope_note_planner(core, core->trt->host_steps(), 0);
    out->answer_ok = ans.ok; out->click = ans.click; out->x = ans.x; out->y = ans.y; out->image_x = ans.image_x; out->image_y = ans.image_y;
    out->t_start_ns = ans.t_start_ns; out->t_end_ns = ans.t_end_ns; out->prefill_ms = core->trt->prefill_ms(); out->decode_ms = core->trt->decode_ms();
    out->host_steps = core->trt->host_steps(); out->requests = core->trt_requests; out->input_tokens = ans.input_tokens; out->output_tokens = ans.output_tokens;
    std::snprintf(out->raw_text, sizeof out->raw_text, "%s", ans.raw.c_str()); std::snprintf(out->error, sizeof out->error, "%s", ans.error.c_str());
    if (ans.ok && ans.click) out->propose_rc = chronotope_propose(core, CHRONOTOPE_ACT_CLICK, ans.x, ans.y, 0, &out->proposal);
    else chronotope_abort_decision(core);
    return CHRONOTOPE_OK;
#else
    (void)out; if (core) core->core.set_error("planner trt: not compiled"); return CHRONOTOPE_ERR;
#endif
}

int chronotope_frame_b_device(chronotope_core* core, uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h) {
    return core && ptr && w && h ? core->core.frame_b_device(scale, ptr, w, h) : CHRONOTOPE_ERR;
}

void* chronotope_planner_stream(chronotope_core* core) { return core ? core->core.planner_stream() : nullptr; }

int chronotope_note_planner(chronotope_core* core, uint64_t host_steps, uint64_t d2h_pixels) {
    if (!core) return CHRONOTOPE_ERR;
    core->core.note_planner(host_steps, d2h_pixels);
    return CHRONOTOPE_OK;
}

int chronotope_ledger(chronotope_core* core, chronotope_ledger_row* buf, uint32_t n, uint32_t* count) { return core && buf && count ? core->core.ledger(buf, n, count) : CHRONOTOPE_ERR; }

int chronotope_bind_external(chronotope_core* core, int32_t x, int32_t y, uint32_t binding) { return core ? core->core.bind_external(x, y, binding) : CHRONOTOPE_ERR; }

int chronotope_observe_receipt(chronotope_core* core, uint64_t decision_id, int64_t o_ns) { return core ? core->core.observe_receipt(decision_id, o_ns) : CHRONOTOPE_ERR; }

int chronotope_frame_newest_device(chronotope_core* core, uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h) {
    return core && ptr && w && h ? core->core.frame_newest_device(scale, ptr, w, h) : CHRONOTOPE_ERR;
}

int chronotope_test_recommit(chronotope_core* core, chronotope_commit_result* out) { return core && out ? core->core.test_recommit(out) : CHRONOTOPE_ERR; }

int chronotope_test_alloc(chronotope_core* core) { return core ? core->core.test_alloc() : CHRONOTOPE_ERR; }

}  // extern "C"
