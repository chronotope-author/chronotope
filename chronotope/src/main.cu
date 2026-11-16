// main.cu — the kernelized GUI-using agent (design §10): frames come in, GPU state changes, actions come out.
//
//   chronotope --source synthetic:v=8 --planner detector --deliberation-ms 5200 --policy late_bind --hid fake --trace out.jsonl
//   chronotope --source v4l2:/dev/video0 --pixfmt NV12 --width 1920 --height 1080 --planner vlm --model Qwen/Qwen3.8-27B \
//        --base-url http://127.0.0.1:8000/v1 --prompt-version 5 --hid serial --calib 2,2,40,600 --trace out.jsonl
//
// The loop below is the whole agent: admit the newest frame (perceive, update, verify on the device); if no
// decision is open and the planner has something to say, open one at B; when the proposal is complete, resolve
// and commit at D on the device; hand the committed action to the HID; record everything after the fact.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "capture.h"
#include "chronotope_api.h"
#include "kmbox.h"
#include "serve.h"
#include "planner_trt.h"
#include "planner_vlm.h"
#ifdef CHRONOTOPE_HOSTED_API
#include "planner_api.h"
#endif
#include "synthetic.h"
#include "trace.h"

using namespace chronotope;

namespace {

int64_t now_ns() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }

struct Args {
    std::string source = "synthetic:v=8", pixfmt = "NV12", planner = "detector", policy = "late_bind", hid = "fake", trace = "";
    std::string base_url = "http://127.0.0.1:8000/v1", model = "", target_rgb = "220,38,38", calib = "1,1,0,0";
    std::string engine_dir = "", multimodal_engine_dir = "";   // --planner trt (TensorRT Edge-LLM, in-process)
    std::string agents3_prompt_path, agents3_prompt_audit_path;
    std::string api_provider, api_key_env, api_audit, api_effort = "low";
    std::string serve = "";                                   // K4b: --serve <unix socket>: the referee drives this process
    int detector_at_d = 0;                                      // K3: 0 off, 1 detector-first, 2 NCC-first
    int pending_radius = 0; double pending_timeout_ms = 0, receipt_timeout_ms = 0; bool ack_aware = false;   // the memory rules (P1/P2)
    std::string binder = "ncc", grounder_engine_dir = "", grounder_multimodal_engine_dir = "";   // P2: ncc | oracle | grounder
    double max_evidence_age_ms = 0;
    bool cuda_graph = true;
    int width = 320, height = 240, fps = 60, prompt_version = 5, image_scale = 2, max_tokens = 64;
    int radius = 12, restart_threshold = 6, max_actions = 1, screen_w = 1920, screen_h = 1080;
    double min_score = 0.8, deliberation_ms = 5200, delivery_ms = 20, duration_s = 20, temperature = 0.0;
    bool no_thinking = true, validate_kind = false;
    uint64_t seed = 1;
};

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--source") a.source = next(); else if (k == "--pixfmt") a.pixfmt = next(); else if (k == "--planner") a.planner = next();
        else if (k == "--policy") a.policy = next(); else if (k == "--hid") a.hid = next(); else if (k == "--trace") a.trace = next();
        else if (k == "--base-url") a.base_url = next(); else if (k == "--model") a.model = next(); else if (k == "--target-rgb") a.target_rgb = next();
        else if (k == "--calib") a.calib = next(); else if (k == "--width") a.width = std::atoi(next().c_str()); else if (k == "--height") a.height = std::atoi(next().c_str());
        else if (k == "--fps") a.fps = std::atoi(next().c_str()); else if (k == "--prompt-version") a.prompt_version = std::atoi(next().c_str());
        else if (k == "--image-scale") a.image_scale = std::atoi(next().c_str()); else if (k == "--max-tokens") a.max_tokens = std::atoi(next().c_str());
        else if (k == "--radius") a.radius = std::atoi(next().c_str()); else if (k == "--restart-threshold") a.restart_threshold = std::atoi(next().c_str());
        else if (k == "--max-actions") a.max_actions = std::atoi(next().c_str()); else if (k == "--screen-w") a.screen_w = std::atoi(next().c_str());
        else if (k == "--screen-h") a.screen_h = std::atoi(next().c_str()); else if (k == "--min-score") a.min_score = std::atof(next().c_str());
        else if (k == "--deliberation-ms") a.deliberation_ms = std::atof(next().c_str()); else if (k == "--delivery-ms") a.delivery_ms = std::atof(next().c_str());
        else if (k == "--duration-s") a.duration_s = std::atof(next().c_str()); else if (k == "--temperature") a.temperature = std::atof(next().c_str());
        else if (k == "--thinking") a.no_thinking = false; else if (k == "--no-thinking") a.no_thinking = true;
        else if (k == "--validate-kind") a.validate_kind = true; else if (k == "--seed") a.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (k == "--engine-dir") a.engine_dir = next(); else if (k == "--multimodal-engine-dir") a.multimodal_engine_dir = next();
        else if (k == "--agents3-prompt") a.agents3_prompt_path = next();
        else if (k == "--agents3-prompt-audit") a.agents3_prompt_audit_path = next();
        else if (k == "--api-provider") a.api_provider = next();
        else if (k == "--api-key-env") a.api_key_env = next();
        else if (k == "--api-audit") a.api_audit = next();
        else if (k == "--api-effort") a.api_effort = next();
        else if (k == "--serve") a.serve = next(); else if (k == "--detector-at-d") a.detector_at_d = std::atoi(next().c_str());
        else if (k == "--pending-radius") a.pending_radius = std::atoi(next().c_str()); else if (k == "--pending-timeout-ms") a.pending_timeout_ms = std::atof(next().c_str());
        else if (k == "--receipt-timeout-ms") a.receipt_timeout_ms = std::atof(next().c_str()); else if (k == "--ack-aware") a.ack_aware = true;
        else if (k == "--no-cuda-graph") a.cuda_graph = false;
        else if (k == "--binder") a.binder = next(); else if (k == "--grounder-engine-dir") a.grounder_engine_dir = next();
        else if (k == "--grounder-multimodal-engine-dir") a.grounder_multimodal_engine_dir = next();
        else if (k == "--max-evidence-age-ms") a.max_evidence_age_ms = std::atof(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); std::exit(2); }
    }
    return a;
}

// "k=v,k=v" options inside --source synthetic:...
double opt(const std::string& s, const char* key, double dflt) {
    std::string k = std::string(key) + "=";
    size_t p = s.find(k);
    if (p == std::string::npos) return dflt;
    return std::atof(s.c_str() + p + k.size());
}

struct Source {
    virtual ~Source() = default;
    virtual bool next(std::vector<uint8_t>& payload, int64_t* source_ns, int64_t* publication_ns, uint64_t* event_id) = 0;
    virtual std::vector<uint8_t> rgb_of(const std::vector<uint8_t>& payload) = 0;   // RGB24 for the VLM planner
};

struct SyntheticSource : Source {
    Synthetic s;
    int64_t t0 = 0, period = 16666667;
    int64_t k = 0;
    bool next(std::vector<uint8_t>& payload, int64_t* source_ns, int64_t* publication_ns, uint64_t* event_id) override {
        int64_t due = t0 + k * period;
        int64_t now = now_ns();
        if (due > now) std::this_thread::sleep_for(std::chrono::nanoseconds(due - now));
        int64_t task = now_ns() - t0;
        s.render(task, payload);
        *source_ns = task; *publication_ns = now_ns(); *event_id = s.visible(task) ? 1 : 0;
        ++k;
        return true;
    }
    std::vector<uint8_t> rgb_of(const std::vector<uint8_t>& payload) override { return payload; }
};

struct V4l2Source : Source {
    Capture* cap = nullptr;
    int w = 0, h = 0; uint32_t pix = 0;
    ~V4l2Source() override { if (cap) capture_close(cap); }
    bool next(std::vector<uint8_t>& payload, int64_t* source_ns, int64_t* publication_ns, uint64_t* event_id) override {
        FrameView f;
        if (!capture_latest(cap, &f, 3.0)) { std::fprintf(stderr, "capture: %s\n", capture_error(cap)); return false; }
        payload.assign(f.data, f.data + f.len);
        *source_ns = f.ts_kernel_ns; *publication_ns = f.t_dq_ns; *event_id = 0;
        capture_release(cap, f);
        return true;
    }
    std::vector<uint8_t> rgb_of(const std::vector<uint8_t>& p) override {
        std::vector<uint8_t> rgb((size_t)w * h * 3);
        if (pix == CHRONOTOPE_PIX_RGB24) return p;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t i = (size_t)y * w + x;
                if (pix == CHRONOTOPE_PIX_GREY8) { rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = p[i]; continue; }
                float yy = (float)p[i] - 16.0f;
                size_t uv = (size_t)w * h + (size_t)(y / 2) * w + (size_t)(x / 2) * 2;
                float d = (float)p[uv] - 128.0f, e = (float)p[uv + 1] - 128.0f;
                auto cl = [](float v) { return (uint8_t)std::min(255.0f, std::max(0.0f, v)); };
                rgb[3 * i] = cl(1.164f * yy + 1.596f * e); rgb[3 * i + 1] = cl(1.164f * yy - 0.392f * d - 0.813f * e); rgb[3 * i + 2] = cl(1.164f * yy + 2.017f * d);
            }
        return rgb;
    }
};

uint64_t fnv(const std::vector<uint8_t>& b) { uint64_t h = 14695981039346656037ULL; for (uint8_t c : b) { h ^= c; h *= 1099511628211ULL; } return h; }

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    // ---- the environment's side: frames
    std::unique_ptr<Source> source;
    Synthetic* synth = nullptr;
    uint32_t pixfmt = CHRONOTOPE_PIX_RGB24;
    if (starts(a.source, "synthetic")) {
        auto* s = new SyntheticSource();
        s->s.w = a.width; s->s.h = a.height;
        s->s.vx = (int64_t)opt(a.source, "v", 8); s->s.vy = (int64_t)opt(a.source, "vy", 0);
        s->s.size = (int)opt(a.source, "size", 24); s->s.plain = opt(a.source, "plain", 0) > 0;
        s->s.t_e_ns = (int64_t)(opt(a.source, "t_e", 0.5) * 1e9); s->s.window_ns = (int64_t)(opt(a.source, "window", 16) * 1e9);
        s->s.place_for_seed(a.seed);
        if (opt(a.source, "x0", -1) >= 0) s->s.x0 = (int)opt(a.source, "x0", 0);   // an explicit start (keep a fast target on the canvas)
        if (opt(a.source, "y0", -1) >= 0) s->s.y0 = (int)opt(a.source, "y0", 0);
        s->period = 1000000000LL / std::max(1, a.fps);
        s->t0 = now_ns();
        synth = &s->s;
        source.reset(s);
    } else if (starts(a.source, "v4l2:")) {
        auto* s = new V4l2Source();
        char err[256];
        std::string fourcc = a.pixfmt == "NV12" ? "NV12" : a.pixfmt == "RGB3" ? "RGB3" : "GREY";
        s->cap = capture_open(a.source.c_str() + 5, a.width, a.height, fourcc.c_str(), a.fps, 4, err, sizeof err);
        if (!s->cap) { std::fprintf(stderr, "%s\n", err); return 1; }
        s->w = a.width; s->h = a.height;
        pixfmt = fourcc == "NV12" ? CHRONOTOPE_PIX_NV12 : fourcc == "RGB3" ? CHRONOTOPE_PIX_RGB24 : CHRONOTOPE_PIX_GREY8;
        s->pix = pixfmt;
        source.reset(s);
    } else { std::fprintf(stderr, "--source synthetic:v=8 | v4l2:/dev/video0\n"); return 2; }

    // ---- the core
    chronotope_config cfg{};
    cfg.width = a.width; cfg.height = a.height; cfg.pixfmt = pixfmt;
    cfg.template_radius_px = a.radius; cfg.min_match_score = a.min_score;
    cfg.policy = a.policy == "frozen" ? CHRONOTOPE_POL_FROZEN : a.policy == "late_bind" ? CHRONOTOPE_POL_LATE_BIND
               : a.policy == "restart_if_moved" ? CHRONOTOPE_POL_RESTART_IF_MOVED : CHRONOTOPE_POL_LATE_BIND_OR_RESTART;
    cfg.restart_threshold_px = cfg.policy == CHRONOTOPE_POL_RESTART_IF_MOVED ? a.restart_threshold : -1;
    { unsigned r, g, b; if (std::sscanf(a.target_rgb.c_str(), "%u,%u,%u", &r, &g, &b) == 3) { cfg.target_rgb[0] = r; cfg.target_rgb[1] = g; cfg.target_rgb[2] = b; cfg.has_target_color = 1; } }
    cfg.movement_threshold_px = 6; cfg.validate_kind = a.validate_kind; cfg.track_every_frame = 1;
    cfg.planner_image_scale = (uint32_t)std::max(1, a.image_scale);
    cfg.max_evidence_age_ns = (int64_t)(a.max_evidence_age_ms * 1e6);
    cfg.detector_at_d = (uint32_t)a.detector_at_d;
    cfg.pending_radius_px = (uint32_t)a.pending_radius; cfg.pending_timeout_ns = (int64_t)(a.pending_timeout_ms * 1e6);
    cfg.ack_aware_pending = a.ack_aware ? 1 : 0; cfg.receipt_timeout_ns = (int64_t)(a.receipt_timeout_ms * 1e6);
    char err[512];
    chronotope_core* core = chronotope_create(&cfg, err, sizeof err);
    if (!core) { std::fprintf(stderr, "chronotope_create: %s\n", err); return 1; }

    // ---- K4b: server mode — the referee (the Python gym) drives this process over a UNIX socket; no source, no HID
    if (!a.serve.empty()) {
        std::unique_ptr<TrtPlanner> strt;
        ApiPlanner* api = nullptr;
#ifdef CHRONOTOPE_HOSTED_API
        std::unique_ptr<ApiPlanner> hosted;
        if (a.planner == "api") {
            ApiConfig config;
            config.provider = a.api_provider; config.model = a.model; config.endpoint = a.base_url;
            config.api_key_env = a.api_key_env; config.audit_path = a.api_audit; config.effort = a.api_effort;
            config.max_output_tokens = a.max_tokens; config.prompt_version = a.prompt_version;
            config.w = a.width; config.h = a.height; config.image_scale = a.image_scale;
            try { hosted = std::make_unique<ApiPlanner>(config); api = hosted.get(); }
            catch (std::exception const& e) { std::fprintf(stderr, "hosted planner: %s\n", e.what()); chronotope_destroy(core); return 1; }
        }
#endif
        if (a.planner == "trt") {
            TrtPlannerConfig tc; tc.engine_dir = a.engine_dir; tc.multimodal_engine_dir = a.multimodal_engine_dir; tc.prompt_version = a.prompt_version;
            tc.image_scale = a.image_scale; tc.w = a.width; tc.h = a.height; tc.max_tokens = a.max_tokens; tc.temperature = a.temperature;
            tc.prompt_path = a.agents3_prompt_path; tc.prompt_audit_path = a.agents3_prompt_audit_path;
            tc.enable_thinking = !a.no_thinking; tc.capture_cuda_graph = a.cuda_graph;
#ifdef CHRONOTOPE_EDGELLM_PLUGIN
            if (!std::getenv("EDGELLM_PLUGIN_PATH")) setenv("EDGELLM_PLUGIN_PATH", CHRONOTOPE_EDGELLM_PLUGIN, 1);
#endif
            try { strt = std::make_unique<TrtPlanner>(tc, chronotope_planner_stream(core)); }
            catch (std::exception const& e) { std::fprintf(stderr, "planner trt: %s\n", e.what()); return 1; }
        } else if (a.planner != "detector" && !api) { std::fprintf(stderr, "--serve supports --planner detector | trt | api (opt-in build)\n"); return 2; }
        ServeOptions so; so.width = a.width; so.height = a.height; so.image_scale = a.image_scale;
        int rc = run_server(core, a.serve, strt.get(), so, api);
        strt.reset(); chronotope_destroy(core);
        return rc;
    }

    // ---- the actuator
    Kmbox* hid = nullptr;
    if (a.hid == "fake") hid = kmbox_open_fake();
    else if (starts(a.hid, "serial")) { hid = kmbox_open(a.hid.size() > 7 ? a.hid.c_str() + 7 : nullptr, 115200, err, sizeof err); if (!hid) { std::fprintf(stderr, "kmbox: %s\n", err); return 1; } }
    else if (a.hid != "none") { std::fprintf(stderr, "--hid fake | serial[:/dev/ttyUSBn] | none\n"); return 2; }
    Cursor cursor; cursor.hid = hid; cursor.w = a.screen_w; cursor.h = a.screen_h;
    double sx = 1, sy = 1, ox = 0, oy = 0;
    std::sscanf(a.calib.c_str(), "%lf,%lf,%lf,%lf", &sx, &sy, &ox, &oy);

    // ---- the planner
    std::unique_ptr<VlmPlanner> vlm;
    std::unique_ptr<TrtPlanner> trt;
    if (a.planner == "vlm") {
        VlmConfig vc; vc.base_url = a.base_url; vc.model = a.model; vc.max_tokens = a.max_tokens; vc.temperature = a.temperature;
        vc.enable_thinking = !a.no_thinking; vc.prompt_version = a.prompt_version; vc.image_scale = a.image_scale; vc.w = a.width; vc.h = a.height;
        vlm = std::make_unique<VlmPlanner>(vc);
    } else if (a.planner == "trt") {
        // the slow VLM in-process: TensorRT Edge-LLM on the core's planner stream; the frame at B stays on the device
        TrtPlannerConfig tc; tc.engine_dir = a.engine_dir; tc.multimodal_engine_dir = a.multimodal_engine_dir; tc.prompt_version = a.prompt_version;
        tc.image_scale = a.image_scale; tc.w = a.width; tc.h = a.height; tc.max_tokens = a.max_tokens; tc.temperature = a.temperature;
        tc.prompt_path = a.agents3_prompt_path; tc.prompt_audit_path = a.agents3_prompt_audit_path;
        tc.enable_thinking = !a.no_thinking; tc.capture_cuda_graph = a.cuda_graph;
#ifdef CHRONOTOPE_EDGELLM_PLUGIN
        if (!std::getenv("EDGELLM_PLUGIN_PATH")) setenv("EDGELLM_PLUGIN_PATH", CHRONOTOPE_EDGELLM_PLUGIN, 1);   // TensorRT loads the custom plugins from here
#endif
        try { trt = std::make_unique<TrtPlanner>(tc, chronotope_planner_stream(core)); }
        catch (std::exception const& e) { std::fprintf(stderr, "planner trt: %s\n", e.what()); return 1; }
        std::printf("planner trt: %s\n", trt->runtime_info().c_str());
    } else if (a.planner != "detector") { std::fprintf(stderr, "--planner detector | vlm | trt\n"); return 2; }
    // P2: the binder at D — the resident NCC (default), the environment's oracle (test only), or a small in-process grounder model
    std::unique_ptr<TrtPlanner> grounder;
    if (a.binder == "grounder") {
        TrtPlannerConfig gc; gc.engine_dir = a.grounder_engine_dir; gc.multimodal_engine_dir = a.grounder_multimodal_engine_dir;
        gc.prompt_version = a.prompt_version; gc.image_scale = a.image_scale; gc.w = a.width; gc.h = a.height; gc.max_tokens = a.max_tokens;
        gc.capture_cuda_graph = a.cuda_graph;
#ifdef CHRONOTOPE_EDGELLM_PLUGIN
        if (!std::getenv("EDGELLM_PLUGIN_PATH")) setenv("EDGELLM_PLUGIN_PATH", CHRONOTOPE_EDGELLM_PLUGIN, 1);
#endif
        try { grounder = std::make_unique<TrtPlanner>(gc, chronotope_planner_stream(core)); }
        catch (std::exception const& e) { std::fprintf(stderr, "binder grounder: %s\n", e.what()); return 1; }
    } else if (a.binder == "oracle") {
        if (!synth) { std::fprintf(stderr, "--binder oracle needs the synthetic source (the environment's truth)\n"); return 2; }
    } else if (a.binder != "ncc") { std::fprintf(stderr, "--binder ncc | oracle | grounder\n"); return 2; }

    // ---- the trace
    JsonlTrace trace;
    if (!a.trace.empty() && !trace.open(a.trace)) { std::fprintf(stderr, "trace %s exists or cannot be created\n", a.trace.c_str()); return 1; }
    trace.append("trial_header", Json().s("core", chronotope_version()).s("source", a.source).s("planner", a.planner).s("policy", a.policy).s("hid", a.hid)
                 .i("width", a.width).i("height", a.height).i("radius", a.radius).d("min_score", a.min_score).d("deliberation_ms", a.deliberation_ms)
                 .d("delivery_ms", a.delivery_ms).s("model", a.model).i("prompt_version", a.prompt_version).i("harness_origin_ns", now_ns()));

    // ---- the loop
    int64_t deadline = now_ns() + (int64_t)(a.duration_s * 1e9), delib_ns = (int64_t)(a.deliberation_ms * 1e6);
    uint64_t obs_id = 0, decisions = 0; int actions = 0;
    int64_t t_b = 0; uint64_t baseline_hash = 0; bool have_baseline = false;
    std::vector<uint8_t> payload;
    std::vector<int64_t> bind_device_ns, bind_host_ns;
    int hits = 0, scored = 0;
    while (now_ns() < deadline && actions < a.max_actions) {
        int64_t src_ns, pub_ns; uint64_t ev;
        if (!source->next(payload, &src_ns, &pub_ns, &ev)) break;
        chronotope_admission adm{};
        ++obs_id;
        if (chronotope_admit(core, payload.data(), payload.size(), obs_id, ev, src_ns, pub_ns, &adm) != CHRONOTOPE_OK) { std::fprintf(stderr, "admit: %s\n", chronotope_last_error(core)); break; }
        trace.append("observation", Json().u("observation_id", obs_id).u("source_event_id", ev).i("source_time_ns", src_ns).i("publication_time_ns", pub_ns)
                     .u("payload_hash", adm.payload_hash).u("observation_version", adm.observation_version).u("control_state_version", adm.control_state_version)
                     .i("relevance", adm.relevance).b("box_present", adm.box_present).i("box_x0", adm.box[0]).i("box_y0", adm.box[1]).i("box_x1", adm.box[2]).i("box_y1", adm.box[3])
                     .d("change_mean", adm.change_mean).b("tracked_valid", adm.tracked_valid).i("tracked_x", adm.tracked_x).i("tracked_y", adm.tracked_y));
        if (!chronotope_decision_open(core)) {
            if (vlm || trt) {
                bool busy = vlm ? vlm->in_flight() : trt->in_flight();
                if (!busy && (!have_baseline || adm.payload_hash != baseline_hash)) {   // invoke on frame change, as the paper's controller
                    chronotope_proposal p{};
                    chronotope_begin_decision(core, obs_id, &p);
                    ++decisions; t_b = now_ns(); baseline_hash = adm.payload_hash; have_baseline = true;
                    if (vlm) vlm->ask(source->rgb_of(payload));
                    else {
                        const uint8_t* dptr = nullptr; uint32_t pw = 0, ph = 0;   // the frame at B, upscaled on the planner stream, never on the host
                        if (chronotope_frame_b_device(core, (uint32_t)a.image_scale, &dptr, &pw, &ph) != CHRONOTOPE_OK) { std::fprintf(stderr, "frame_b_device: %s\n", chronotope_last_error(core)); break; }
                        trt->ask(DeviceFrame{dptr, (int)pw, (int)ph});
                    }
                    trace.append("decision_begin", Json().u("decision_id", decisions).u("observation_id", obs_id).i("begin_time_ns", t_b).i("b_device_ns", p.b_device_ns).s("invocation_kind", "fixed_input").s("planner", vlm ? "vlm_http" : "tensorrt_edgellm"));
                }
            } else if (adm.box_present) {
                chronotope_proposal p{};
                chronotope_begin_decision(core, obs_id, &p);
                ++decisions; t_b = now_ns();
                trace.append("decision_begin", Json().u("decision_id", decisions).u("observation_id", obs_id).i("begin_time_ns", t_b).i("b_device_ns", p.b_device_ns).s("invocation_kind", "resident_detector"));
                chronotope_decide_detector(core, &p);
                trace.append("action_proposed", Json().u("decision_id", decisions).b("proposed", p.status == 1).i("x", p.x).i("y", p.y).u("observation_id", p.observation_id).s("coordinate_status", "provisional"));
            }
            continue;
        }
        // a decision is open
        bool ready = false;
        if (vlm || trt) {
            bool answered = vlm ? vlm->done() : trt->done();
            if (answered) {
                VlmAnswer ans = vlm ? vlm->take() : trt->take();
                Json inv = Json().u("decision_id", decisions).b("ok", ans.ok).b("click", ans.click).s("raw_text", ans.raw).s("error", ans.error)
                             .s("invocation_id", ans.request_id).s("model_reported", ans.model_reported).i("input_tokens", ans.input_tokens).i("output_tokens", ans.output_tokens)
                             .i("end_time_ns", ans.t_end_ns).i("image_x", ans.image_x).i("image_y", ans.image_y).d("deliberation_ms", (ans.t_end_ns - ans.t_start_ns) / 1e6);
                if (trt) {
                    chronotope_note_planner(core, trt->host_steps(), 0);   // the runtime's per-token host syncs; the frame never left the device
                    inv.u("planner_host_steps", trt->host_steps()).d("prefill_ms", trt->prefill_ms()).d("decode_ms", trt->decode_ms()).s("planner", "tensorrt_edgellm");
                }
                trace.append("decision_invocation", inv);
                if (ans.ok && ans.click) {
                    chronotope_proposal p{};
                    chronotope_propose(core, CHRONOTOPE_ACT_CLICK, ans.x, ans.y, 0, &p);
                    trace.append("action_proposed", Json().u("decision_id", decisions).b("proposed", p.status == 1).i("x", p.x).i("y", p.y).u("observation_id", p.observation_id).s("coordinate_status", "provisional"));
                    ready = p.status == 1;
                } else {
                    chronotope_abort_decision(core);
                    trace.append("decision_no_action", Json().u("decision_id", decisions).i("end_time_ns", now_ns()));
                }
            }
        } else if (now_ns() - t_b >= delib_ns) {
            ready = true;   // the emulated deliberation is over
        }
        if (!ready) continue;
        chronotope_commit_result r{};
        int64_t grounder_ms_x1000 = 0;
        if (a.binder == "ncc") {
            chronotope_resolve_and_commit(core, &r);
        } else if (a.binder == "oracle") {
            int64_t task_now = now_ns() - static_cast<SyntheticSource*>(source.get())->t0;
            SynthBox bnow = synth->box_at(task_now);
            chronotope_bind_external(core, bnow.present ? bnow.cx() : 0, bnow.present ? bnow.cy() : 0, CHRONOTOPE_BIND_ORACLE);
            chronotope_commit_only(core, &r);
        } else {   // grounder: the small model re-grounds the named target on the NEWEST frame, handed over as a device tensor
            const uint8_t* dptr = nullptr; uint32_t pw = 0, ph = 0;
            chronotope_frame_newest_device(core, (uint32_t)a.image_scale, &dptr, &pw, &ph);
            int64_t g0 = now_ns();
            VlmAnswer g = grounder->invoke(DeviceFrame{dptr, (int)pw, (int)ph});
            grounder_ms_x1000 = (now_ns() - g0) / 1000;
            chronotope_note_planner(core, grounder->host_steps(), 0);
            if (g.ok && g.click) chronotope_bind_external(core, g.x, g.y, CHRONOTOPE_BIND_GROUNDER);
            else chronotope_bind_external(core, r.proposed_x, r.proposed_y, CHRONOTOPE_BIND_GROUNDER);   // no grounding: the proposal stands
            chronotope_commit_only(core, &r);
            trace.append("grounding", Json().u("decision_id", decisions).b("ok", g.ok).b("click", g.click).i("x", g.x).i("y", g.y).s("raw_text", g.raw).s("error", g.error)
                         .d("grounder_ms", grounder_ms_x1000 / 1000.0).u("host_steps", grounder->host_steps()));
        }
        Json cj = Json().u("decision_id", decisions).i("outcome", r.outcome).i("binding", r.binding).i("x", r.x).i("y", r.y).i("proposed_x", r.proposed_x).i("proposed_y", r.proposed_y)
                       .u("action_id", r.action_id).u("observation_id_at_d", r.observation_id_at_d).u("proposal_observation_id", r.proposal_observation_id)
                       .i("commitment_time_ns", r.d_device_mapped_ns).i("commitment_globaltimer_ns", r.d_device_ns).i("commit_host_ns", r.d_host_ns)
                       .d("match_score", r.match_score).d("second_best", r.second_best).i("plateau", r.plateau).b("matched", r.matched).b("ambiguous", r.ambiguous).d("moved_px", r.moved_px)
                       .i("track_ns", r.track_ns).i("track_device_ns", r.track_device_ns).u("control_state_version_at_commit", r.control_state_version_at_commit)
                       .u("supporting_control_state_version", r.supporting_control_state_version).s("serialized_action_hash", std::to_string(r.serialized_action_hash))
                       .i("evidence_age_ns", r.evidence_age_ns).i("evidence_bracket_ns", r.evidence_bracket_ns).u("ledger_row", r.ledger_row).s("binder", a.binder);
        if (r.outcome == CHRONOTOPE_CS_COMMITTED) {
            trace.append("commit", cj);
            bind_device_ns.push_back(r.track_device_ns); bind_host_ns.push_back(r.track_ns);
            // delivery (after D): the HID moves to the calibrated screen point and clicks
            int64_t t_write = -1;
            int scr_x = (int)std::lround(sx * r.x + ox), scr_y = (int)std::lround(sy * r.y + oy);
            if (hid) { cursor.move_to(scr_x, scr_y); t_write = kmbox_click(hid, "left", 20); }
            trace.append("delivery", Json().u("action_id", r.action_id).i("submit_time_ns", r.d_host_ns).i("serial_write_ns", t_write).i("screen_x", scr_x).i("screen_y", scr_y)
                         .s("route", hid ? (kmbox_is_fake(hid) ? "fake_hid" : "kmbox_serial") : "none"));
            ++actions;
            if (synth && hid && kmbox_is_fake(hid)) {   // the synthetic environment scores the click at its (simulated) receipt
                int64_t t_a = t_write + (int64_t)(a.delivery_ms * 1e6);
                int64_t task_a = t_a - static_cast<SyntheticSource*>(source.get())->t0;
                int fx = (int)std::lround((scr_x - ox) / sx), fy = (int)std::lround((scr_y - oy) / sy);
                SynthBox ba = synth->box_at(task_a);
                bool valid = ba.contains(fx, fy);
                double e = ba.present ? std::hypot((double)(fx - ba.cx()), (double)(fy - ba.cy())) : -1.0;
                trace.append("receipt", Json().u("action_id", r.action_id).i("receipt_time_ns", t_a).i("task_time_at_receipt_ns", task_a).b("target_visible", ba.present)
                             .b("valid_at_receipt", valid).d("error_px", e).i("click_x", fx).i("click_y", fy).i("target_cx", ba.cx()).i("target_cy", ba.cy()));
                hits += valid; ++scored;
                std::printf("action %d: click (%d,%d) at D+%.0f ms -> target centre (%d,%d): %s, error %.1f px; bind %.3f ms device / %.3f ms host; "
                            "score %.3f second %.3f plateau %d matched %d (proposal %d,%d)\n",
                            actions, fx, fy, a.delivery_ms, ba.cx(), ba.cy(), valid ? "HIT" : "miss", e, r.track_device_ns / 1e6, r.track_ns / 1e6,
                            r.match_score, r.second_best, r.plateau, r.matched, r.proposed_x, r.proposed_y);
            }
        } else {
            trace.append("decision_restarted", cj);
            std::printf("decision %llu: outcome %u (not committed)\n", (unsigned long long)decisions, r.outcome);
        }
    }
    chronotope_audit_report au{};
    chronotope_audit(core, &au);
    trace.append("audit", Json().u("h2d_publications", au.h2d_publications).u("d2h_log_in_window", au.d2h_log_in_window).u("d2h_state_in_window", au.d2h_state_in_window)
                 .u("host_syncs_log_in_window", au.host_syncs_log_in_window).u("host_syncs_action_in_window", au.host_syncs_action_in_window)
                 .u("declared_commit_waits", au.declared_commit_waits).u("host_action_computations", au.host_action_computations)
                 .u("mallocs_after_warmup", au.mallocs_after_warmup).u("windows_opened", au.windows_opened).b("residency_ok", au.residency_ok)
                 .u("planner_host_steps", au.planner_host_steps).u("planner_d2h_pixels", au.planner_d2h_pixels).u("planner_requests", au.planner_requests)
                 .u("binder_host_steps", au.binder_host_steps)
                 .i("clock_calibration_latency_ns", au.clock_calibration_latency_ns).u("device_bytes_allocated", au.device_bytes_allocated));
    {   // the event ledger: the agent's own four-clock rows (a log read after D)
        std::vector<chronotope_ledger_row> rows(256); uint32_t n = 0;
        if (chronotope_ledger(core, rows.data(), 256, &n) == CHRONOTOPE_OK)
            for (uint32_t i = 0; i < n; ++i) {
                const chronotope_ledger_row& l = rows[i];
                trace.append("ledger_row", Json().u("observation_id", l.observation_id).i("o_ns", l.o_ns).i("source_time_ns", l.source_time_ns).i("e_bracket_lo_ns", l.e_bracket_lo_ns)
                             .u("frames_since_prev", l.frames_since_prev).u("relevance", l.relevance).u("control_state_version", l.control_state_version)
                             .u("decision_proposed", l.decision_proposed).u("decision_bound", l.decision_bound).i("d_ns", l.d_device_ns)
                             .i("effect_observed_ns", l.effect_observed_ns).i("receipt_observed_ns", l.receipt_observed_ns));
            }
    }
    trace.append("trial_end", Json().s("reason", actions >= a.max_actions ? "actions" : "duration").i("harness_time_ns", now_ns()).i("observations", (int64_t)obs_id).u("decisions", decisions).i("actions", actions));
    auto med = [](std::vector<int64_t> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2] / 1e6; };
    std::printf("chronotope %s: %llu observations, %llu decisions, %d actions%s; residency %s (d2h state in window %llu, host action computations %llu, mallocs after warm-up %llu); "
                "bind median %.3f ms device / %.3f ms host\n", chronotope_version(), (unsigned long long)obs_id, (unsigned long long)decisions, actions,
                scored ? (std::string(" (") + std::to_string(hits) + "/" + std::to_string(scored) + " hits)").c_str() : "",
                au.residency_ok ? "OK" : "VIOLATED", (unsigned long long)au.d2h_state_in_window, (unsigned long long)au.host_action_computations,
                (unsigned long long)au.mallocs_after_warmup, med(bind_device_ns), med(bind_host_ns));
    if (hid) kmbox_close(hid);
    chronotope_destroy(core);
    return 0;
}
