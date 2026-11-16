// core.cu — the host side of the control core. Every function is either a publication (H2D), a
// launch, a declared wait, or a log read; the audit counts each one and whether it fell inside the
// reactive window (B -> D).
#include "core.h"
#include "control.cuh"
#include "vision.cuh"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>

namespace chronotope {

int64_t host_now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

namespace {
constexpr int MAX_PENDING_RADIUS = 64;

template <class T>
T* host_pinned(size_t count) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, sizeof(T) * count, cudaHostAllocDefault) != cudaSuccess) return nullptr;
    std::memset(p, 0, sizeof(T) * count);
    return static_cast<T*>(p);
}
}  // namespace

bool Core::fail(const std::string& what, cudaError_t e) {
    err_ = what;
    if (e != cudaSuccess) err_ += std::string(": ") + cudaGetErrorString(e);
    ok_ = false;
    return false;
}

void* Core::dalloc(size_t bytes) {
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) { fail("cudaMalloc(" + std::to_string(bytes) + ")", e); return nullptr; }
    if (warm_) audit_.mallocs_after_warmup++;
    device_bytes_ += bytes;
    allocs_.push_back(p);
    cudaMemset(p, 0, bytes);
    return p;
}

size_t Core::expected_len() const {
    switch (cfg_.pixfmt) {
        case PIX_RGB24: return (size_t)3 * n_;
        case PIX_GREY8: return (size_t)n_;
        default: return (size_t)n_ * 3 / 2;
    }
}

Core::Core(const chronotope_config& cfg) : pub_(cfg) {
    w_ = (int)cfg.width; h_ = (int)cfg.height;
    if (w_ < 8 || h_ < 8) { fail("width and height must be >= 8"); return; }
    if (cfg.pixfmt > 2) { fail("pixfmt must be RGB24, GREY8 or NV12"); return; }
    if (cfg.pixfmt == CHRONOTOPE_PIX_NV12 && ((w_ % 2) || (h_ % 2))) { fail("NV12 needs even dimensions"); return; }
    if (cfg.template_radius_px < 1) { fail("template_radius_px must be >= 1"); return; }
    side_ = 2 * (int)cfg.template_radius_px + 1;
    if (side_ > w_ || side_ > h_) { fail("template larger than the frame"); return; }
    if (cfg.policy > 3) { fail("policy"); return; }
    if (cfg.policy == CHRONOTOPE_POL_RESTART_IF_MOVED && cfg.restart_threshold_px < 0) { fail("restart_threshold_px is required for restart_if_moved"); return; }
    if (cfg.validate_kind && !cfg.has_target_color) { fail("validate_kind needs the target colour"); return; }
    if (!(cfg.min_match_score >= 0.0 && cfg.min_match_score <= 1.0)) { fail("0 <= min_match_score <= 1"); return; }
    if ((int)cfg.pending_radius_px > MAX_PENDING_RADIUS) { fail("pending_radius_px <= 64"); return; }
    cfg_.w = w_; cfg_.h = h_; cfg_.pixfmt = cfg.pixfmt; cfg_.radius = (int)cfg.template_radius_px;
    cfg_.min_score = cfg.min_match_score; cfg_.policy = cfg.policy; cfg_.restart_px = cfg.restart_threshold_px;
    cfg_.target_r = cfg.target_rgb[0]; cfg_.target_g = cfg.target_rgb[1]; cfg_.target_b = cfg.target_rgb[2];
    cfg_.has_target = cfg.has_target_color; cfg_.offset_px = cfg.coordinate_offset_px;
    cfg_.movement_px = (int)cfg.movement_threshold_px; cfg_.validate_kind = cfg.validate_kind; cfg_.rgb_template = cfg.rgb_template;
    cfg_.pending_radius = (int)cfg.pending_radius_px; cfg_.pending_timeout_ns = cfg.pending_timeout_ns; cfg_.max_repeat = cfg.max_repeat_without_effect;
    cfg_.trace_entries = cfg.trace_entries ? cfg.trace_entries : 256; cfg_.slots = cfg.committed_slots ? cfg.committed_slots : 64;
    cfg_.track_every_frame = cfg.track_every_frame;
    planner_scale_ = cfg.planner_image_scale ? cfg.planner_image_scale : 2;
    cfg_.ledger_rows = cfg.ledger_rows ? cfg.ledger_rows : 256; cfg_.max_evidence_age_ns = cfg.max_evidence_age_ns;
    cfg_.ack_aware_pending = cfg.ack_aware_pending; cfg_.receipt_timeout_ns = cfg.receipt_timeout_ns;
    cfg_.detector_at_d = cfg.detector_at_d;
    chans_ = cfg.rgb_template ? 3 : 1;
    n_ = w_ * h_; Wv_ = w_ - side_ + 1; Hv_ = h_ - side_ + 1; nscores_ = Wv_ * Hv_;
    nblocks_ = vision::argmax_blocks(nscores_);
    payload_bytes_ = (size_t)3 * n_; plane_bytes_ = (size_t)n_ * sizeof(float);
    ok_ = true;

    cudaError_t e;
    for (cudaStream_t* s : {&s_vision_, &s_control_, &s_planner_, &s_memory_})
        if ((e = cudaStreamCreateWithFlags(s, cudaStreamNonBlocking)) != cudaSuccess) { fail("cudaStreamCreate", e); return; }
    for (cudaEvent_t* ev : {&evt_vis_[0], &evt_vis_[1], &evt_ctrl_[0], &evt_ctrl_[1], &evt_plan_, &evt_admit_, &evt_step_})
        if ((e = cudaEventCreateWithFlags(ev, cudaEventDisableTiming)) != cudaSuccess) { fail("cudaEventCreate", e); return; }
    for (cudaEvent_t* ev : {&evt_bind_start_, &evt_commit_})
        if ((e = cudaEventCreate(ev)) != cudaSuccess) { fail("cudaEventCreate(timing)", e); return; }
    cudaEventRecord(evt_ctrl_[0], s_control_); cudaEventRecord(evt_ctrl_[1], s_control_);
    cudaEventRecord(evt_vis_[0], s_vision_); cudaEventRecord(evt_vis_[1], s_vision_);
    cudaEventRecord(evt_plan_, s_planner_);

    for (int i = 0; i < 2; ++i) {
        d_payload_[i] = (uint8_t*)dalloc(payload_bytes_);
        d_rgb8_[i] = (cfg_.pixfmt == PIX_RGB24) ? nullptr : (uint8_t*)dalloc((size_t)3 * n_);
        d_grey_[i] = (float*)dalloc(plane_bytes_);
        d_rgbf_[i] = (chans_ == 3) ? (float*)dalloc(3 * plane_bytes_) : nullptr;
        d_box_[i] = (int*)dalloc(4 * sizeof(int));
        d_probe_payload_[i] = (uint8_t*)dalloc(payload_bytes_);
        d_probe_rgb8_[i] = (uint8_t*)dalloc((size_t)3 * n_);
        d_probe_grey_[i] = (float*)dalloc(plane_bytes_);
        d_probe_rgbf_[i] = (float*)dalloc(3 * plane_bytes_);
    }
    d_rgb8_b_ = (uint8_t*)dalloc((size_t)3 * n_);
    d_grey_b_ = (float*)dalloc(plane_bytes_);
    d_rgbf_b_ = (chans_ == 3) ? (float*)dalloc(3 * plane_bytes_) : nullptr;
    size_t tpl = (size_t)side_ * side_ * 3;
    d_traw_ = (float*)dalloc(tpl * sizeof(float)); d_tz_ = (float*)dalloc(tpl * sizeof(float));
    d_tstats_ = (TemplateStats*)dalloc(sizeof(TemplateStats));
    d_scores_ = (double*)dalloc((size_t)nscores_ * sizeof(double));
    d_am_partial_ = (ArgmaxPair*)dalloc((size_t)nblocks_ * sizeof(ArgmaxPair)); d_am_ = (ArgmaxPair*)dalloc(sizeof(ArgmaxPair));
    d_pk_partial_ = (PeakStats*)dalloc((size_t)nblocks_ * sizeof(PeakStats)); d_pk_ = (PeakStats*)dalloc(sizeof(PeakStats));
    d_chunks_ = (uint64_t*)dalloc((size_t)vision::hash_chunks(payload_bytes_) * sizeof(uint64_t));
    d_hash_ = (uint64_t*)dalloc(sizeof(uint64_t)); d_change_ = (unsigned long long*)dalloc(sizeof(unsigned long long));
    d_patch_flag_ = (int*)dalloc(sizeof(int)); d_patch_ = (uint8_t*)dalloc((size_t)3 * (2 * MAX_PENDING_RADIUS + 1) * (2 * MAX_PENDING_RADIUS + 1));
    d_env_ = (ObservationEnvelope*)dalloc(sizeof(ObservationEnvelope));
    d_state_ = (AgentState*)dalloc(sizeof(AgentState)); d_memory_ = (Memory*)dalloc(sizeof(Memory));
    d_slots_ = (CommittedAction*)dalloc((size_t)cfg_.slots * sizeof(CommittedAction));
    d_trace_ = (DecisionTrace*)dalloc((size_t)cfg_.trace_entries * sizeof(DecisionTrace)); d_trace_count_ = (uint32_t*)dalloc(sizeof(uint32_t));
    d_stamp_ = (unsigned long long*)dalloc(sizeof(unsigned long long));
    d_planner_img_ = (uint8_t*)dalloc((size_t)3 * n_ * planner_scale_ * planner_scale_);
    d_grounder_img_ = (uint8_t*)dalloc((size_t)3 * n_ * planner_scale_ * planner_scale_);
    d_ledger_ = (LedgerRow*)dalloc((size_t)cfg_.ledger_rows * sizeof(LedgerRow)); d_ledger_count_ = (uint32_t*)dalloc(sizeof(uint32_t));
    d_probe_state_ = (AgentState*)dalloc(sizeof(AgentState));
    d_probe_traw_ = (float*)dalloc(tpl * sizeof(float)); d_probe_tz_ = (float*)dalloc(tpl * sizeof(float));
    d_probe_tstats_ = (TemplateStats*)dalloc(sizeof(TemplateStats));
    if (!ok_) return;
    h_env_ = host_pinned<ObservationEnvelope>(1); h_state_ = host_pinned<AgentState>(1); h_slot_ = host_pinned<CommittedAction>(1);
    h_memory_ = host_pinned<Memory>(1); h_trace_ = host_pinned<DecisionTrace>(cfg_.trace_entries); h_trace_count_ = host_pinned<uint32_t>(1);
    h_am_ = host_pinned<ArgmaxPair>(1); h_pk_ = host_pinned<PeakStats>(1); h_tstats_ = host_pinned<TemplateStats>(1);
    h_stamp_ = host_pinned<unsigned long long>(1);
    h_ledger_ = host_pinned<LedgerRow>(cfg_.ledger_rows); h_ledger_count_ = host_pinned<uint32_t>(1);
    if (!h_ledger_ || !h_ledger_count_) { fail("cudaHostAlloc(ledger)"); return; }
    if (!h_env_ || !h_state_ || !h_slot_ || !h_memory_ || !h_trace_ || !h_trace_count_ || !h_am_ || !h_pk_ || !h_tstats_ || !h_stamp_) { fail("cudaHostAlloc"); return; }
    control::reset(d_state_, d_memory_, s_control_);
    control::reset(d_probe_state_, d_memory_, s_control_);   // the probe state; the memory is re-reset below
    control::reset(d_state_, d_memory_, s_control_);
    if ((e = cudaStreamSynchronize(s_control_)) != cudaSuccess) { fail("reset", e); return; }
    if (!calibrate_clock()) return;
    if (!warmup()) return;
    warm_ = true;
}

Core::~Core() {
    for (cudaStream_t s : {s_vision_, s_control_, s_planner_, s_memory_}) if (s) cudaStreamSynchronize(s);
    for (void* p : allocs_) cudaFree(p);
    for (void* p : {(void*)h_env_, (void*)h_state_, (void*)h_slot_, (void*)h_memory_, (void*)h_trace_, (void*)h_trace_count_,
                    (void*)h_am_, (void*)h_pk_, (void*)h_tstats_, (void*)h_stamp_, (void*)h_ledger_, (void*)h_ledger_count_}) if (p) cudaFreeHost(p);
    for (cudaEvent_t ev : {evt_vis_[0], evt_vis_[1], evt_ctrl_[0], evt_ctrl_[1], evt_plan_, evt_admit_, evt_step_, evt_bind_start_, evt_commit_}) if (ev) cudaEventDestroy(ev);
    for (cudaStream_t s : {s_vision_, s_control_, s_planner_, s_memory_}) if (s) cudaStreamDestroy(s);
}

bool Core::calibrate_clock() {
    // %globaltimer against CLOCK_MONOTONIC: the sample with the smallest round trip bounds the offset error
    int64_t best_lat = INT64_MAX, best_off = 0;
    for (int i = 0; i < 64; ++i) {
        int64_t h0 = host_now_ns();
        vision::stamp(d_stamp_, s_control_);
        cudaError_t e = cudaStreamSynchronize(s_control_);
        int64_t h1 = host_now_ns();
        if (e != cudaSuccess) return fail("clock calibration", e);
        cudaMemcpy(h_stamp_, d_stamp_, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        int64_t lat = h1 - h0;
        if (lat < best_lat) { best_lat = lat; best_off = (int64_t)*h_stamp_ - (h0 + h1) / 2; }
    }
    clock_offset_ = best_off; clock_latency_ = best_lat;
    return true;
}

void Core::reset_all() {
    control::reset(d_state_, d_memory_, s_control_);
    cudaMemsetAsync(d_trace_count_, 0, sizeof(uint32_t), s_control_);
    cudaMemsetAsync(d_ledger_count_, 0, sizeof(uint32_t), s_control_);
    cudaMemsetAsync(d_slots_, 0, (size_t)cfg_.slots * sizeof(CommittedAction), s_control_);
    cudaStreamSynchronize(s_control_);
    cur_ = 0; have_prev_ = false; last_obs_id_ = 0; admitted_ = 0; decision_open_ = false; proposal_active_ = false;
    decision_counter_ = 0; envelopes_.clear(); audit_ = Audit{};
}

bool Core::warmup() {
    // exercise every kernel once so that module loading and first-launch costs never fall inside a trial
    std::vector<uint8_t> z(expected_len(), 0);
    ObservationEnvelope env{1, 0, 0, 0, 0, 0};
    chronotope_admission a{}; chronotope_proposal p{}; chronotope_commit_result r{};
    if (admit(z.data(), z.size(), env, &a) != CHRONOTOPE_OK) return false;
    env.observation_id = 2;
    if (admit(z.data(), z.size(), env, &a) != CHRONOTOPE_OK) return false;
    if (begin_decision(2, &p) != CHRONOTOPE_OK) return false;
    if (decide_detector(&p) != CHRONOTOPE_OK) return false;     // no target on a black frame: no action
    if (begin_decision(2, &p) != CHRONOTOPE_OK) return false;
    if (propose(CHRONOTOPE_ACT_CLICK, w_ / 2, h_ / 2, 0, &p) != CHRONOTOPE_OK) return false;
    env.observation_id = 3;
    if (admit(z.data(), z.size(), env, &a) != CHRONOTOPE_OK) return false;
    if (resolve(true, false, &r) != CHRONOTOPE_OK) return false;
    if (!ok_) return false;
    reset_all();
    return true;
}

// ---------------------------------------------------------------------------------------------- O
int Core::admit(const uint8_t* payload, size_t len, const ObservationEnvelope& env_in, chronotope_admission* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (len != expected_len()) { fail("payload length " + std::to_string(len) + " != " + std::to_string(expected_len())); ok_ = true; return CHRONOTOPE_ERR; }
    std::memset(out, 0, sizeof(*out));
    if (admitted_ > 0 && env_in.observation_id <= last_obs_id_) {   // §7.2 / §16 item 14: never a superseded observation
        out->status = CHRONOTOPE_STALE;
        return CHRONOTOPE_STALE;
    }
    int slot = admitted_ > 0 ? (cur_ ^ 1) : 0;
    int prev = cur_;
    int64_t t_host = host_now_ns();
    audit_.h2d_publications++;
    if (audit_.window_open) audit_.kernel_launches_in_window++;
    // vision stream: this slot may be overwritten only after the control stream and the planner are done with it
    cudaStreamWaitEvent(s_vision_, evt_ctrl_[slot], 0);
    cudaStreamWaitEvent(s_vision_, evt_plan_, 0);
    cudaMemcpyAsync(d_payload_[slot], payload, len, cudaMemcpyHostToDevice, s_vision_);
    uint8_t* rgb8 = rgb8view(slot);
    vision::to_planes(d_payload_[slot], cfg_.pixfmt, w_, h_, rgb8, d_grey_[slot], d_rgbf_[slot], s_vision_);
    vision::hash_payload(d_payload_[slot], len, d_chunks_, d_hash_, s_vision_);
    if (have_prev_) {
        cudaMemsetAsync(d_change_, 0, sizeof(unsigned long long), s_vision_);
        vision::change_score(d_grey_[slot], d_grey_[prev], n_, d_change_, s_vision_);
    }
    if (cfg_.has_target) vision::detect_color(rgb8, w_, h_, cfg_.target_r, cfg_.target_g, cfg_.target_b, d_box_[slot], s_vision_);
    else vision::zero_box(d_box_[slot], s_vision_);
    cudaEventRecord(evt_vis_[slot], s_vision_);
    // control stream: update the state from what was perceived
    cudaStreamWaitEvent(s_control_, evt_vis_[slot], 0);
    *h_env_ = env_in;
    h_env_->observation_version = admitted_ + 1;
    cudaMemcpyAsync(d_env_, h_env_, sizeof(ObservationEnvelope), cudaMemcpyHostToDevice, s_control_);
    bool track = decision_open_ && proposal_active_ && cfg_.track_every_frame && cfg_.policy != POL_FROZEN;
    if (track) launch_ncc_on(slot, s_control_);
    int* pflag = nullptr;
    if (cfg_.pending_timeout_ns > 0) {
        cudaMemsetAsync(d_patch_flag_, 0, sizeof(int), s_control_);
        vision::patch_diff(rgb8, w_, h_, d_memory_, d_patch_, d_patch_flag_, s_control_);
        pflag = d_patch_flag_;
    }
    control::admit_update(d_state_, d_memory_, d_env_, d_hash_, d_change_, d_box_[slot], track ? d_am_ : nullptr, d_pk_, d_tstats_,
                          pflag, cfg_, have_prev_ ? 1 : 0, n_, t_host, Wv_, d_trace_, d_trace_count_, d_ledger_, d_ledger_count_, s_control_);
    cudaEventRecord(evt_ctrl_[slot], s_control_);
    cudaEventRecord(evt_admit_, s_control_);
    // the admission record for the trace: a log read of a completed event (§5.2)
    if (audit_.window_open) { audit_.host_syncs_log_in_window++; audit_.d2h_log_in_window++; }
    cudaError_t e = cudaEventSynchronize(evt_admit_);
    if (e != cudaSuccess) { fail("admission", e); return CHRONOTOPE_ERR; }
    cudaMemcpy(&h_state_->control, &d_state_->control, sizeof(ControlState), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_memory_->pending, &d_memory_->pending, sizeof(PendingAction), cudaMemcpyDeviceToHost);
    const ControlState& c = h_state_->control;
    out->status = CHRONOTOPE_OK; out->relevance = c.last_relevance;
    out->control_state_version = c.control_state_version; out->observation_version = c.observation_version;
    out->payload_hash = c.payload_hash; out->controller_state_hash = c.controller_state_hash; out->pending_actions_hash = c.pending_actions_hash;
    out->change_mean = c.change_mean;
    out->box[0] = c.box.x0; out->box[1] = c.box.y0; out->box[2] = c.box.x1; out->box[3] = c.box.y1; out->box_present = c.box.present;
    out->tracked_x = c.tracked_x; out->tracked_y = c.tracked_y; out->tracked_score = c.tracked_score; out->tracked_valid = c.tracked_valid;
    out->pending_active = h_memory_->pending.active; out->pending_event = c.last_pending_event;
    out->admit_device_ns = map_device_ns(c.last_admit_device_ns); out->admit_host_ns = t_host;
    cur_ = slot; have_prev_ = true; last_obs_id_ = env_in.observation_id; admitted_++;
    ObservationEnvelope logged = env_in; logged.observation_version = admitted_; logged.payload_hash = c.payload_hash;
    envelopes_.push_back(logged);
    return CHRONOTOPE_OK;
}

void Core::launch_ncc_on(int slot, cudaStream_t s) {
    vision::ncc(d_grey_[slot], d_rgbf_[slot], w_, h_, side_, chans_, d_tz_, d_tstats_, d_scores_, s);
    vision::argmax(d_scores_, nscores_, d_am_partial_, d_am_, s);
    vision::peak_stats(d_scores_, Wv_, Hv_, d_am_, side_, d_pk_partial_, d_pk_, s);
}

// ---------------------------------------------------------------------------------------------- B
int Core::begin_decision(uint64_t observation_id, chronotope_proposal* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (decision_open_) { fail("a decision is already open"); ok_ = true; return CHRONOTOPE_ERR; }
    if (admitted_ == 0 || observation_id != last_obs_id_) { fail("a decision begins on the newest admitted observation"); ok_ = true; return CHRONOTOPE_ERR; }
    decision_counter_++;
    audit_.window_open = true; audit_.windows_opened++;
    int64_t b_host = host_now_ns();
    // planner stream: pin the frame at B (device to device), so that the template is cut from it whenever the planner answers
    cudaStreamWaitEvent(s_planner_, evt_vis_[cur_], 0);
    cudaMemcpyAsync(d_rgb8_b_, rgb8view(cur_), (size_t)3 * n_, cudaMemcpyDeviceToDevice, s_planner_);
    cudaMemcpyAsync(d_grey_b_, d_grey_[cur_], plane_bytes_, cudaMemcpyDeviceToDevice, s_planner_);
    if (chans_ == 3) cudaMemcpyAsync(d_rgbf_b_, d_rgbf_[cur_], 3 * plane_bytes_, cudaMemcpyDeviceToDevice, s_planner_);
    cudaEventRecord(evt_plan_, s_planner_);
    control::begin(d_state_, decision_counter_, observation_id, b_host, d_ledger_, d_ledger_count_, cfg_, s_control_);
    cudaEventRecord(evt_step_, s_control_);
    decision_open_ = true; proposal_active_ = false;
    if (out) {
        std::memset(out, 0, sizeof(*out));
        audit_.host_syncs_log_in_window++; audit_.d2h_log_in_window++;
        cudaEventSynchronize(evt_step_);
        cudaMemcpy(&h_state_->decision, &d_state_->decision, sizeof(DecisionInfo), cudaMemcpyDeviceToHost);
        out->decision_id = decision_counter_; out->observation_id = observation_id;
        out->supporting_control_state_version = h_state_->decision.supporting_control_state_version;
        out->b_device_ns = map_device_ns(h_state_->decision.b_device_ns); out->b_host_ns = b_host;
    }
    return CHRONOTOPE_OK;
}

int Core::cut_template_now(chronotope_proposal* out) {
    cudaStreamWaitEvent(s_control_, evt_plan_, 0);
    vision::cut_template(d_grey_b_, d_rgbf_b_, w_, h_, cfg_.radius, chans_, d_state_, d_traw_, s_control_);
    vision::template_stats(d_traw_, side_ * side_ * chans_, side_, chans_, d_tstats_, d_tz_, s_control_);
    cudaEventRecord(evt_step_, s_control_);
    return read_proposal(out);
}

int Core::read_proposal(chronotope_proposal* out) {
    // whether a proposal exists, and what it is, for the trace (action_proposed): a log read after the planner kernel
    audit_.host_syncs_log_in_window++; audit_.d2h_log_in_window++;
    cudaError_t e = cudaEventSynchronize(evt_step_);
    if (e != cudaSuccess) { fail("proposal", e); return CHRONOTOPE_ERR; }
    cudaMemcpy(&h_state_->proposal, &d_state_->proposal, sizeof(Proposal), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_state_->decision, &d_state_->decision, sizeof(DecisionInfo), cudaMemcpyDeviceToHost);
    const Proposal& p = h_state_->proposal;
    proposal_active_ = (p.status == PS_PROPOSED);
    if (out) {
        std::memset(out, 0, sizeof(*out));
        out->status = proposal_active_ ? 1 : 0;
        out->kind = p.action.kind; out->x = p.action.x; out->y = p.action.y; out->value = p.action.value;
        out->observation_id = p.action.observation_id; out->decision_id = p.decision_id;
        out->supporting_control_state_version = h_state_->decision.supporting_control_state_version;
        out->b_device_ns = map_device_ns(h_state_->decision.b_device_ns); out->b_host_ns = h_state_->decision.b_host_ns;
    }
    if (!proposal_active_) close_window();   // no action this decision: the window closes, a new decision may begin
    return CHRONOTOPE_OK;
}

int Core::close_window() {
    decision_open_ = false; proposal_active_ = false; audit_.window_open = false;
    return CHRONOTOPE_OK;
}

int Core::decide_detector(chronotope_proposal* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (!decision_open_) { fail("no open decision"); ok_ = true; return CHRONOTOPE_ERR; }
    control::decide_detector(d_state_, cfg_, s_control_);
    return cut_template_now(out);
}

int Core::propose(uint32_t kind, int32_t x, int32_t y, int32_t value, chronotope_proposal* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (!decision_open_) { fail("no open decision"); ok_ = true; return CHRONOTOPE_ERR; }
    audit_.h2d_proposals++;
    Action a{kind, x, y, value, 1.0f, 0};
    control::set_proposal(d_state_, a, s_control_);
    return cut_template_now(out);
}

int Core::abort_decision() {
    if (!ok_) return CHRONOTOPE_ERR;
    control::discard(d_state_, s_control_);
    cudaStreamSynchronize(s_control_);
    return close_window();
}

int Core::restart() { return abort_decision(); }

// ---------------------------------------------------------------------------------------------- D
int Core::resolve(bool do_commit, bool commit_only, chronotope_commit_result* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    std::memset(out, 0, sizeof(*out));
    if (!decision_open_) { out->outcome = CHRONOTOPE_CS_NO_PROPOSAL; return CHRONOTOPE_NO_DECISION; }
    int64_t t0 = host_now_ns();
    bool use_ncc = proposal_active_ && cfg_.policy != POL_FROZEN;
    if (!commit_only) {
        cudaEventRecord(evt_bind_start_, s_control_);
        if (use_ncc) launch_ncc_on(cur_, s_control_);
        control::resolve(d_state_, d_am_, d_pk_, d_tstats_, d_box_[cur_], cfg_, use_ncc ? 1 : 0, Wv_, s_control_);
    }
    if (do_commit) {
        control::commit(d_state_, d_memory_, d_slots_, cfg_, d_trace_, d_trace_count_, host_now_ns(), 0, d_ledger_, d_ledger_count_, s_control_);
        if (cfg_.pending_timeout_ns > 0) vision::cut_patch(rgb8view(cur_), w_, h_, d_state_, d_memory_, d_patch_, s_control_);
    }
    cudaEventRecord(evt_ctrl_[cur_], s_control_);
    cudaEventRecord(evt_commit_, s_control_);
    audit_.declared_commit_waits++;
    cudaError_t e = cudaEventSynchronize(evt_commit_);
    int64_t t1 = host_now_ns();
    if (e != cudaSuccess) { fail("resolve/commit", e); return CHRONOTOPE_ERR; }
    if (!do_commit) { out->outcome = CHRONOTOPE_CS_NOT_ATTEMPTED; out->track_ns = t1 - t0; return CHRONOTOPE_OK; }   // no readback before D
    cudaMemcpy(h_state_, d_state_, sizeof(AgentState), cudaMemcpyDeviceToHost);
    const BindResult& b = h_state_->bind;
    const CommitScratch& c = h_state_->commit;
    const Proposal& p = h_state_->proposal;
    out->outcome = c.status; out->reject_reason = c.reject_reason; out->repetition_count = c.repetition_count;
    out->kind = p.action.kind; out->proposed_x = p.action.x; out->proposed_y = p.action.y; out->value = p.action.value;
    out->decision_id = p.decision_id; out->proposal_observation_id = p.action.observation_id;
    out->match_score = b.score; out->second_best = b.second; out->moved_px = b.moved_px; out->plateau = b.plateau;
    out->matched = b.matched; out->ambiguous = b.ambiguous; out->bind_valid = b.valid; out->binding = b.binding;
    out->bind_x = b.x; out->bind_y = b.y; out->observation_id_at_d = b.observation_id;
    out->control_state_version_at_commit = h_state_->control.control_state_version;
    out->supporting_control_state_version = h_state_->decision.supporting_control_state_version;
    if (c.status == CS_COMMITTED) {
        cudaMemcpy(h_slot_, &d_slots_[c.action_id % cfg_.slots], sizeof(CommittedAction), cudaMemcpyDeviceToHost);
        out->action_id = h_slot_->action_id; out->x = h_slot_->x; out->y = h_slot_->y; out->value = h_slot_->value;
        out->serialized_action_hash = h_slot_->serialized_action_hash;
        out->control_state_version_at_commit = h_slot_->control_state_version_at_commit;
        out->supporting_control_state_version = h_slot_->supporting_control_state_version;
        out->d_device_ns = h_slot_->commitment_device_ns; out->d_device_mapped_ns = map_device_ns(h_slot_->commitment_device_ns);
        out->pending_created = (cfg_.pending_timeout_ns > 0 && h_slot_->kind == ACT_CLICK) ? 1 : 0;
    } else {
        out->x = b.resolved_x; out->y = b.resolved_y;
    }
    out->evidence_age_ns = b.evidence_age_ns; out->evidence_bracket_ns = b.evidence_bracket_ns; out->ledger_row = b.ledger_row;
    out->d_host_ns = t1;
    out->track_ns = t1 - t0;
    if (!commit_only) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, evt_bind_start_, evt_commit_) == cudaSuccess) out->track_device_ns = (int64_t)(ms * 1e6);
    }
    close_window();
    return CHRONOTOPE_OK;
}

int Core::read_frame(uint8_t* buf, size_t len) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (len < (size_t)3 * n_) { fail("read_frame buffer too small"); ok_ = true; return CHRONOTOPE_ERR; }
    if (admitted_ == 0) { fail("no frame"); ok_ = true; return CHRONOTOPE_ERR; }
    // decision STATE leaves the device: inside the reactive window this is what §5.3 prohibits
    if (audit_.window_open) { audit_.d2h_state_in_window++; audit_.host_syncs_action_in_window++; }
    cudaEventSynchronize(evt_vis_[cur_]);
    cudaMemcpy(buf, rgb8view(cur_), (size_t)3 * n_, cudaMemcpyDeviceToHost);
    return CHRONOTOPE_OK;
}

int Core::override_binding(int32_t x, int32_t y) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (!decision_open_) { fail("no open decision"); ok_ = true; return CHRONOTOPE_ERR; }
    audit_.host_action_computations++;   // a host thread returned a revised proposal to the GPU (§5.3)
    control::override_binding(d_state_, x, y, s_control_);
    return CHRONOTOPE_OK;
}

void Core::audit(chronotope_audit_report* out) const {
    std::memset(out, 0, sizeof(*out));
    out->h2d_publications = audit_.h2d_publications; out->h2d_proposals = audit_.h2d_proposals;
    out->d2h_log_in_window = audit_.d2h_log_in_window; out->d2h_state_in_window = audit_.d2h_state_in_window;
    out->host_syncs_log_in_window = audit_.host_syncs_log_in_window; out->host_syncs_action_in_window = audit_.host_syncs_action_in_window;
    out->declared_commit_waits = audit_.declared_commit_waits; out->host_action_computations = audit_.host_action_computations;
    out->mallocs_after_warmup = audit_.mallocs_after_warmup; out->kernel_launches_in_window = audit_.kernel_launches_in_window;
    out->windows_opened = audit_.windows_opened; out->window_open = audit_.window_open ? 1 : 0; out->residency_ok = audit_.ok() ? 1 : 0;
    out->clock_offset_ns = clock_offset_; out->clock_calibration_latency_ns = clock_latency_; out->device_bytes_allocated = device_bytes_;
    out->planner_host_steps = audit_.planner_host_steps; out->planner_d2h_pixels = audit_.planner_d2h_pixels; out->planner_requests = audit_.planner_requests;
    out->binder_host_steps = audit_.binder_host_steps;
}

// ---------------------------------------------------------------------------------------------- the in-process planner
int Core::frame_b_device(uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (!decision_open_) { fail("no open decision (the frame at B is pinned by begin_decision)"); ok_ = true; return CHRONOTOPE_ERR; }
    if (scale == 0 || scale > planner_scale_) { fail("planner scale exceeds the preallocated buffer"); ok_ = true; return CHRONOTOPE_ERR; }
    // planner stream, after the pin: the upscaled frame is produced where the planner will consume it; the host
    // receives only the pointer
    vision::upscale_nn(d_rgb8_b_, w_, h_, (int)scale, d_planner_img_, s_planner_);
    cudaEventRecord(evt_plan_, s_planner_);
    *ptr = d_planner_img_; *w = (uint32_t)w_ * scale; *h = (uint32_t)h_ * scale;
    return CHRONOTOPE_OK;
}

int Core::ledger(chronotope_ledger_row* buf, uint32_t n, uint32_t* count) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (audit_.window_open) { audit_.host_syncs_log_in_window++; audit_.d2h_log_in_window++; }
    cudaStreamWaitEvent(s_memory_, evt_ctrl_[cur_], 0);
    cudaMemcpyAsync(h_ledger_count_, d_ledger_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost, s_memory_);
    cudaMemcpyAsync(h_ledger_, d_ledger_, (size_t)cfg_.ledger_rows * sizeof(LedgerRow), cudaMemcpyDeviceToHost, s_memory_);
    cudaError_t e = cudaStreamSynchronize(s_memory_);
    if (e != cudaSuccess) { fail("ledger", e); return CHRONOTOPE_ERR; }
    uint32_t total = *h_ledger_count_, avail = std::min(total, cfg_.ledger_rows), start = total - avail, m = std::min(avail, n);
    for (uint32_t i = 0; i < m; ++i) {
        const LedgerRow& r = h_ledger_[(start + i) % cfg_.ledger_rows];
        chronotope_ledger_row& o = buf[i];
        o.observation_id = r.observation_id; o.o_ns = r.o_ns; o.o_device_ns = map_device_ns(r.o_device_ns); o.source_time_ns = r.source_time_ns;
        o.e_bracket_lo_ns = r.e_bracket_lo_ns; o.frames_since_prev = r.frames_since_prev; o.relevance = r.relevance;
        o.control_state_version = r.control_state_version; o.decision_proposed = r.decision_proposed; o.decision_bound = r.decision_bound;
        o.d_device_ns = r.d_device_ns ? map_device_ns(r.d_device_ns) : 0; o.effect_observed_ns = r.effect_observed_ns; o.receipt_observed_ns = r.receipt_observed_ns;
    }
    *count = m;
    return CHRONOTOPE_OK;
}

int Core::bind_external(int32_t x, int32_t y, uint32_t binding) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (!decision_open_) { fail("no open decision"); ok_ = true; return CHRONOTOPE_ERR; }
    if (binding == CHRONOTOPE_BIND_ORACLE) audit_.host_action_computations++;   // the environment's truth: host-determined, test only
    else audit_.binder_host_steps++;                                     // an in-process model's answer crossed host memory
    control::bind_external(d_state_, x, y, binding, s_control_);
    return CHRONOTOPE_OK;
}

int Core::observe_receipt(uint64_t decision_id, int64_t o_ns) {
    if (!ok_) return CHRONOTOPE_ERR;
    control::observe_receipt(d_memory_, decision_id, o_ns, d_ledger_, d_ledger_count_, cfg_, s_control_);
    return CHRONOTOPE_OK;
}

int Core::frame_newest_device(uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (admitted_ == 0) { fail("no frame"); ok_ = true; return CHRONOTOPE_ERR; }
    if (scale == 0 || scale > planner_scale_) { fail("scale exceeds the preallocated buffer"); ok_ = true; return CHRONOTOPE_ERR; }
    cudaStreamWaitEvent(s_planner_, evt_vis_[cur_], 0);
    vision::upscale_nn(rgb8view(cur_), w_, h_, (int)scale, d_grounder_img_, s_planner_);
    cudaEventRecord(evt_plan_, s_planner_);
    *ptr = d_grounder_img_; *w = (uint32_t)w_ * scale; *h = (uint32_t)h_ * scale;
    return CHRONOTOPE_OK;
}

void Core::note_planner(uint64_t host_steps, uint64_t d2h_pixels) {
    audit_.planner_host_steps += host_steps; audit_.planner_d2h_pixels += d2h_pixels; audit_.planner_requests += 1;
}

int Core::trace(chronotope_trace_entry* buf, uint32_t n, uint32_t* count) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (audit_.window_open) { audit_.host_syncs_log_in_window++; audit_.d2h_log_in_window++; }
    cudaStreamWaitEvent(s_memory_, evt_ctrl_[cur_], 0);
    cudaMemcpyAsync(h_trace_count_, d_trace_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost, s_memory_);
    cudaMemcpyAsync(h_trace_, d_trace_, (size_t)cfg_.trace_entries * sizeof(DecisionTrace), cudaMemcpyDeviceToHost, s_memory_);
    cudaError_t e = cudaStreamSynchronize(s_memory_);
    if (e != cudaSuccess) { fail("trace", e); return CHRONOTOPE_ERR; }
    uint32_t total = *h_trace_count_;
    uint32_t avail = std::min(total, cfg_.trace_entries);
    uint32_t start = total - avail;
    uint32_t m = std::min(avail, n);
    for (uint32_t i = 0; i < m; ++i) {
        const DecisionTrace& t = h_trace_[(start + i) % cfg_.trace_entries];
        chronotope_trace_entry& o = buf[i];
        o.observation_id = t.observation_id; o.decision_id = t.decision_id; o.kind = t.proposed.kind; o.x = t.proposed.x; o.y = t.proposed.y;
        o.accepted = t.accepted; o.reason = t.reason; o.screen_change = t.screen_change; o.repetition_count = t.repetition_count;
        o.device_ns = t.device_ns; o.device_mapped_ns = map_device_ns(t.device_ns); o.control_state_version = t.control_state_version;
    }
    *count = m;
    return CHRONOTOPE_OK;
}

int Core::state_hash(uint64_t* h, uint64_t* v) {
    if (!ok_) return CHRONOTOPE_ERR;
    if (audit_.window_open) { audit_.host_syncs_log_in_window++; audit_.d2h_log_in_window++; }
    cudaStreamSynchronize(s_control_);
    cudaMemcpy(&h_state_->control, &d_state_->control, sizeof(ControlState), cudaMemcpyDeviceToHost);
    *h = h_state_->control.controller_state_hash; *v = h_state_->control.control_state_version;
    return CHRONOTOPE_OK;
}

// ---------------------------------------------------------------------------------------------- the binder alone
int Core::track(const uint8_t* frame_b, const uint8_t* frame_d, size_t len, int32_t x, int32_t y, uint32_t radius, uint8_t rgb, chronotope_match* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    std::memset(out, 0, sizeof(*out));
    if (len != expected_len()) { fail("track: payload length"); ok_ = true; return CHRONOTOPE_ERR; }
    if ((int)radius != cfg_.radius) { fail("track: radius must equal the core's template radius"); ok_ = true; return CHRONOTOPE_ERR; }
    int chans = rgb ? 3 : 1;
    cudaStream_t s = s_memory_;
    cudaStreamSynchronize(s_control_);   // the probe shares the score buffer; never concurrent with the reactive path
    cudaMemcpyAsync(d_probe_payload_[0], frame_b, len, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_probe_payload_[1], frame_d, len, cudaMemcpyHostToDevice, s);
    uint8_t* rgb8_0 = (cfg_.pixfmt == PIX_RGB24) ? d_probe_payload_[0] : d_probe_rgb8_[0];
    uint8_t* rgb8_1 = (cfg_.pixfmt == PIX_RGB24) ? d_probe_payload_[1] : d_probe_rgb8_[1];
    vision::to_planes(d_probe_payload_[0], cfg_.pixfmt, w_, h_, rgb8_0, d_probe_grey_[0], d_probe_rgbf_[0], s);
    vision::to_planes(d_probe_payload_[1], cfg_.pixfmt, w_, h_, rgb8_1, d_probe_grey_[1], d_probe_rgbf_[1], s);
    control::begin(d_probe_state_, 1, 1, 0, d_ledger_, d_ledger_count_, cfg_, s);
    control::set_proposal(d_probe_state_, Action{ACT_CLICK, x, y, 0, 1.0f, 1}, s);
    vision::cut_template(d_probe_grey_[0], d_probe_rgbf_[0], w_, h_, cfg_.radius, chans, d_probe_state_, d_probe_traw_, s);
    vision::template_stats(d_probe_traw_, side_ * side_ * chans, side_, chans, d_probe_tstats_, d_probe_tz_, s);
    vision::ncc(d_probe_grey_[1], d_probe_rgbf_[1], w_, h_, side_, chans, d_probe_tz_, d_probe_tstats_, d_scores_, s);
    vision::argmax(d_scores_, nscores_, d_am_partial_, d_am_, s);
    vision::peak_stats(d_scores_, Wv_, Hv_, d_am_, side_, d_pk_partial_, d_pk_, s);
    cudaMemcpyAsync(h_am_, d_am_, sizeof(ArgmaxPair), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_pk_, d_pk_, sizeof(PeakStats), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_tstats_, d_probe_tstats_, sizeof(TemplateStats), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&h_state_->proposal, &d_probe_state_->proposal, sizeof(Proposal), cudaMemcpyDeviceToHost, s);
    cudaError_t e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) { fail("track", e); return CHRONOTOPE_ERR; }
    const Proposal& p = h_state_->proposal;
    out->tpl_x0 = p.tpl_x0; out->tpl_y0 = p.tpl_y0; out->tpl_dx = p.tpl_dx; out->tpl_dy = p.tpl_dy;
    // tracker.py v2's verdict (the same rule as control.cu's bind_from_ncc, here for the offline probe)
    if (h_tstats_->flat) {
        out->x = p.tpl_dx; out->y = p.tpl_dy; out->score = 0.0; out->second = 1.0; out->plateau = FLAT_PLATEAU; out->flat = 1;
    } else {
        int iy = h_am_->index / Wv_, ix = h_am_->index % Wv_;
        out->x = ix + p.tpl_dx; out->y = iy + p.tpl_dy;
        out->score = std::min(1.0, std::max(-1.0, h_am_->score)); out->second = h_pk_->second; out->plateau = h_pk_->plateau;
    }
    out->ambiguous = (out->second >= out->score - AMBIGUITY_MARGIN || out->plateau > PLATEAU_MAX) ? 1 : 0;
    out->matched = (out->score >= cfg_.min_score && !out->ambiguous) ? 1 : 0;
    out->moved_px = out->matched ? std::hypot((double)(out->x - x), (double)(out->y - y)) : 0.0;
    return CHRONOTOPE_OK;
}

// ---------------------------------------------------------------------------------------------- contract-test hooks
int Core::test_recommit(chronotope_commit_result* out) {
    if (!ok_) return CHRONOTOPE_ERR;
    std::memset(out, 0, sizeof(*out));
    control::commit(d_state_, d_memory_, d_slots_, cfg_, d_trace_, d_trace_count_, host_now_ns(), 1, d_ledger_, d_ledger_count_, s_control_);
    cudaStreamSynchronize(s_control_);
    cudaMemcpy(&h_state_->commit, &d_state_->commit, sizeof(CommitScratch), cudaMemcpyDeviceToHost);
    out->outcome = h_state_->commit.status; out->action_id = h_state_->commit.action_id;
    return CHRONOTOPE_OK;
}

int Core::test_alloc() {
    if (!ok_) return CHRONOTOPE_ERR;
    return dalloc(4096) ? CHRONOTOPE_OK : CHRONOTOPE_ERR;
}

}  // namespace chronotope
