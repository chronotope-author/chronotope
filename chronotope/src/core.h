// core.h — the host wrapper: preallocated device buffers, four streams, the audit. Nothing here
// computes an action; it publishes observations, launches kernels, waits at D, and reads logs.
#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "chronotope_api.h"
#include "state.cuh"

namespace chronotope {

struct Audit {
    uint64_t h2d_publications = 0, h2d_proposals = 0, d2h_log_in_window = 0, d2h_state_in_window = 0;
    uint64_t host_syncs_log_in_window = 0, host_syncs_action_in_window = 0, declared_commit_waits = 0;
    uint64_t host_action_computations = 0, mallocs_after_warmup = 0, kernel_launches_in_window = 0, windows_opened = 0;
    uint64_t planner_host_steps = 0, planner_d2h_pixels = 0, planner_requests = 0, binder_host_steps = 0;
    bool window_open = false;
    bool ok() const {
        return d2h_state_in_window == 0 && host_syncs_action_in_window == 0 && host_action_computations == 0 && mallocs_after_warmup == 0;
    }
};

class Core {
public:
    explicit Core(const chronotope_config& cfg);
    ~Core();
    bool ok() const { return ok_; }
    const std::string& error() const { return err_; }
    void set_error(const std::string& e) { err_ = e; }          // K4d: the ABI's planner errors
    uint32_t width() const { return (uint32_t)cfg_.w; }
    uint32_t height() const { return (uint32_t)cfg_.h; }

    int admit(const uint8_t* payload, size_t len, const ObservationEnvelope& env_in, chronotope_admission* out);
    int begin_decision(uint64_t observation_id, chronotope_proposal* out);
    int decide_detector(chronotope_proposal* out);
    int propose(uint32_t kind, int32_t x, int32_t y, int32_t value, chronotope_proposal* out);
    int abort_decision();
    int resolve(bool do_commit, bool commit_only, chronotope_commit_result* out);
    int restart();
    int decision_open() const { return decision_open_ ? 1 : 0; }
    int read_frame(uint8_t* buf, size_t len);
    int override_binding(int32_t x, int32_t y);
    void audit(chronotope_audit_report* out) const;
    int trace(chronotope_trace_entry* buf, uint32_t n, uint32_t* count);
    int state_hash(uint64_t* h, uint64_t* v);
    int track(const uint8_t* frame_b, const uint8_t* frame_d, size_t len, int32_t x, int32_t y, uint32_t radius, uint8_t rgb, chronotope_match* out);
    int test_recommit(chronotope_commit_result* out);
    int test_alloc();
    int frame_b_device(uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h);
    void* planner_stream() const { return (void*)s_planner_; }
    void note_planner(uint64_t host_steps, uint64_t d2h_pixels);
    int ledger(chronotope_ledger_row* buf, uint32_t n, uint32_t* count);
    int bind_external(int32_t x, int32_t y, uint32_t binding);
    int observe_receipt(uint64_t decision_id, int64_t o_ns);
    int frame_newest_device(uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h);

private:
    bool fail(const std::string& what, cudaError_t e = cudaSuccess);
    void* dalloc(size_t bytes);
    bool calibrate_clock();
    bool warmup();
    void reset_all();
    int64_t map_device_ns(int64_t t) const { return t - clock_offset_; }
    void launch_ncc_on(int slot, cudaStream_t s);
    int read_proposal(chronotope_proposal* out);
    int close_window();
    uint8_t* rgb8view(int slot) const { return cfg_.pixfmt == PIX_RGB24 ? d_payload_[slot] : d_rgb8_[slot]; }
    size_t expected_len() const;
    int cut_template_now(chronotope_proposal* out);

    chronotope_config pub_{};
    Config cfg_{};
    bool ok_ = false;
    std::string err_;
    int w_ = 0, h_ = 0, n_ = 0, side_ = 0, chans_ = 1, Wv_ = 0, Hv_ = 0, nscores_ = 0, nblocks_ = 0;
    size_t payload_bytes_ = 0, plane_bytes_ = 0;
    bool warm_ = false;
    size_t device_bytes_ = 0;
    std::vector<void*> allocs_;

    cudaStream_t s_vision_ = nullptr, s_control_ = nullptr, s_planner_ = nullptr, s_memory_ = nullptr;
    cudaEvent_t evt_vis_[2] = {nullptr, nullptr}, evt_ctrl_[2] = {nullptr, nullptr}, evt_plan_ = nullptr;
    cudaEvent_t evt_admit_ = nullptr, evt_step_ = nullptr, evt_bind_start_ = nullptr, evt_commit_ = nullptr;

    // preallocated device buffers
    uint8_t* d_payload_[2] = {nullptr, nullptr};
    uint8_t* d_rgb8_[2] = {nullptr, nullptr};
    float* d_grey_[2] = {nullptr, nullptr};
    float* d_rgbf_[2] = {nullptr, nullptr};
    int* d_box_[2] = {nullptr, nullptr};
    uint8_t* d_rgb8_b_ = nullptr;
    float* d_grey_b_ = nullptr;
    float* d_rgbf_b_ = nullptr;
    float* d_traw_ = nullptr;
    float* d_tz_ = nullptr;
    TemplateStats* d_tstats_ = nullptr;
    double* d_scores_ = nullptr;
    ArgmaxPair* d_am_partial_ = nullptr;
    ArgmaxPair* d_am_ = nullptr;
    PeakStats* d_pk_partial_ = nullptr;
    PeakStats* d_pk_ = nullptr;
    uint64_t* d_chunks_ = nullptr;
    uint64_t* d_hash_ = nullptr;
    unsigned long long* d_change_ = nullptr;
    int* d_patch_flag_ = nullptr;
    uint8_t* d_patch_ = nullptr;
    ObservationEnvelope* d_env_ = nullptr;
    AgentState* d_state_ = nullptr;
    Memory* d_memory_ = nullptr;
    CommittedAction* d_slots_ = nullptr;
    DecisionTrace* d_trace_ = nullptr;
    uint32_t* d_trace_count_ = nullptr;
    unsigned long long* d_stamp_ = nullptr;
    uint8_t* d_planner_img_ = nullptr;   // the frame at B upscaled for the in-process planner
    uint8_t* d_grounder_img_ = nullptr;  // the newest frame upscaled for a grounder at D
    LedgerRow* d_ledger_ = nullptr;
    uint32_t* d_ledger_count_ = nullptr;
    LedgerRow* h_ledger_ = nullptr;
    uint32_t* h_ledger_count_ = nullptr;
    uint32_t planner_scale_ = 2;
    // probe buffers (chronotope_track): two frames' planes and a private state
    uint8_t* d_probe_payload_[2] = {nullptr, nullptr};
    uint8_t* d_probe_rgb8_[2] = {nullptr, nullptr};
    float* d_probe_grey_[2] = {nullptr, nullptr};
    float* d_probe_rgbf_[2] = {nullptr, nullptr};
    AgentState* d_probe_state_ = nullptr;
    float* d_probe_traw_ = nullptr;
    float* d_probe_tz_ = nullptr;
    TemplateStats* d_probe_tstats_ = nullptr;
    // pinned host mirrors (log reads after events; the envelope publication)
    ObservationEnvelope* h_env_ = nullptr;
    AgentState* h_state_ = nullptr;
    CommittedAction* h_slot_ = nullptr;
    Memory* h_memory_ = nullptr;
    DecisionTrace* h_trace_ = nullptr;
    uint32_t* h_trace_count_ = nullptr;
    ArgmaxPair* h_am_ = nullptr;
    PeakStats* h_pk_ = nullptr;
    TemplateStats* h_tstats_ = nullptr;
    unsigned long long* h_stamp_ = nullptr;

    int cur_ = 0;                 // slot of the newest admitted frame
    bool have_prev_ = false;
    uint64_t last_obs_id_ = 0;
    uint64_t admitted_ = 0;
    bool decision_open_ = false, proposal_active_ = false;
    uint64_t decision_counter_ = 0;
    int64_t clock_offset_ = 0, clock_latency_ = 0;
    Audit audit_;
    std::vector<ObservationEnvelope> envelopes_;   // the host log of every admitted envelope (§7.2)
};

int64_t host_now_ns();

}  // namespace chronotope
