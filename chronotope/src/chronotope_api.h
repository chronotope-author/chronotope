/* chronotope_api.h — the C ABI of the kGUA control core (libchronotope.so).
 *
 * One core = one agent. The host publishes observations (O), opens a decision (B), lets the resident
 * detector plan or hands over an external planner's proposal, and asks for resolve+commit (D). Every
 * host call is audited: reads of decision state inside the reactive window (B -> D) and host-side
 * action computation are counted as residency violations (experiments.md §5.3, §15.4); log reads
 * after an event has completed are allowed (§5.2) and counted separately.
 */
#ifndef CHRONOTOPE_API_H
#define CHRONOTOPE_API_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chronotope_core chronotope_core;

enum { CHRONOTOPE_PIX_RGB24 = 0, CHRONOTOPE_PIX_GREY8 = 1, CHRONOTOPE_PIX_NV12 = 2 };
enum { CHRONOTOPE_ACT_NONE = 0, CHRONOTOPE_ACT_CLICK = 1, CHRONOTOPE_ACT_TYPE = 2, CHRONOTOPE_ACT_KEY = 3, CHRONOTOPE_ACT_WAIT = 4 };
enum { CHRONOTOPE_POL_FROZEN = 0, CHRONOTOPE_POL_LATE_BIND = 1, CHRONOTOPE_POL_RESTART_IF_MOVED = 2, CHRONOTOPE_POL_LATE_BIND_OR_RESTART = 3 };
enum { CHRONOTOPE_REL_NONE = 0, CHRONOTOPE_REL_ACTION = 1, CHRONOTOPE_REL_VALIDITY = 2, CHRONOTOPE_REL_PENDING = 3 };
enum { CHRONOTOPE_CS_NOT_ATTEMPTED = 0, CHRONOTOPE_CS_COMMITTED = 1, CHRONOTOPE_CS_FAILED_SCHEMA = 2, CHRONOTOPE_CS_FAILED_STATE_CHANGED = 3,
       CHRONOTOPE_CS_FAILED_IMMUTABLE = 4, CHRONOTOPE_CS_RESTART_MOVED = 5, CHRONOTOPE_CS_RESTART_NOT_FOUND = 6, CHRONOTOPE_CS_RESTART_KIND = 7,
       CHRONOTOPE_CS_REJECTED_DUPLICATE = 8, CHRONOTOPE_CS_REJECTED_REPEATED = 9, CHRONOTOPE_CS_NO_PROPOSAL = 10, CHRONOTOPE_CS_RESTART_STALE_EVIDENCE = 11 };
enum { CHRONOTOPE_BIND_STATE_AT_B = 0, CHRONOTOPE_BIND_TRACKED_AT_D = 1, CHRONOTOPE_BIND_HOST_OVERRIDE = 2, CHRONOTOPE_BIND_GROUNDER = 3, CHRONOTOPE_BIND_ORACLE = 4, CHRONOTOPE_BIND_DETECTED_AT_D = 5 };
enum { CHRONOTOPE_PE_NONE = 0, CHRONOTOPE_PE_CREATED = 1, CHRONOTOPE_PE_RESOLVED = 2, CHRONOTOPE_PE_EXPIRED = 3 };
enum { CHRONOTOPE_OK = 0, CHRONOTOPE_ERR = -1, CHRONOTOPE_STALE = 1, CHRONOTOPE_NO_DECISION = 2 };

typedef struct {
    uint32_t width, height, pixfmt;
    uint32_t template_radius_px;
    double min_match_score;
    uint32_t policy;
    int32_t restart_threshold_px;      /* restart_if_moved only; -1 otherwise */
    uint8_t target_rgb[3];
    uint8_t has_target_color;          /* the evaluator-only colour detector (detector planner, kind check) */
    int32_t coordinate_offset_px;
    uint32_t movement_threshold_px;    /* tracked displacement that advances the control-state version (0 = off) */
    uint8_t validate_kind;
    uint8_t rgb_template;
    uint32_t pending_radius_px;
    int64_t pending_timeout_ns;        /* 0 = no pending-action record */
    uint32_t max_repeat_without_effect;
    uint32_t trace_entries;            /* DecisionTrace ring (default 256 when 0) */
    uint32_t committed_slots;          /* CommittedAction ring (default 64 when 0) */
    uint8_t track_every_frame;         /* bind on every admitted frame for the relevance rule (default 1) */
    uint32_t planner_image_scale;      /* the in-process planner's frame upscale factor (0 = 2): buffer preallocated */
    uint32_t ledger_rows;              /* P2: the event ledger ring (0 = 256) */
    int64_t max_evidence_age_ns;       /* P2: staleness gate at commit (0 = off) */
    uint8_t ack_aware_pending;         /* P2: a pending click with no VISIBLE receipt after receipt_timeout_ns may be retried */
    int64_t receipt_timeout_ns;
    uint32_t detector_at_d;            /* 2026-08-27: the detector as binder at D: 0 off, 1 detector->tracked->original, 2 tracked->detector->original */
} chronotope_config;

typedef struct {
    uint32_t status;                   /* CHRONOTOPE_OK / CHRONOTOPE_STALE */
    uint32_t relevance;                /* CHRONOTOPE_REL_* */
    uint64_t control_state_version, observation_version, payload_hash, controller_state_hash, pending_actions_hash;
    double change_mean;
    int32_t box[4];                    /* x0, y0, x1, y1 */
    uint8_t box_present;
    int32_t tracked_x, tracked_y;
    double tracked_score;
    uint8_t tracked_valid, pending_active;
    uint32_t pending_event;
    int64_t admit_device_ns, admit_host_ns;
} chronotope_admission;

typedef struct {
    uint32_t status;                   /* 0 = no proposal (wait / no action), 1 = proposed */
    uint32_t kind;
    int32_t x, y, value;
    uint64_t observation_id, decision_id, supporting_control_state_version;
    int64_t b_device_ns, b_host_ns;
} chronotope_proposal;

typedef struct {
    uint32_t outcome;                  /* CHRONOTOPE_CS_* */
    uint32_t kind;
    int32_t x, y, value;               /* the committed action (when outcome == COMMITTED) */
    int32_t proposed_x, proposed_y;
    uint64_t action_id, decision_id, serialized_action_hash, supporting_control_state_version, control_state_version_at_commit;
    uint64_t observation_id_at_d, proposal_observation_id;
    int64_t d_device_ns, d_device_mapped_ns, d_host_ns;
    double match_score, second_best, moved_px;
    int32_t plateau;
    uint8_t matched, ambiguous, bind_valid;
    uint32_t binding;                  /* CHRONOTOPE_BIND_* */
    int64_t track_ns, track_device_ns; /* host wall (charged to D) and device time from bind start to the D stamp */
    uint32_t reject_reason, repetition_count;
    int32_t bind_x, bind_y;            /* where the template was found (bind_valid) */
    uint8_t pending_created;
    int64_t evidence_age_ns;           /* P2: D − O of the newest observation (host clock) */
    int64_t evidence_bracket_ns;       /* P2: O − e_bracket_lo of the bound row (−1 when unknown) */
    uint32_t ledger_row;
} chronotope_commit_result;

typedef struct {
    uint64_t h2d_publications, h2d_proposals;
    uint64_t d2h_log_in_window, d2h_state_in_window;
    uint64_t host_syncs_log_in_window, host_syncs_action_in_window, declared_commit_waits;
    uint64_t host_action_computations, mallocs_after_warmup, kernel_launches_in_window, windows_opened;
    uint8_t window_open, residency_ok;
    int64_t clock_offset_ns, clock_calibration_latency_ns;
    size_t device_bytes_allocated;
    uint64_t planner_host_steps;       /* the in-process planner's per-token host synchronizations (all requests) */
    uint64_t planner_d2h_pixels;       /* frames copied to the host for a planner (0 on the device-frame path) */
    uint64_t planner_requests;
    uint64_t binder_host_steps;        /* P2: bindings whose answer crossed host memory (the grounder) */
} chronotope_audit_report;

typedef struct {
    uint64_t observation_id, decision_id;
    uint32_t kind; int32_t x, y;
    uint8_t accepted;
    uint32_t reason;
    float screen_change;
    uint32_t repetition_count;
    int64_t device_ns, device_mapped_ns;
    uint64_t control_state_version;
} chronotope_trace_entry;

typedef struct {                       /* P2: one row of the event ledger (the agent's own four-clock row) */
    uint64_t observation_id;
    int64_t o_ns, o_device_ns, source_time_ns, e_bracket_lo_ns;   /* the change happened in (e_bracket_lo, o] */
    uint32_t frames_since_prev, relevance;
    uint64_t control_state_version, decision_proposed, decision_bound;
    int64_t d_device_ns, effect_observed_ns, receipt_observed_ns;   /* −1 = never observed */
} chronotope_ledger_row;

typedef struct {
    int32_t x, y;                      /* where the click lands in frame_d */
    double score, second;
    int32_t plateau;
    uint8_t matched, ambiguous, flat;
    double moved_px;
    int32_t tpl_x0, tpl_y0, tpl_dx, tpl_dy;
    int64_t device_ns;
} chronotope_match;

const char* chronotope_version(void);
chronotope_core* chronotope_create(const chronotope_config* cfg, char* err, size_t errlen);
void chronotope_destroy(chronotope_core* core);
const char* chronotope_last_error(chronotope_core* core);

/* O: publish and admit one observation (the only host -> device transfer of the reactive path). */
int chronotope_admit(chronotope_core* core, const uint8_t* payload, size_t len, uint64_t observation_id, uint64_t source_event_id,
               int64_t source_time_ns, int64_t publication_time_ns, chronotope_admission* out);
/* B: a decision begins on the newest admitted observation (its planes are pinned on the device). */
int chronotope_begin_decision(chronotope_core* core, uint64_t observation_id, chronotope_proposal* out);
/* the resident planner: click the detected target's centre in the frame at B */
int chronotope_decide_detector(chronotope_core* core, chronotope_proposal* out);
/* the external planner's answer (frame-at-B coordinates); kind WAIT/NONE = no action this decision */
int chronotope_propose(chronotope_core* core, uint32_t kind, int32_t x, int32_t y, int32_t value, chronotope_proposal* out);
int chronotope_abort_decision(chronotope_core* core);
/* D: resolve the open field on the newest frame, validate, commit — one stream-ordered sequence on the device */
int chronotope_resolve_and_commit(chronotope_core* core, chronotope_commit_result* out);
int chronotope_resolve(chronotope_core* core, chronotope_commit_result* out);        /* bind only (contract tests) */
int chronotope_commit_only(chronotope_core* core, chronotope_commit_result* out);    /* commit after a separate resolve (contract tests) */
int chronotope_restart(chronotope_core* core);                                  /* discard the proposal; the next decision is new */
int chronotope_decision_open(chronotope_core* core);
/* the NEGATIVE path (never on the resident path): the host reads the newest frame and determines the binding itself */
int chronotope_read_frame(chronotope_core* core, uint8_t* buf, size_t len);
int chronotope_override_binding(chronotope_core* core, int32_t x, int32_t y);
/* after D */
int chronotope_audit(chronotope_core* core, chronotope_audit_report* out);
int chronotope_trace(chronotope_core* core, chronotope_trace_entry* buf, uint32_t n, uint32_t* count);
int chronotope_state_hash(chronotope_core* core, uint64_t* controller_state_hash, uint64_t* control_state_version);
/* the binder alone, on two frames (the U-K2 replay against tracker.py) */
int chronotope_track(chronotope_core* core, const uint8_t* frame_b, const uint8_t* frame_d, size_t len, int32_t x, int32_t y,
               uint32_t radius, uint8_t rgb, chronotope_match* out);
/* the in-process planner (chronotope/src/planner_trt): the frame at B upscaled on the planner stream into a preallocated
 * device buffer; the pointer is consumed by the planner in this process and never read by the host */
int chronotope_frame_b_device(chronotope_core* core, uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h);
/* K4d (2026-08-27): the in-process TensorRT Edge-LLM planner INSIDE the library (no socket, no server): the referee that
 * loads libchronotope.so through ctypes gets the same model call the `chronotope` binary makes. Compiled only with
 * CHRONOTOPE_EDGELLM_BUILD; otherwise open() returns CHRONOTOPE_ERR with "not compiled". */
typedef struct {
    const char* engine_dir;             /* llm_build output */
    const char* multimodal_engine_dir;  /* visual_build output root */
    int32_t prompt_version, image_scale, max_tokens;
    double temperature;
    uint8_t enable_thinking, capture_cuda_graph;
} chronotope_trt_config;
typedef struct {
    uint8_t answer_ok, click;           /* answer_ok: the request completed and parsed; click: a click was proposed (else wait/unparseable) */
    int32_t x, y, image_x, image_y;     /* canvas px; as answered */
    int64_t t_start_ns, t_end_ns;       /* CLOCK_MONOTONIC */
    double prefill_ms, decode_ms;
    uint64_t host_steps, requests;      /* the runtime's per-token host syncs of this request; requests so far */
    int32_t input_tokens, output_tokens;
    int32_t propose_rc;                 /* rc of chronotope_propose when click (else -1) */
    char raw_text[512];
    char error[256];
    chronotope_proposal proposal;       /* valid when click and propose_rc == CHRONOTOPE_OK */
} chronotope_trt_answer;
int chronotope_planner_trt_open(chronotope_core* core, const chronotope_trt_config* cfg);   /* loads the engines on the planner stream */
int chronotope_planner_trt_close(chronotope_core* core);
int chronotope_decide_trt(chronotope_core* core, chronotope_trt_answer* out);              /* blocks: the model on the frame at B (device pointer) */
void* chronotope_planner_stream(chronotope_core* core);          /* cudaStream_t of the planner stream */
int chronotope_note_planner(chronotope_core* core, uint64_t host_steps, uint64_t d2h_pixels);   /* audit accounting of one planner request */
/* P2: the event ledger (a log read after D), the external binder, and a VISIBLE receipt */
int chronotope_ledger(chronotope_core* core, chronotope_ledger_row* buf, uint32_t n, uint32_t* count);
int chronotope_bind_external(chronotope_core* core, int32_t x, int32_t y, uint32_t binding);   /* CHRONOTOPE_BIND_GROUNDER | CHRONOTOPE_BIND_ORACLE */
int chronotope_observe_receipt(chronotope_core* core, uint64_t decision_id, int64_t observation_publication_ns);
int chronotope_frame_newest_device(chronotope_core* core, uint32_t scale, const uint8_t** ptr, uint32_t* w, uint32_t* h);   /* for a grounder at D */
/* contract-test hooks */
int chronotope_test_recommit(chronotope_core* core, chronotope_commit_result* out);
int chronotope_test_alloc(chronotope_core* core);

#ifdef __cplusplus
}
#endif
#endif
