// state.cuh — the vocabulary of the kernelized GUI-using agent (kGUA). Data, not behaviour.
//
// If the agent's situation cannot be explained by looking at AgentState, something is wrong (design §3).
// Every struct here is plain data shared by device and host code. The contract names come from
// experiments.md: ObservationEnvelope (§7), ControlState (§8), the one typed pre-commit state — the
// Proposal with its open click coordinate and the template that resolves it (§9.3) — the write-once
// CommittedAction (§9.6), Memory (design §6) and the DecisionTrace ring (design §15).
#pragma once
#include <cstddef>
#include <cstdint>

namespace chronotope {

inline constexpr const char* BINDER_VERSION = "chronotope-1";   // the resident port of chronotope/tracker.py v2
inline constexpr const char* CORE_VERSION = "chronotope-core-1";

enum PixFmt : uint32_t { PIX_RGB24 = 0, PIX_GREY8 = 1, PIX_NV12 = 2 };
// §9.2 ActionKind {PressKey, Click, TypeText, Wait}; None marks "no action this decision".
enum ActionKind : uint32_t { ACT_NONE = 0, ACT_CLICK = 1, ACT_TYPE = 2, ACT_KEY = 3, ACT_WAIT = 4 };
// at-commit policies of chronotope.controller.VLMLateBindingController
enum Policy : uint32_t { POL_FROZEN = 0, POL_LATE_BIND = 1, POL_RESTART_IF_MOVED = 2, POL_LATE_BIND_OR_RESTART = 3 };
// §8: what a control-state update is relevant to
enum Relevance : uint32_t { REL_NONE = 0, REL_ACTION = 1, REL_VALIDITY = 2, REL_PENDING = 3 };
enum ProposalStatus : uint32_t { PS_IDLE = 0, PS_OPEN = 1, PS_PROPOSED = 2, PS_NO_ACTION = 3, PS_COMMITTED = 4, PS_DISCARDED = 5 };
// outcome of the commit kernel (§9.5). Restart* = the proposal is discarded and a NEW decision begins.
enum CommitStatus : uint32_t {
    CS_NOT_ATTEMPTED = 0, CS_COMMITTED = 1, CS_FAILED_SCHEMA = 2, CS_FAILED_STATE_CHANGED = 3,
    CS_FAILED_IMMUTABLE = 4, CS_RESTART_MOVED = 5, CS_RESTART_NOT_FOUND = 6, CS_RESTART_KIND = 7,
    CS_REJECTED_DUPLICATE = 8, CS_REJECTED_REPEATED = 9, CS_NO_PROPOSAL = 10, CS_RESTART_STALE_EVIDENCE = 11
};
enum Binding : uint32_t { BIND_STATE_AT_B = 0, BIND_TRACKED_AT_D = 1, BIND_HOST_OVERRIDE = 2, BIND_GROUNDER = 3, BIND_ORACLE = 4,
                          BIND_DETECTED_AT_D = 5 };   // the detector located the target on the newest frame at D (2026-08-27)
// the explicit decision rule at D over (original proposal, detector location, tracked template) — cfg.detector_at_d
enum DetectorRule : uint32_t { DET_OFF = 0, DET_FIRST = 1, DET_AFTER_TRACK = 2 };
enum PendingEvent : uint32_t { PE_NONE = 0, PE_CREATED = 1, PE_RESOLVED = 2, PE_EXPIRED = 3 };

// tracker.py v2 constants
inline constexpr double AMBIGUITY_MARGIN = 0.05;
inline constexpr double PLATEAU_TOL = 0.02;
inline constexpr int PLATEAU_MAX = 4;
inline constexpr int FLAT_PLATEAU = 1000000;     // a flat template "matches" everywhere: tracker.py returns 10**6
inline constexpr uint32_t MEMORY_SIZE = 64;      // design §6: a fixed ring
inline constexpr int HASH_CHUNK = 4096;

// Device-usable copy of the public chronotope_config (chronotope_api.h).
struct Config {
    int32_t w, h;
    uint32_t pixfmt;
    int32_t radius;               // template radius (side = 2r + 1)
    double min_score;             // min NCC for a match (double: compared as Python compares)
    uint32_t policy;
    int32_t restart_px;           // restart_if_moved threshold; -1 when unused
    uint8_t target_r, target_g, target_b, has_target;   // the evaluator-only colour detector (scripted planner, kind check)
    int32_t offset_px;            // detector planner: coordinate offset added to x (as the scripted oracle's)
    int32_t movement_px;          // control-state relevance: tracked displacement crossing this advances the version
    uint8_t validate_kind;        // V1: discard a proposal whose kind contradicts the newest frame
    uint8_t rgb_template;         // colour template (balloon_select) instead of grey
    int32_t pending_radius;       // pending-action record radius; timeout 0 = record off
    int64_t pending_timeout_ns;
    uint32_t max_repeat;          // reject a proposal repeated this many times without a visible effect (0 = off)
    uint32_t trace_entries, slots;
    uint8_t track_every_frame;    // bind on every admitted frame (relevance) as well as at D
    uint32_t ledger_rows;         // the event ledger ring (P2)
    int64_t max_evidence_age_ns;  // staleness gate at commit: 0 = off (P2, H-M3)
    uint8_t ack_aware_pending;    // P2 H-M2: a pending click with NO visible receipt after receipt_timeout_ns may be retried
    int64_t receipt_timeout_ns;
    uint32_t detector_at_d;       // DetectorRule: 0 off; 1 detector -> tracked -> original; 2 tracked -> detector -> original
};

struct ObservationEnvelope {      // §7
    uint64_t observation_id, source_event_id;
    int64_t source_time_ns, publication_time_ns;
    uint64_t payload_hash, observation_version;
};

struct Box { int32_t x0, y0, x1, y1; uint8_t present; };

struct Action {
    uint32_t kind;
    int32_t x, y, value;
    float confidence;
    uint64_t observation_id;      // the frame the action was derived from (design §13: every result carries one)
};

struct ControlState {             // §8 (+ the perceived action-relevant field)
    uint64_t control_state_version, latest_observation_id;
    int64_t latest_source_time_ns;
    uint64_t controller_state_hash, pending_actions_hash;
    uint64_t observation_version, payload_hash;
    Box box;                      // colour detector on the newest frame
    uint64_t change_q;            // screen change vs the previous frame, sum |dgrey| * 4096 (exact integer)
    double change_mean;
    int32_t tracked_x, tracked_y; // the proposal's template in the newest frame (when tracking)
    double tracked_score;
    uint8_t tracked_valid;
    int32_t ref_x, ref_y;         // tracked position at the last version advance
    uint64_t version_advanced_on_observation;
    uint32_t last_relevance, last_pending_event;
    int64_t last_admit_device_ns;
    int64_t latest_publication_ns, prev_publication_ns;   // O of the newest and of the previous observation (host clock)
    uint64_t frames_since_row;    // observations admitted since the last ledger row
};

struct SemanticState { uint32_t subgoal, target, desired_state; };   // design §3/§7: resident, filled by a planner

struct TemplateStats { float mean_f; double norm; int32_t n, side, chans; uint8_t flat; };

struct Proposal {                 // the one typed pre-commit state (§9.3)
    uint64_t decision_id;
    Action action;                // kind + proposed (x, y): the click coordinate is the open field
    uint64_t control_state_version;
    uint32_t status;              // ProposalStatus
    int32_t tpl_x0, tpl_y0, tpl_dx, tpl_dy, side, chans;
};

struct DecisionInfo {
    uint64_t decision_id, observation_id_b, supporting_control_state_version;
    int64_t b_device_ns, b_host_ns;
    uint8_t open;
    Box box_b;                    // the detector's box in the frame at B
};

struct BindResult {
    int32_t x, y;                 // where the template's click lands in the newest frame
    double score, second;
    int32_t plateau;
    uint8_t matched, ambiguous, valid;
    double moved_px;
    uint64_t observation_id, control_state_version_at_bind;
    int64_t device_ns;
    uint32_t binding;             // Binding
    int32_t resolved_x, resolved_y;
    uint32_t plan;                // the CommitStatus the policy calls for (CS_COMMITTED or a restart)
    int64_t evidence_age_ns, evidence_bracket_ns;   // at commit: D − O_newest; O_newest − e_bracket_lo of its row
    uint32_t ledger_row;          // index of the row the open field was bound on
};

struct PendingAction {            // design: "did the thing I previously did actually work?"
    Action action;
    uint64_t decision_id;
    int64_t decided_ns;
    uint64_t observation_id;
    uint8_t active;
    int32_t px0, py0, px1, py1;   // patch box (rgb8 bytes compared exactly, as PendingMemory._patch)
    int64_t receipt_observed_ns;  // −1 until an observation showed the input was received (echo panel)
};

struct Transition {
    uint64_t observation_before, observation_after;
    Action action;
    float visual_change;
    uint8_t effect_verified, used;
};

struct Memory { Transition ring[MEMORY_SIZE]; uint32_t head, count; PendingAction pending; };

struct CommittedAction {          // §9.6, write-once
    uint64_t action_id, decision_id;
    uint32_t kind;
    int32_t x, y, value;
    uint64_t serialized_action_hash, supporting_control_state_version, control_state_version_at_commit;
    uint64_t observation_id_at_d, proposal_observation_id;
    int64_t commitment_device_ns;
    uint32_t committed, binding;
};

struct CommitScratch { uint32_t status; uint64_t action_id; int64_t d_device_ns; uint32_t reject_reason, repetition_count; };

struct DecisionTrace {            // design §15: one small fixed structure per event, in a ring
    uint64_t observation_id, decision_id;
    Action proposed;
    uint8_t accepted;
    uint32_t reason;              // CommitStatus (0 = an observation while a proposal was open)
    float screen_change;
    uint32_t repetition_count;
    int64_t device_ns;
    uint64_t control_state_version;
};

struct AgentState {               // the truth
    ControlState control;
    SemanticState semantic;
    DecisionInfo decision;
    Proposal proposal;
    BindResult bind;
    CommitScratch commit;
    uint32_t repetition_count, failure_count;
    uint8_t need_replan, task_complete;
    uint64_t next_action_id;
};

// P2: the event ledger — the four-clock trace as the agent's own memory. One row per OBSERVED state change (a
// control-state version advance). E is not observable: the change happened in (e_bracket_lo_ns, o_ns]. A is not
// observable either: effect_observed_ns is the first O at which the effect of the action bound on this row was seen.
struct LedgerRow {
    uint64_t observation_id;
    int64_t o_ns, o_device_ns, source_time_ns, e_bracket_lo_ns;
    uint32_t frames_since_prev, relevance;
    uint64_t control_state_version, decision_proposed, decision_bound;
    int64_t d_device_ns, effect_observed_ns, receipt_observed_ns;   // −1 until seen
};

struct ArgmaxPair { double score; int32_t index; };
struct PeakStats { int32_t plateau; double second; };

}  // namespace chronotope
