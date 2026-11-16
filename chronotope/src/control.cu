// control.cu — the control kernels. Single thread each: the state is small and the point is that the
// host never touches it between O and D.
#include "control.cuh"
#include <climits>

namespace chronotope::control {

namespace {

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint64_t mix(uint64_t h, uint64_t v) {
    for (int k = 0; k < 8; ++k) { h ^= (v >> (8 * k)) & 0xFF; h *= 1099511628211ULL; }
    return h;
}

__device__ __forceinline__ double clamp1(double v) { return fmin(1.0, fmax(-1.0, v)); }

__device__ void trace_append(DecisionTrace* ring, uint32_t* count, Config cfg, const DecisionTrace& e) {
    uint32_t i = (*count) % cfg.trace_entries;
    ring[i] = e;
    *count += 1;
}

// tracker.py v2 verdict from the NCC results for the current proposal
__device__ void bind_from_ncc(const AgentState* st, const ArgmaxPair* am, const PeakStats* pk, const TemplateStats* ts, int Wv,
                              Config cfg, int* mx, int* my, double* score, double* second, int* plateau,
                              uint8_t* matched, uint8_t* ambiguous, double* moved) {
    const Proposal& p = st->proposal;
    if (ts->flat) {  // a template cut from a flat region cannot be tracked: tracker.py returns (dx, dy, 0, 1, 10**6)
        *mx = p.tpl_dx; *my = p.tpl_dy; *score = 0.0; *second = 1.0; *plateau = FLAT_PLATEAU;
    } else {
        int iy = am->index / Wv, ix = am->index % Wv;
        *mx = ix + p.tpl_dx; *my = iy + p.tpl_dy;
        *score = clamp1(am->score); *second = pk->second; *plateau = pk->plateau;
    }
    *ambiguous = (*second >= *score - AMBIGUITY_MARGIN || *plateau > PLATEAU_MAX) ? 1 : 0;
    *matched = (*score >= cfg.min_score && !*ambiguous) ? 1 : 0;
    double dx = (double)(*mx - p.action.x), dy = (double)(*my - p.action.y);
    *moved = *matched ? sqrt(dx * dx + dy * dy) : 0.0;
}

__global__ void k_admit_update(AgentState* st, Memory* mem, const ObservationEnvelope* env, const uint64_t* hash,
                               const unsigned long long* change, const int* box4, const ArgmaxPair* am, const PeakStats* pk,
                               const TemplateStats* ts, const int* patch_changed, Config cfg, int have_prev, int n_pixels,
                               int64_t host_now_ns, int Wv, DecisionTrace* ring, uint32_t* ring_count, LedgerRow* ledger,
                               uint32_t* ledger_count) {
    ControlState& c = st->control;
    c.prev_publication_ns = c.latest_publication_ns;
    c.latest_publication_ns = env->publication_time_ns;
    c.frames_since_row += 1;
    uint64_t prev_hash = c.payload_hash;
    Box prev_box = c.box;
    c.observation_version += 1;
    c.latest_observation_id = env->observation_id;
    c.latest_source_time_ns = env->source_time_ns;
    c.payload_hash = *hash;
    c.change_q = have_prev ? *change : 0ULL;
    c.change_mean = have_prev ? ((double)c.change_q / 4096.0 / (double)n_pixels) : 0.0;
    c.last_admit_device_ns = (int64_t)gtimer();
    Box nb;
    nb.present = (cfg.has_target && box4[2] >= 0) ? 1 : 0;
    nb.x0 = nb.present ? box4[0] : 0; nb.y0 = nb.present ? box4[1] : 0;
    nb.x1 = nb.present ? box4[2] + 1 : 0; nb.y1 = nb.present ? box4[3] + 1 : 0;
    c.box = nb;
    uint32_t rel = REL_NONE;
    if (cfg.has_target) {
        bool changed = (nb.present != prev_box.present) || (nb.present && (nb.x0 != prev_box.x0 || nb.y0 != prev_box.y0 ||
                                                                         nb.x1 != prev_box.x1 || nb.y1 != prev_box.y1));
        if (!have_prev || changed) rel = REL_ACTION;
    } else if (!have_prev || c.payload_hash != prev_hash) {
        rel = REL_ACTION;   // no semantic model: any pixel change counts (as VLMDirectController.admit)
    }
    // tracked field: the proposal's template followed in every admitted frame (streaming binder)
    c.tracked_valid = 0;
    if (st->proposal.status == PS_PROPOSED && cfg.track_every_frame && cfg.policy != POL_FROZEN && am != nullptr) {
        int mx, my, plateau; double score, second, moved; uint8_t matched, ambiguous;
        bind_from_ncc(st, am, pk, ts, Wv, cfg, &mx, &my, &score, &second, &plateau, &matched, &ambiguous, &moved);
        c.tracked_x = mx; c.tracked_y = my; c.tracked_score = score; c.tracked_valid = matched;
        if (matched && cfg.movement_px > 0) {
            double dx = (double)(mx - c.ref_x), dy = (double)(my - c.ref_y);
            if (sqrt(dx * dx + dy * dy) > (double)cfg.movement_px) { rel = REL_ACTION; c.ref_x = mx; c.ref_y = my; }
        }
    }
    // verify the pending action (design §8 verify_pending_action): resolved when its patch changed, expired by timeout
    uint32_t pev = PE_NONE;
    PendingAction& pa = mem->pending;
    if (pa.active) {
        if (host_now_ns - pa.decided_ns >= cfg.pending_timeout_ns) { pa.active = 0; pev = PE_EXPIRED; }
        else if (patch_changed && *patch_changed) {
            pa.active = 0; pev = PE_RESOLVED;
            // mark the transition of that decision as verified
            for (uint32_t k = 0; k < MEMORY_SIZE; ++k) {
                Transition& t = mem->ring[k];
                if (t.used && t.action.observation_id == pa.action.observation_id && t.action.x == pa.action.x && t.action.y == pa.action.y) {
                    t.effect_verified = 1; t.observation_after = env->observation_id; t.visual_change = (float)c.change_mean;
                }
            }
        }
        if (pev == PE_RESOLVED) {   // the ledger's effect column: the row whose decision this effect belongs to
            for (uint32_t k = 0; k < cfg.ledger_rows && k < *ledger_count; ++k) {
                LedgerRow& row = ledger[(*ledger_count - 1 - k) % cfg.ledger_rows];
                if (row.decision_bound == pa.decision_id) { row.effect_observed_ns = env->publication_time_ns; break; }
            }
        }
        if (pev != PE_NONE && rel == REL_NONE) rel = REL_PENDING;
    }
    if (rel != REL_NONE) {
        c.control_state_version += 1; c.version_advanced_on_observation = env->observation_id;
        // P2: one ledger row per observed state change — the agent's own four-clock row. E is a bracket.
        LedgerRow& row = ledger[(*ledger_count) % cfg.ledger_rows];
        row = LedgerRow{};
        row.observation_id = env->observation_id; row.o_ns = env->publication_time_ns; row.o_device_ns = c.last_admit_device_ns;
        row.source_time_ns = env->source_time_ns; row.e_bracket_lo_ns = have_prev ? c.prev_publication_ns : -1;
        row.frames_since_prev = (uint32_t)c.frames_since_row; row.relevance = rel; row.control_state_version = c.control_state_version;
        row.effect_observed_ns = -1; row.receipt_observed_ns = -1;
        *ledger_count += 1;
        c.frames_since_row = 0;
    }
    c.last_relevance = rel; c.last_pending_event = pev;
    uint64_t hsh = 14695981039346656037ULL;
    hsh = mix(hsh, (uint64_t)nb.present); hsh = mix(hsh, (uint64_t)(uint32_t)nb.x0); hsh = mix(hsh, (uint64_t)(uint32_t)nb.y0);
    hsh = mix(hsh, (uint64_t)(uint32_t)nb.x1); hsh = mix(hsh, (uint64_t)(uint32_t)nb.y1);
    if (!cfg.has_target) hsh = mix(hsh, c.payload_hash);
    if (c.tracked_valid) { hsh = mix(hsh, (uint64_t)(uint32_t)c.tracked_x); hsh = mix(hsh, (uint64_t)(uint32_t)c.tracked_y); }
    uint64_t ph = 14695981039346656037ULL;
    ph = mix(ph, (uint64_t)pa.active);
    if (pa.active) { ph = mix(ph, (uint64_t)(uint32_t)pa.action.x); ph = mix(ph, (uint64_t)(uint32_t)pa.action.y); ph = mix(ph, pa.decision_id); }
    c.controller_state_hash = mix(hsh, ph);
    c.pending_actions_hash = ph;
    if (st->proposal.status == PS_PROPOSED) {   // an observation while a proposal is open: one trace entry (design §15)
        DecisionTrace e{};
        e.observation_id = env->observation_id; e.decision_id = st->decision.decision_id; e.proposed = st->proposal.action;
        e.accepted = 0; e.reason = 0; e.screen_change = (float)c.change_mean; e.repetition_count = st->repetition_count;
        e.device_ns = c.last_admit_device_ns; e.control_state_version = c.control_state_version;
        trace_append(ring, ring_count, cfg, e);
    }
}

__global__ void k_begin(AgentState* st, uint64_t decision_id, uint64_t obs_id, int64_t host_b_ns, LedgerRow* ledger, const uint32_t* ledger_count, Config cfg) {
    DecisionInfo& d = st->decision;
    if (*ledger_count > 0) ledger[(*ledger_count - 1) % cfg.ledger_rows].decision_proposed = decision_id;
    d.decision_id = decision_id; d.observation_id_b = obs_id;
    d.supporting_control_state_version = st->control.control_state_version;
    d.b_device_ns = (int64_t)gtimer(); d.b_host_ns = host_b_ns; d.open = 1; d.box_b = st->control.box;
    Proposal& p = st->proposal;
    p.decision_id = decision_id; p.status = PS_OPEN; p.control_state_version = st->control.control_state_version;
    p.action = Action{ACT_NONE, 0, 0, 0, 0.0f, obs_id};
    st->bind = BindResult{};
    st->commit = CommitScratch{};
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__global__ void k_decide_detector(AgentState* st, Config cfg) {
    DecisionInfo& d = st->decision;
    Proposal& p = st->proposal;
    if (!d.open || p.status != PS_OPEN) return;
    if (!d.box_b.present) { p.status = PS_NO_ACTION; return; }   // nothing to click: no action this decision (wait)
    int cx = (d.box_b.x0 + d.box_b.x1) / 2, cy = (d.box_b.y0 + d.box_b.y1) / 2;   // the scripted oracle's centre
    p.action.kind = ACT_CLICK;
    p.action.x = clampi(cx + cfg.offset_px, 0, cfg.w - 1);
    p.action.y = clampi(cy, 0, cfg.h - 1);
    p.action.value = 0; p.action.confidence = 1.0f; p.action.observation_id = d.observation_id_b;
    p.status = PS_PROPOSED;
    st->control.ref_x = p.action.x; st->control.ref_y = p.action.y;
}

__global__ void k_set_proposal(AgentState* st, Action a) {
    DecisionInfo& d = st->decision;
    Proposal& p = st->proposal;
    if (!d.open || p.status != PS_OPEN) return;
    if (a.kind == ACT_WAIT || a.kind == ACT_NONE) { p.status = PS_NO_ACTION; return; }
    a.observation_id = d.observation_id_b;
    p.action = a; p.status = PS_PROPOSED;
    st->control.ref_x = a.x; st->control.ref_y = a.y;
}

__global__ void k_resolve(AgentState* st, const ArgmaxPair* am, const PeakStats* pk, const TemplateStats* ts, const int* box_newest,
                          Config cfg, int use_ncc, int Wv) {
    Proposal& p = st->proposal;
    BindResult& b = st->bind;
    b = BindResult{};
    b.observation_id = st->control.latest_observation_id;
    b.control_state_version_at_bind = st->control.control_state_version;
    b.device_ns = (int64_t)gtimer();
    b.binding = BIND_STATE_AT_B; b.resolved_x = p.action.x; b.resolved_y = p.action.y; b.plan = CS_COMMITTED;
    if (!st->decision.open || p.status != PS_PROPOSED) { b.plan = CS_NO_PROPOSAL; return; }
    if (cfg.validate_kind && cfg.has_target) {   // V1: the kind first, then the field
        uint32_t expected = (box_newest[2] >= 0) ? ACT_CLICK : ACT_WAIT;
        if (p.action.kind != expected) { b.plan = CS_RESTART_KIND; return; }
    }
    if (cfg.policy == POL_FROZEN || !use_ncc) return;      // commit the proposal (state at B)
    int mx, my, plateau; double score, second, moved; uint8_t matched, ambiguous;
    bind_from_ncc(st, am, pk, ts, Wv, cfg, &mx, &my, &score, &second, &plateau, &matched, &ambiguous, &moved);
    b.x = mx; b.y = my; b.score = score; b.second = second; b.plateau = plateau;
    b.matched = matched; b.ambiguous = ambiguous; b.moved_px = moved; b.valid = 1;
    if (cfg.detector_at_d != DET_OFF && cfg.has_target && (cfg.policy == POL_LATE_BIND || cfg.policy == POL_LATE_BIND_OR_RESTART)) {
        // the detector as binder (2026-08-27): the colour/feature detector located the target on the NEWEST frame
        // (box_newest, computed at admission); the rule over (original, detector, tracked) is explicit and logged.
        bool found = box_newest[2] >= 0;
        int dx = clampi((box_newest[0] + box_newest[2] + 1) / 2 + cfg.offset_px, 0, cfg.w - 1);   // as the admission Box (x1 = max + 1)
        int dy = clampi((box_newest[1] + box_newest[3] + 1) / 2, 0, cfg.h - 1);
        bool use_det = cfg.detector_at_d == DET_FIRST ? found : (found && !matched);
        if (use_det) {
            b.resolved_x = dx; b.resolved_y = dy; b.binding = BIND_DETECTED_AT_D; b.matched = 1; b.ambiguous = 0;
            b.moved_px = sqrt((double)((dx - p.action.x) * (dx - p.action.x) + (dy - p.action.y) * (dy - p.action.y)));
            return;
        }
        if (matched) {   // the tracked template (DET_AFTER_TRACK, or DET_FIRST with no detection)
            b.resolved_x = clampi(mx, 0, cfg.w - 1); b.resolved_y = clampi(my, 0, cfg.h - 1); b.binding = BIND_TRACKED_AT_D; return;
        }
        if (cfg.policy == POL_LATE_BIND_OR_RESTART) { b.plan = CS_RESTART_NOT_FOUND; return; }
        return;         // neither: the original proposal, recorded state_at_B with matched = 0
    }
    if (cfg.policy == POL_RESTART_IF_MOVED) {
        if (matched && moved > (double)cfg.restart_px) b.plan = CS_RESTART_MOVED;
        return;   // else commit the proposal unchanged
    }
    if (cfg.policy == POL_LATE_BIND_OR_RESTART && !matched) { b.plan = CS_RESTART_NOT_FOUND; return; }
    // late binding: translate the proposal by the template's displacement (the planner's own offset is kept);
    // an unmatched template commits the proposal unchanged, recorded matched = 0
    int x = matched ? mx : p.action.x, y = matched ? my : p.action.y;
    b.resolved_x = clampi(x, 0, cfg.w - 1); b.resolved_y = clampi(y, 0, cfg.h - 1);
    b.binding = BIND_TRACKED_AT_D;
}

__global__ void k_commit(AgentState* st, Memory* mem, CommittedAction* slots, Config cfg, DecisionTrace* ring, uint32_t* ring_count,
                         int64_t host_now_ns, int recommit_test, LedgerRow* ledger, const uint32_t* ledger_count) {
    Proposal& p = st->proposal;
    BindResult& b = st->bind;
    CommitScratch& c = st->commit;
    c = CommitScratch{};
    if (recommit_test) {   // §16 item 10: try to write the last committed record again
        uint64_t id = st->next_action_id - 1;
        CommittedAction* slot = &slots[id % cfg.slots];
        c.action_id = id;
        c.status = (slot->committed && slot->action_id == id) ? CS_FAILED_IMMUTABLE : CS_FAILED_SCHEMA;
        return;
    }
    if (!st->decision.open || p.status != PS_PROPOSED) { c.status = CS_NO_PROPOSAL; return; }
    DecisionTrace e{};
    e.observation_id = st->control.latest_observation_id; e.decision_id = st->decision.decision_id; e.proposed = p.action;
    e.screen_change = (float)st->control.change_mean; e.control_state_version = st->control.control_state_version;
    if (b.plan != CS_COMMITTED) {           // the policy discards the proposal: a NEW decision begins (restart) or the kind was wrong
        c.status = b.plan; p.status = PS_DISCARDED; st->decision.open = 0;
        e.accepted = 0; e.reason = c.status; e.device_ns = (int64_t)gtimer();
        trace_append(ring, ring_count, cfg, e);
        return;
    }
    // step 1: the complete concrete action (all open fields resolved)
    Action a = p.action;
    a.x = b.resolved_x; a.y = b.resolved_y;
    // step 2: the declared action schema
    bool ok = (a.kind == ACT_CLICK || a.kind == ACT_TYPE || a.kind == ACT_KEY);
    if (a.kind == ACT_CLICK) ok = ok && a.x >= 0 && a.x < cfg.w && a.y >= 0 && a.y < cfg.h;
    uint32_t status = ok ? CS_COMMITTED : CS_FAILED_SCHEMA;
    // P2 H-M3: the staleness gate — the newest evidence must be younger than the declared bound, else a new decision
    if (status == CS_COMMITTED && cfg.max_evidence_age_ns > 0 && host_now_ns - st->control.latest_publication_ns > cfg.max_evidence_age_ns)
        status = CS_RESTART_STALE_EVIDENCE;
    // validation (design §8 validate_action): a duplicate of the pending click, a repeat without effect
    uint32_t rep = 0;
    if (status == CS_COMMITTED && a.kind == ACT_CLICK) {
        PendingAction& pa = mem->pending;
        if (cfg.pending_timeout_ns > 0 && pa.active) {
            double dx = (double)(a.x - pa.action.x), dy = (double)(a.y - pa.action.y);
            bool same_spot = sqrt(dx * dx + dy * dy) <= (double)cfg.pending_radius;
            // P2 H-M2 (ack-aware): a pending click whose receipt was never SEEN within receipt_timeout is treated as dropped:
            // the retry is allowed and the old record closes. With a visible receipt the record protects until the effect/expiry.
            bool dropped = cfg.ack_aware_pending && pa.receipt_observed_ns < 0 && cfg.receipt_timeout_ns > 0
                           && (host_now_ns - pa.decided_ns) > cfg.receipt_timeout_ns;
            if (same_spot && !dropped) status = CS_REJECTED_DUPLICATE;
            if (same_spot && dropped) pa.active = 0;
        }
        for (uint32_t k = 0; k < MEMORY_SIZE; ++k) {
            const Transition& t = mem->ring[k];
            if (!t.used || t.effect_verified || t.action.kind != a.kind) continue;
            double dx = (double)(a.x - t.action.x), dy = (double)(a.y - t.action.y);
            if (sqrt(dx * dx + dy * dy) <= (double)cfg.pending_radius) ++rep;
        }
        if (cfg.max_repeat > 0 && rep >= cfg.max_repeat && status == CS_COMMITTED) status = CS_REJECTED_REPEATED;
    }
    c.repetition_count = rep; st->repetition_count = rep;
    // steps 3-4: the control-state version read at binding must still be current (§9.5: not the observation version)
    if (status == CS_COMMITTED && st->control.control_state_version != b.control_state_version_at_bind) status = CS_FAILED_STATE_CHANGED;
    if (status != CS_COMMITTED) {
        c.status = status; c.reject_reason = status; p.status = PS_DISCARDED; st->decision.open = 0; st->failure_count += 1;
        e.accepted = 0; e.reason = status; e.repetition_count = rep; e.device_ns = (int64_t)gtimer();
        trace_append(ring, ring_count, cfg, e);
        return;
    }
    // step 5: the immutable record, write-once per action id
    uint64_t id = st->next_action_id;
    CommittedAction* slot = &slots[id % cfg.slots];
    if (slot->committed && slot->action_id == id) { c.status = CS_FAILED_IMMUTABLE; return; }
    slot->committed = 0;
    slot->action_id = id; slot->decision_id = st->decision.decision_id;
    slot->kind = a.kind; slot->x = a.x; slot->y = a.y; slot->value = a.value;
    uint64_t hsh = 14695981039346656037ULL;
    hsh = mix(hsh, (uint64_t)a.kind); hsh = mix(hsh, (uint64_t)(uint32_t)a.x); hsh = mix(hsh, (uint64_t)(uint32_t)a.y); hsh = mix(hsh, (uint64_t)(uint32_t)a.value);
    slot->serialized_action_hash = hsh;
    slot->supporting_control_state_version = st->decision.supporting_control_state_version;
    slot->control_state_version_at_commit = st->control.control_state_version;
    slot->observation_id_at_d = st->control.latest_observation_id;
    slot->proposal_observation_id = p.action.observation_id;
    slot->binding = b.binding;
    // step 6: D
    slot->commitment_device_ns = (int64_t)gtimer();
    __threadfence();
    atomicExch(&slot->committed, 1u);       // step 7: published; nothing may change its meaning from here
    st->next_action_id = id + 1;
    // memory: what happened (design §6)
    Transition& t = mem->ring[mem->head % MEMORY_SIZE];
    t.observation_before = st->control.latest_observation_id; t.observation_after = 0; t.action = a;
    t.visual_change = 0.0f; t.effect_verified = 0; t.used = 1;
    mem->head = (mem->head + 1) % MEMORY_SIZE; mem->count += 1;
    // the ledger: the row the open field was bound on carries this decision and its D
    {
        BindResult& bb = st->bind;
        uint32_t found = 0xFFFFFFFFu;
        for (uint32_t k = 0; k < cfg.ledger_rows && k < *ledger_count; ++k) {
            uint32_t idx = (*ledger_count - 1 - k) % cfg.ledger_rows;
            if (ledger[idx].observation_id == bb.observation_id) { found = idx; break; }
        }
        if (found == 0xFFFFFFFFu && *ledger_count > 0) found = (*ledger_count - 1) % cfg.ledger_rows;   // the newest row
        if (found != 0xFFFFFFFFu) {
            ledger[found].decision_bound = st->decision.decision_id; ledger[found].d_device_ns = slot->commitment_device_ns;
            bb.ledger_row = found;
            bb.evidence_bracket_ns = ledger[found].e_bracket_lo_ns >= 0 ? (st->control.latest_publication_ns - ledger[found].e_bracket_lo_ns) : -1;
        }
        bb.evidence_age_ns = host_now_ns - st->control.latest_publication_ns;
    }
    if (cfg.pending_timeout_ns > 0 && a.kind == ACT_CLICK) {
        PendingAction& pa = mem->pending;
        pa.action = a; pa.decision_id = st->decision.decision_id; pa.decided_ns = host_now_ns;
        pa.observation_id = st->control.latest_observation_id; pa.active = 1; pa.receipt_observed_ns = -1;
        int r = cfg.pending_radius;
        pa.px0 = max(0, a.x - r); pa.py0 = max(0, a.y - r); pa.px1 = min(cfg.w, a.x + r + 1); pa.py1 = min(cfg.h, a.y + r + 1);
    }
    p.status = PS_COMMITTED; st->decision.open = 0;
    c.status = CS_COMMITTED; c.action_id = id; c.d_device_ns = slot->commitment_device_ns;
    e.accepted = 1; e.reason = CS_COMMITTED; e.repetition_count = rep; e.device_ns = c.d_device_ns;
    e.proposed = a;
    trace_append(ring, ring_count, cfg, e);
}

__global__ void k_override(AgentState* st, int x, int y) {
    BindResult& b = st->bind;
    b.resolved_x = x; b.resolved_y = y; b.binding = BIND_HOST_OVERRIDE; b.plan = CS_COMMITTED;
    b.control_state_version_at_bind = st->control.control_state_version;
    b.observation_id = st->control.latest_observation_id;
}

__global__ void k_bind_external(AgentState* st, int x, int y, uint32_t binding) {
    BindResult& b = st->bind;
    b.resolved_x = x; b.resolved_y = y; b.binding = binding; b.plan = CS_COMMITTED; b.valid = 1; b.matched = 1;
    b.x = x; b.y = y; b.control_state_version_at_bind = st->control.control_state_version;
    b.observation_id = st->control.latest_observation_id; b.device_ns = (int64_t)gtimer();
    double dx = (double)(x - st->proposal.action.x), dy = (double)(y - st->proposal.action.y);
    b.moved_px = sqrt(dx * dx + dy * dy);
}

__global__ void k_observe_receipt(Memory* mem, uint64_t decision_id, int64_t o_ns, LedgerRow* ledger, const uint32_t* ledger_count, Config cfg) {
    if (mem->pending.active && mem->pending.decision_id == decision_id && mem->pending.receipt_observed_ns < 0) mem->pending.receipt_observed_ns = o_ns;
    for (uint32_t k = 0; k < cfg.ledger_rows && k < *ledger_count; ++k) {
        LedgerRow& row = ledger[(*ledger_count - 1 - k) % cfg.ledger_rows];
        if (row.decision_bound == decision_id) { if (row.receipt_observed_ns < 0) row.receipt_observed_ns = o_ns; break; }
    }
}

__global__ void k_discard(AgentState* st) {
    st->proposal.status = PS_DISCARDED; st->decision.open = 0;
}

__global__ void k_reset(AgentState* st, Memory* mem) {
    *st = AgentState{};
    st->next_action_id = 1;
    *mem = Memory{};
}

}  // namespace

void admit_update(AgentState* st, Memory* mem, const ObservationEnvelope* env, const uint64_t* hash, const unsigned long long* change,
                  const int* box4, const ArgmaxPair* am, const PeakStats* pk, const TemplateStats* ts, const int* patch_changed,
                  Config cfg, int have_prev, int n_pixels, int64_t host_now_ns, int Wv, DecisionTrace* ring, uint32_t* ring_count,
                  LedgerRow* ledger, uint32_t* ledger_count, cudaStream_t s) {
    k_admit_update<<<1, 1, 0, s>>>(st, mem, env, hash, change, box4, am, pk, ts, patch_changed, cfg, have_prev, n_pixels,
                                   host_now_ns, Wv, ring, ring_count, ledger, ledger_count);
}

void begin(AgentState* st, uint64_t decision_id, uint64_t obs_id, int64_t host_b_ns, LedgerRow* ledger, const uint32_t* ledger_count,
           Config cfg, cudaStream_t s) {
    k_begin<<<1, 1, 0, s>>>(st, decision_id, obs_id, host_b_ns, ledger, ledger_count, cfg);
}

void decide_detector(AgentState* st, Config cfg, cudaStream_t s) { k_decide_detector<<<1, 1, 0, s>>>(st, cfg); }

void set_proposal(AgentState* st, Action a, cudaStream_t s) { k_set_proposal<<<1, 1, 0, s>>>(st, a); }

void resolve(AgentState* st, const ArgmaxPair* am, const PeakStats* pk, const TemplateStats* ts, const int* box_newest, Config cfg,
             int use_ncc, int Wv, cudaStream_t s) {
    k_resolve<<<1, 1, 0, s>>>(st, am, pk, ts, box_newest, cfg, use_ncc, Wv);
}

void commit(AgentState* st, Memory* mem, CommittedAction* slots, Config cfg, DecisionTrace* ring, uint32_t* ring_count,
            int64_t host_now_ns, int recommit_test, LedgerRow* ledger, const uint32_t* ledger_count, cudaStream_t s) {
    k_commit<<<1, 1, 0, s>>>(st, mem, slots, cfg, ring, ring_count, host_now_ns, recommit_test, ledger, ledger_count);
}

void bind_external(AgentState* st, int x, int y, uint32_t binding, cudaStream_t s) { k_bind_external<<<1, 1, 0, s>>>(st, x, y, binding); }

void observe_receipt(Memory* mem, uint64_t decision_id, int64_t o_ns, LedgerRow* ledger, const uint32_t* ledger_count, Config cfg, cudaStream_t s) {
    k_observe_receipt<<<1, 1, 0, s>>>(mem, decision_id, o_ns, ledger, ledger_count, cfg);
}

void override_binding(AgentState* st, int x, int y, cudaStream_t s) { k_override<<<1, 1, 0, s>>>(st, x, y); }

void discard(AgentState* st, cudaStream_t s) { k_discard<<<1, 1, 0, s>>>(st); }

void reset(AgentState* st, Memory* mem, cudaStream_t s) { k_reset<<<1, 1, 0, s>>>(st, mem); }

}  // namespace chronotope::control
