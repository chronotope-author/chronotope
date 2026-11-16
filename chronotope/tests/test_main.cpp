// chronotope contract tests: kernels against CPU references, the §9.5 commit protocol, the §16 negative tests
// (items 8, 9, 10, 13, 14), the pending record, the audit, and a closed loop on the synthetic target.
// No framework: CHECK() counts failures; the process exit code is the number of failed tests.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include "chronotope_api.h"
#include "synthetic.h"

using namespace chronotope;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_MSG(cond, ...) do { ++g_checks; if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static chronotope_config base_cfg(uint32_t policy = CHRONOTOPE_POL_LATE_BIND) {
    chronotope_config c{};
    c.width = 320; c.height = 240; c.pixfmt = CHRONOTOPE_PIX_RGB24;
    c.template_radius_px = 12; c.min_match_score = 0.8; c.policy = policy; c.restart_threshold_px = (policy == CHRONOTOPE_POL_RESTART_IF_MOVED) ? 6 : -1;
    c.target_rgb[0] = 220; c.target_rgb[1] = 38; c.target_rgb[2] = 38; c.has_target_color = 1;
    c.movement_threshold_px = 6; c.track_every_frame = 1;
    return c;
}

struct Ctx {
    chronotope_core* core = nullptr;
    char err[256] = {0};
    explicit Ctx(const chronotope_config& c) { core = chronotope_create(&c, err, sizeof err); if (!core) std::printf("  create failed: %s\n", err); }
    ~Ctx() { if (core) chronotope_destroy(core); }
};

static uint64_t g_obs = 0;
static int64_t host_now() { timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec; }
static int64_t g_last_pub = 0;
static chronotope_admission admit(Ctx& c, const std::vector<uint8_t>& f, int64_t task_ns, uint64_t* id = nullptr) {
    chronotope_admission a{};
    uint64_t oid = ++g_obs;
    g_last_pub = host_now();   // publication on the host clock, as the runner stamps O (the staleness gate compares with it)
    int rc = chronotope_admit(c.core, f.data(), f.size(), oid, task_ns >= 500000000 ? 1 : 0, task_ns, g_last_pub, &a);
    if (rc != CHRONOTOPE_OK) std::printf("  admit rc=%d err=%s\n", rc, chronotope_last_error(c.core));
    if (id) *id = oid;
    return a;
}

// ---- kernels ---------------------------------------------------------------------------------------
static void t_grey_hash_detect() {
    std::printf("[t_grey_hash_detect]\n");
    Synthetic s; s.vx = 0;
    std::vector<uint8_t> f0, f1;
    s.render(600000000, f0);
    Ctx c(base_cfg()); if (!c.core) { g_fail++; return; }
    uint64_t id;
    chronotope_admission a = admit(c, f0, 600000000, &id);
    CHECK(a.status == CHRONOTOPE_OK);
    SynthBox b = s.box_at(600000000);
    CHECK_MSG(a.box_present && a.box[0] == b.x0 && a.box[1] == b.y0 && a.box[2] == b.x1 && a.box[3] == b.y1,
              "box %d %d %d %d vs %d %d %d %d", a.box[0], a.box[1], a.box[2], a.box[3], b.x0, b.y0, b.x1, b.y1);
    CHECK(a.relevance == CHRONOTOPE_REL_ACTION && a.control_state_version == 1);
    uint64_t h0 = a.payload_hash;
    chronotope_admission a2 = admit(c, f0, 616666667);   // identical frame: same hash, no relevance, version unchanged
    CHECK(a2.payload_hash == h0 && a2.relevance == CHRONOTOPE_REL_NONE && a2.control_state_version == 1 && a2.change_mean == 0.0);
    s.render(200000000, f1);                         // no target yet: box absent -> relevant change
    chronotope_admission a3 = admit(c, f1, 633333334);
    CHECK(a3.payload_hash != h0 && !a3.box_present && a3.relevance == CHRONOTOPE_REL_ACTION && a3.control_state_version == 2 && a3.change_mean > 0.0);
    // stale observation (§16 item 14): an id not newer than the newest admitted is refused
    chronotope_admission st{};
    int rc = chronotope_admit(c.core, f0.data(), f0.size(), 1, 0, 0, 0, &st);
    CHECK(rc == CHRONOTOPE_STALE && st.status == CHRONOTOPE_STALE);
}

static void t_track_shift() {
    std::printf("[t_track_shift]\n");
    Synthetic s; s.vx = 16; s.distractor = true;
    std::vector<uint8_t> f0, ft;
    s.render(200000000 + 500000000, f0);   // event at 0.5 s; t = 0.7 s
    Ctx c(base_cfg()); if (!c.core) { g_fail++; return; }
    SynthBox b0 = s.box_at(700000000);
    for (int64_t t : {700000000LL, 1100000000LL, 1500000000LL}) {
        s.render(t, ft);
        SynthBox bt = s.box_at(t);
        chronotope_match m{};
        CHECK(chronotope_track(c.core, f0.data(), ft.data(), f0.size(), b0.cx() + 4, b0.cy() - 3, 12, 0, &m) == CHRONOTOPE_OK);
        CHECK_MSG(m.x == bt.cx() + 4 && m.y == bt.cy() - 3 && m.score > 0.99 && m.matched && !m.ambiguous,
                  "t=%lld got (%d,%d) score %.4f second %.4f plateau %d, want (%d,%d)", (long long)t, m.x, m.y, m.score, m.second, m.plateau, bt.cx() + 4, bt.cy() - 3);
    }
    // an RGB template on the same frames
    chronotope_match m{};
    s.render(1100000000, ft);
    CHECK(chronotope_track(c.core, f0.data(), ft.data(), f0.size(), b0.cx(), b0.cy(), 12, 1, &m) == CHRONOTOPE_OK);
    CHECK_MSG(m.x == s.box_at(1100000000).cx() && m.y == s.box_at(1100000000).cy() && m.matched, "rgb (%d,%d) score %.4f", m.x, m.y, m.score);
}

static void t_track_flat_and_ambiguous() {
    std::printf("[t_track_flat_and_ambiguous]\n");
    Ctx c(base_cfg()); if (!c.core) { g_fail++; return; }
    std::vector<uint8_t> flat((size_t)320 * 240 * 3, 200);
    chronotope_match m{};
    CHECK(chronotope_track(c.core, flat.data(), flat.data(), flat.size(), 100, 100, 12, 0, &m) == CHRONOTOPE_OK);
    CHECK_MSG(m.flat && m.score == 0.0 && m.second == 1.0 && m.plateau == 1000000 && !m.matched && m.ambiguous && m.x == m.tpl_dx && m.y == m.tpl_dy,
              "flat: score %.3f second %.3f plateau %d", m.score, m.second, m.plateau);
    // two identical squares: the template re-occurs elsewhere -> second ~ best -> ambiguous, not matched
    std::vector<uint8_t> two((size_t)320 * 240 * 3, 30);
    auto sq = [&](int x0, int y0) { for (int y = y0; y < y0 + 10; ++y) for (int x = x0; x < x0 + 10; ++x) { size_t o = ((size_t)y * 320 + x) * 3; two[o] = 250; two[o + 1] = 250; two[o + 2] = 250; } };
    sq(60, 60); sq(200, 150);
    CHECK(chronotope_track(c.core, two.data(), two.data(), two.size(), 65, 65, 12, 0, &m) == CHRONOTOPE_OK);
    CHECK_MSG(m.score > 0.99 && m.second > 0.99 && m.ambiguous && !m.matched, "ambiguous: score %.4f second %.4f", m.score, m.second);
    // a wide uniform region: the template slides -> plateau > PLATEAU_MAX
    std::vector<uint8_t> wide((size_t)320 * 240 * 3, 30);
    for (int y = 100; y < 140; ++y) for (int x = 0; x < 320; ++x) { size_t o = ((size_t)y * 320 + x) * 3; wide[o] = 250; wide[o + 1] = 250; wide[o + 2] = 250; }
    CHECK(chronotope_track(c.core, wide.data(), wide.data(), wide.size(), 160, 120, 12, 0, &m) == CHRONOTOPE_OK);
    CHECK_MSG(m.ambiguous && !m.matched, "plateau: score %.4f second %.4f plateau %d", m.score, m.second, m.plateau);
}

// ---- the commit protocol ----------------------------------------------------------------------------
static void t_commit_frozen_and_immutable() {
    std::printf("[t_commit_frozen_and_immutable]\n");
    Synthetic s; s.vx = 40;
    std::vector<uint8_t> f;
    Ctx c(base_cfg(CHRONOTOPE_POL_FROZEN)); if (!c.core) { g_fail++; return; }
    s.render(500000000, f);
    uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{};
    CHECK(chronotope_begin_decision(c.core, id, &p) == CHRONOTOPE_OK && chronotope_decision_open(c.core) == 1);
    CHECK(chronotope_decide_detector(c.core, &p) == CHRONOTOPE_OK && p.status == 1 && p.kind == CHRONOTOPE_ACT_CLICK);
    SynthBox b = s.box_at(500000000);
    CHECK_MSG(p.x == b.cx() && p.y == b.cy(), "proposal (%d,%d) vs centre (%d,%d)", p.x, p.y, b.cx(), b.cy());
    s.render(1000000000, f); admit(c, f, 1000000000);   // the target moved 20 px during "deliberation"
    chronotope_commit_result r{};
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK);
    CHECK_MSG(r.outcome == CHRONOTOPE_CS_COMMITTED && r.x == p.x && r.y == p.y && r.binding == CHRONOTOPE_BIND_STATE_AT_B && r.action_id == 1,
              "outcome %u (%d,%d) binding %u", r.outcome, r.x, r.y, r.binding);
    CHECK(r.decision_id == 1 && r.proposal_observation_id == id && r.d_device_ns > 0 && r.d_host_ns > 0 && r.serialized_action_hash != 0);
    CHECK(chronotope_decision_open(c.core) == 0);
    // §16 item 10: the committed record cannot be written again
    chronotope_commit_result r2{};
    CHECK(chronotope_test_recommit(c.core, &r2) == CHRONOTOPE_OK && r2.outcome == CHRONOTOPE_CS_FAILED_IMMUTABLE && r2.action_id == 1);
    // and a commit without an open decision is NO_PROPOSAL, never a second record
    chronotope_commit_result r3{};
    CHECK(chronotope_resolve_and_commit(c.core, &r3) == CHRONOTOPE_NO_DECISION && r3.outcome == CHRONOTOPE_CS_NO_PROPOSAL);
}

static void t_late_bind_same_decision() {
    std::printf("[t_late_bind_same_decision]\n");
    Synthetic s; s.vx = 40;
    std::vector<uint8_t> f;
    Ctx c(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c.core) { g_fail++; return; }
    s.render(500000000, f);
    uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{};
    chronotope_begin_decision(c.core, id, &p);
    chronotope_decide_detector(c.core, &p);
    // frames during B->D are admitted; the streaming binder tracks the template and the version advances on motion
    chronotope_admission last{};
    int64_t t_last = 0;
    for (int k = 0; k < 30; ++k) { t_last = 516666667 + (int64_t)k * 16666667; s.render(t_last, f); last = admit(c, f, t_last); }
    CHECK_MSG(last.tracked_valid && last.tracked_x == s.box_at(t_last).cx(), "tracked (%d,%d) valid %d", last.tracked_x, last.tracked_y, last.tracked_valid);
    chronotope_commit_result r{};
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK);
    SynthBox bd = s.box_at(t_last);
    CHECK_MSG(r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == CHRONOTOPE_BIND_TRACKED_AT_D && r.matched && !r.ambiguous && r.x == bd.cx() && r.y == bd.cy(),
              "outcome %u binding %u matched %d (%d,%d) vs (%d,%d) score %.4f", r.outcome, r.binding, r.matched, r.x, r.y, bd.cx(), bd.cy(), r.match_score);
    CHECK(r.decision_id == 1 && r.proposed_x == p.x && r.moved_px > 12.0 && r.track_device_ns > 0 && r.track_ns > 0);
    CHECK(r.control_state_version_at_commit == last.control_state_version);   // the guard read the same version the bind did
    std::printf("  bind latency: device %.3f ms, host %.3f ms (GPU shared with a running study)\n", r.track_device_ns / 1e6, r.track_ns / 1e6);
    chronotope_audit_report au{};
    chronotope_audit(c.core, &au);
    CHECK_MSG(au.residency_ok && au.d2h_state_in_window == 0 && au.host_action_computations == 0 && au.mallocs_after_warmup == 0 && au.host_syncs_action_in_window == 0,
              "audit: d2h_state %llu host_actions %llu mallocs %llu", (unsigned long long)au.d2h_state_in_window, (unsigned long long)au.host_action_computations, (unsigned long long)au.mallocs_after_warmup);
    CHECK(au.windows_opened == 1 && au.declared_commit_waits == 1 && au.h2d_publications == 31 && !au.window_open);   // 1 + 30 frames
    // the trace ring saw the observations while the proposal was open and the accepted commit
    std::vector<chronotope_trace_entry> tr(64); uint32_t n = 0;
    CHECK(chronotope_trace(c.core, tr.data(), 64, &n) == CHRONOTOPE_OK && n >= 2 && tr[n - 1].accepted == 1 && tr[n - 1].reason == CHRONOTOPE_CS_COMMITTED && tr[n - 2].accepted == 0 && tr[n - 2].reason == 0);
}

static void t_detector_at_d() {
    std::printf("[t_detector_at_d]\n");
    for (uint32_t rule = 1; rule <= 2; ++rule) {
        Synthetic s; s.vx = 40;
        std::vector<uint8_t> f;
        chronotope_config cfg = base_cfg(CHRONOTOPE_POL_LATE_BIND); cfg.detector_at_d = rule;
        Ctx c(cfg); if (!c.core) { g_fail++; return; }
        s.render(500000000, f);
        uint64_t id; admit(c, f, 500000000, &id);
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        int64_t t_last = 0;
        for (int k = 0; k < 30; ++k) { t_last = 516666667 + (int64_t)k * 16666667; s.render(t_last, f); admit(c, f, t_last); }
        chronotope_commit_result r{};
        CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK);
        SynthBox bd = s.box_at(t_last);
        uint32_t want = rule == 1 ? CHRONOTOPE_BIND_DETECTED_AT_D : CHRONOTOPE_BIND_TRACKED_AT_D;   // rule 2: the template matched, so the tracker wins
        CHECK_MSG(r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == want && r.matched && r.x == bd.cx() && r.y == bd.cy(),
                  "rule %u: outcome %u binding %u matched %d (%d,%d) vs (%d,%d)", rule, r.outcome, r.binding, r.matched, r.x, r.y, bd.cx(), bd.cy());
        CHECK(r.moved_px > 12.0 && r.proposed_x == p.x);
        chronotope_audit_report au{}; chronotope_audit(c.core, &au);
        CHECK_MSG(au.residency_ok && au.host_action_computations == 0, "rule %u audit: residency %d host_actions %llu", rule, (int)au.residency_ok, (unsigned long long)au.host_action_computations);
    }
    {   // rule 1 with the target GONE at D: no detection, the tracker does not match -> the original proposal, state_at_B
        Synthetic s; s.vx = 0;
        std::vector<uint8_t> f;
        chronotope_config cfg = base_cfg(CHRONOTOPE_POL_LATE_BIND); cfg.detector_at_d = 1;
        Ctx c(cfg); if (!c.core) { g_fail++; return; }
        s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        s.window_ns = 50000000; s.render(600000000, f); admit(c, f, 600000000);   // the window closed: the target is gone at D
        chronotope_commit_result r{}; CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK);
        CHECK_MSG(r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == CHRONOTOPE_BIND_STATE_AT_B && !r.matched && r.x == p.x, "gone: binding %u matched %d", r.binding, r.matched);
    }
}

static void t_version_guard() {
    std::printf("[t_version_guard]\n");
    Synthetic s; s.vx = 0; s.distractor = true;
    std::vector<uint8_t> f;
    Ctx c(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c.core) { g_fail++; return; }
    s.render(500000000, f);
    uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
    chronotope_commit_result r{};
    CHECK(chronotope_resolve(c.core, &r) == CHRONOTOPE_OK);
    // §16 item 13: an IRRELEVANT admission (the blinking distractor: observation version advances, control state does not)
    s.render(516666667, f); chronotope_admission a = admit(c, f, 516666667);
    CHECK(a.relevance == CHRONOTOPE_REL_NONE && a.observation_version == 2 && a.control_state_version == 1);
    CHECK(chronotope_commit_only(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED);
    // a RELEVANT change between the bind and the commit fails the commit explicitly (§9.5)
    s.render(1000000000, f); admit(c, f, 1000000000);
    chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
    CHECK(chronotope_resolve(c.core, &r) == CHRONOTOPE_OK);
    s.x0 += 30; s.render(1016666667, f); a = admit(c, f, 1016666667);
    CHECK(a.relevance == CHRONOTOPE_REL_ACTION);
    CHECK(chronotope_commit_only(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_FAILED_STATE_CHANGED && chronotope_decision_open(c.core) == 0);
}

static void t_restart_policies() {
    std::printf("[t_restart_policies]\n");
    Synthetic s; s.vx = 40;
    std::vector<uint8_t> f;
    {
        Ctx c(base_cfg(CHRONOTOPE_POL_RESTART_IF_MOVED)); if (!c.core) { g_fail++; return; }
        s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        s.render(1000000000, f); admit(c, f, 1000000000);
        chronotope_commit_result r{};
        CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_RESTART_MOVED && r.moved_px > 6.0 && chronotope_decision_open(c.core) == 0);
        // the next decision begins on the newest frame and commits when nothing moved since
        chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
        s.render(1016666667, f); admit(c, f, 1016666667);
        CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == CHRONOTOPE_BIND_STATE_AT_B && r.decision_id == 2);
    }
    {
        Ctx c(base_cfg(CHRONOTOPE_POL_LATE_BIND_OR_RESTART)); if (!c.core) { g_fail++; return; }
        s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        s.render(400000000, f); admit(c, f, 17000000000);   // the target is gone at D
        chronotope_commit_result r{};
        CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_RESTART_NOT_FOUND && !r.matched);
    }
    {   // late_bind with the target gone: the proposal is committed unchanged, recorded matched = 0 (the fall-back)
        Ctx c(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c.core) { g_fail++; return; }
        s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        s.render(400000000, f); admit(c, f, 17000000000);
        chronotope_commit_result r{};
        CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && !r.matched && r.x == p.x && r.y == p.y && r.binding == CHRONOTOPE_BIND_TRACKED_AT_D);
    }
    {   // V1 kind validation: a click proposed while the newest frame shows no target is discarded (a new decision begins)
        chronotope_config cfg = base_cfg(CHRONOTOPE_POL_LATE_BIND); cfg.validate_kind = 1;
        Ctx c(cfg); if (!c.core) { g_fail++; return; }
        s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        s.render(400000000, f); admit(c, f, 17000000000);
        chronotope_commit_result r{};
        CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_RESTART_KIND);
    }
}

static void t_pending_record() {
    std::printf("[t_pending_record]\n");
    chronotope_config cfg = base_cfg(CHRONOTOPE_POL_FROZEN); cfg.pending_radius_px = 8; cfg.pending_timeout_ns = 2000000000;
    Synthetic s; s.vx = 0;
    std::vector<uint8_t> f;
    Ctx c(cfg); if (!c.core) { g_fail++; return; }
    s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
    chronotope_commit_result r{};
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.pending_created);
    // the same click again while the effect has not appeared: rejected as a duplicate
    s.render(516666667, f); chronotope_admission a = admit(c, f, 516666667);
    CHECK(a.pending_active && a.pending_event == CHRONOTOPE_PE_NONE);
    chronotope_begin_decision(c.core, g_obs, &p);
    CHECK(chronotope_propose(c.core, CHRONOTOPE_ACT_CLICK, r.x + 2, r.y - 1, 0, &p) == CHRONOTOPE_OK && p.status == 1);
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_REJECTED_DUPLICATE);
    // the effect appears (the patch around the click changes): the record resolves, the next click passes
    s.x0 += 60; s.render(533333334, f); a = admit(c, f, 533333334);
    CHECK_MSG(!a.pending_active && a.pending_event == CHRONOTOPE_PE_RESOLVED, "pending active %d event %u", a.pending_active, a.pending_event);
    chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.action_id == 2);
    // expiry: a short timeout and a frame that does not change the patch
    chronotope_config cfg2 = cfg; cfg2.pending_timeout_ns = 20000000;
    Ctx c2(cfg2); if (!c2.core) { g_fail++; return; }
    s.render(600000000, f); admit(c2, f, 600000000);
    chronotope_begin_decision(c2.core, g_obs, &p); chronotope_decide_detector(c2.core, &p);
    CHECK(chronotope_resolve_and_commit(c2.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    a = admit(c2, f, 616666667);
    CHECK_MSG(!a.pending_active && a.pending_event == CHRONOTOPE_PE_EXPIRED, "expiry: active %d event %u", a.pending_active, a.pending_event);
}

static void t_repeat_without_effect() {
    std::printf("[t_repeat_without_effect]\n");
    chronotope_config cfg = base_cfg(CHRONOTOPE_POL_FROZEN); cfg.max_repeat_without_effect = 1; cfg.pending_radius_px = 4;
    Synthetic s; s.vx = 0;
    std::vector<uint8_t> f;
    Ctx c(cfg); if (!c.core) { g_fail++; return; }
    s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
    chronotope_commit_result r{};
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED);
    admit(c, f, 516666667);   // nothing happened
    chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_REJECTED_REPEATED && r.repetition_count == 1);
}

// ---- the negative controls: host participation is visible in the audit (§16 items 8, 9) ---------------
static void t_audit_host_binder() {
    std::printf("[t_audit_host_binder]\n");
    Synthetic s; s.vx = 40;
    std::vector<uint8_t> f;
    Ctx c(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c.core) { g_fail++; return; }
    s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
    s.render(1000000000, f); admit(c, f, 1000000000);
    std::vector<uint8_t> buf(f.size());
    CHECK(chronotope_read_frame(c.core, buf.data(), buf.size()) == CHRONOTOPE_OK && buf == f);   // the host reads decision state inside O->D
    SynthBox bd = s.box_at(1000000000);
    CHECK(chronotope_override_binding(c.core, bd.cx(), bd.cy()) == CHRONOTOPE_OK);              // and returns a revised proposal
    chronotope_commit_result r{};
    CHECK(chronotope_commit_only(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == CHRONOTOPE_BIND_HOST_OVERRIDE && r.x == bd.cx());
    chronotope_audit_report au{};
    chronotope_audit(c.core, &au);
    CHECK_MSG(!au.residency_ok && au.d2h_state_in_window == 1 && au.host_syncs_action_in_window == 1 && au.host_action_computations == 1,
              "audit: d2h_state %llu syncs %llu actions %llu", (unsigned long long)au.d2h_state_in_window, (unsigned long long)au.host_syncs_action_in_window, (unsigned long long)au.host_action_computations);
    // buffers are preallocated: an allocation after warm-up is a violation the audit reports
    Ctx c2(base_cfg()); if (!c2.core) { g_fail++; return; }
    chronotope_audit(c2.core, &au);
    CHECK(au.residency_ok && au.mallocs_after_warmup == 0 && au.device_bytes_allocated > 0);
    CHECK(chronotope_test_alloc(c2.core) == CHRONOTOPE_OK);
    chronotope_audit(c2.core, &au);
    CHECK(!au.residency_ok && au.mallocs_after_warmup == 1);
}

// ---- the closed loop: the equal-timing relation on the device -------------------------------------------
static void t_closed_loop() {
    std::printf("[t_closed_loop]\n");
    const int64_t period = 16666667, delib = 500000000, delivery = 20000000;
    for (uint32_t pol : {CHRONOTOPE_POL_FROZEN, CHRONOTOPE_POL_LATE_BIND}) {
        int hits = 0, n = 0; double err_sum = 0.0;
        for (uint64_t seed = 1; seed <= 5; ++seed) {
            Synthetic s; s.vx = 40; s.place_for_seed(seed); s.x0 = std::min(s.x0, s.w - s.size - 100);   // stays inside the canvas for 2 s
            std::vector<uint8_t> f;
            Ctx c(base_cfg(pol)); if (!c.core) { g_fail++; return; }
            int64_t t = 0; uint64_t id = 0; chronotope_proposal p{}; bool began = false; int64_t t_b = 0;
            chronotope_commit_result r{}; bool done = false;
            while (t < 2000000000 && !done) {
                s.render(t, f);
                chronotope_admission a = admit(c, f, t, &id);
                if (!began && a.box_present) {
                    chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p); began = true; t_b = t;
                } else if (began && t - t_b >= delib) {
                    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED);
                    done = true;
                }
                t += period;
            }
            int64_t t_d = t - period, t_a = t_d + delivery;
            SynthBox ba = s.box_at(t_a);
            bool hit = ba.contains(r.x, r.y);
            double err = std::hypot((double)(r.x - ba.cx()), (double)(r.y - ba.cy()));
            hits += hit; ++n; err_sum += err;
            // the error law: the unchanged coordinate misses by v (t_A - tau); the bound coordinate by v (t_A - tau_D)
            double travel_frozen = 40.0 * (double)(t_a - t_b) / 1e9, travel_bound = 40.0 * (double)(t_a - t_d) / 1e9;
            if (pol == CHRONOTOPE_POL_FROZEN) CHECK_MSG(std::fabs(err - travel_frozen) <= 2.0, "frozen err %.1f vs travel %.1f", err, travel_frozen);
            else CHECK_MSG(err <= travel_bound + 2.0, "late_bind err %.1f vs travel %.1f", err, travel_bound);
        }
        std::printf("  policy %s: %d/%d hits, mean error %.1f px (v = 40 px/s, deliberation 0.5 s, delivery 20 ms)\n",
                    pol == CHRONOTOPE_POL_FROZEN ? "frozen   " : "late_bind", hits, n, err_sum / n);
        if (pol == CHRONOTOPE_POL_FROZEN) CHECK(hits == 0); else CHECK(hits == n);
    }
}

static void t_latency() {
    std::printf("[t_latency]\n");
    Synthetic s; s.vx = 8;
    std::vector<uint8_t> f;
    Ctx c(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c.core) { g_fail++; return; }
    std::vector<double> dev, host, adm;
    for (int i = 0; i < 100; ++i) {
        int64_t t = 500000000 + (int64_t)i * 33333334;
        s.render(t, f); uint64_t id;
        auto t0 = std::chrono::steady_clock::now();
        admit(c, f, t, &id);
        adm.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
        s.render(t + 16666667, f); admit(c, f, t + 16666667);
        chronotope_commit_result r{};
        chronotope_resolve_and_commit(c.core, &r);
        dev.push_back(r.track_device_ns / 1e6); host.push_back(r.track_ns / 1e6);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto p90 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[(size_t)(v.size() * 0.9)]; };
    std::printf("  admission (H2D+perceive+update, host wall): median %.3f ms, p90 %.3f ms\n", med(adm), p90(adm));
    std::printf("  bind+commit: device median %.3f ms p90 %.3f ms; host wall median %.3f ms p90 %.3f ms (n = 100, 320x240, r = 12)\n",
                med(dev), p90(dev), med(host), p90(host));
    std::printf("AUDIT_LATENCY_MS %.4f %.4f %.4f %.4f %.4f %.4f\n", med(adm), p90(adm), med(dev), p90(dev), med(host), p90(host));
    CHECK(med(dev) < 50.0);
}


// ---- P2: the event ledger, the staleness gate, the ack-aware duplicate rule, the external binder ------------------
static void t_ledger_and_gate() {
    std::printf("[t_ledger_and_gate]\n");
    chronotope_config cfg = base_cfg(CHRONOTOPE_POL_LATE_BIND); cfg.pending_radius_px = 8; cfg.pending_timeout_ns = 2000000000;
    Synthetic s; s.vx = 40;
    std::vector<uint8_t> f;
    Ctx c(cfg); if (!c.core) { g_fail++; return; }
    s.render(200000000, f); admit(c, f, 200000000);           // no target: row 1 (first admission is relevant)
    admit(c, f, 216666667); admit(c, f, 233333334);            // identical: no rows
    int64_t pub_prev = g_last_pub;
    s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);   // the target appears: row 2, bracket = previous O
    std::vector<chronotope_ledger_row> rows(64); uint32_t n = 0;
    CHECK(chronotope_ledger(c.core, rows.data(), 64, &n) == CHRONOTOPE_OK && n == 2);
    CHECK_MSG(rows[1].observation_id == id && rows[1].e_bracket_lo_ns == pub_prev && rows[1].o_ns == g_last_pub && rows[1].frames_since_prev == 3 && rows[1].relevance == CHRONOTOPE_REL_ACTION,
              "row: obs %llu bracket_lo %lld frames %u", (unsigned long long)rows[1].observation_id, (long long)rows[1].e_bracket_lo_ns, rows[1].frames_since_prev);
    CHECK(rows[1].decision_proposed == 0 && rows[1].decision_bound == 0 && rows[1].effect_observed_ns == -1 && rows[1].receipt_observed_ns == -1);
    chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
    chronotope_ledger(c.core, rows.data(), 64, &n);
    CHECK(rows[1].decision_proposed == 1);
    s.render(1000000000, f); admit(c, f, 1000000000);          // the target moved: row 3 (the row the bind will land on)
    chronotope_commit_result r{};
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == CHRONOTOPE_BIND_TRACKED_AT_D);
    CHECK_MSG(r.evidence_age_ns >= 0 && r.evidence_age_ns < 2000000000LL && r.evidence_bracket_ns > 0, "age %lld bracket %lld", (long long)r.evidence_age_ns, (long long)r.evidence_bracket_ns);
    chronotope_ledger(c.core, rows.data(), 64, &n);
    CHECK_MSG(n >= 3 && rows[n - 1].decision_bound == 1 && rows[n - 1].d_device_ns > 0 && r.ledger_row == (n - 1) % 256, "bound row: n %u bound %llu", n, (unsigned long long)rows[n - 1].decision_bound);
    // the effect column: the pending patch changes -> the bound row records the observation
    s.x0 += 60; s.render(1016666667, f); chronotope_admission a = admit(c, f, 1016666667);
    CHECK(a.pending_event == CHRONOTOPE_PE_RESOLVED);
    chronotope_ledger(c.core, rows.data(), 64, &n);
    bool seen = false; for (uint32_t i = 0; i < n; ++i) if (rows[i].decision_bound == 1 && rows[i].effect_observed_ns == g_last_pub) seen = true;
    CHECK(seen);
    // the staleness gate: evidence older than the bound -> a new decision, no click
    chronotope_config cfg2 = base_cfg(CHRONOTOPE_POL_LATE_BIND); cfg2.max_evidence_age_ns = 20000000;
    Ctx c2(cfg2); if (!c2.core) { g_fail++; return; }
    s.render(500000000, f); admit(c2, f, 500000000, &id);
    chronotope_begin_decision(c2.core, id, &p); chronotope_decide_detector(c2.core, &p);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    CHECK(chronotope_resolve_and_commit(c2.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_RESTART_STALE_EVIDENCE && chronotope_decision_open(c2.core) == 0);
    s.render(516666667, f); admit(c2, f, 516666667);            // a fresh frame: the next decision commits
    chronotope_begin_decision(c2.core, g_obs, &p); chronotope_decide_detector(c2.core, &p);
    CHECK(chronotope_resolve_and_commit(c2.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED);
}

static void t_ack_aware_pending_and_external_binder() {
    std::printf("[t_ack_aware_pending_and_external_binder]\n");
    chronotope_config cfg = base_cfg(CHRONOTOPE_POL_FROZEN); cfg.pending_radius_px = 8; cfg.pending_timeout_ns = 2000000000;
    cfg.ack_aware_pending = 1; cfg.receipt_timeout_ns = 30000000;
    Synthetic s; s.vx = 0;
    std::vector<uint8_t> f;
    Ctx c(cfg); if (!c.core) { g_fail++; return; }
    s.render(500000000, f); uint64_t id; admit(c, f, 500000000, &id);
    chronotope_proposal p{}; chronotope_begin_decision(c.core, id, &p); chronotope_decide_detector(c.core, &p);
    chronotope_commit_result r{};
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED);
    // no receipt seen, before the receipt timeout: a duplicate is refused
    admit(c, f, 516666667);
    chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_REJECTED_DUPLICATE);
    // no receipt seen, after the receipt timeout: the click counts as dropped and the retry is allowed
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    admit(c, f, 533333334);
    chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.action_id == 2);
    // a VISIBLE receipt protects the record past the receipt timeout (until the effect or the pending expiry)
    CHECK(chronotope_observe_receipt(c.core, r.decision_id, 533333334 + 30000000) == CHRONOTOPE_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    admit(c, f, 550000001);
    chronotope_begin_decision(c.core, g_obs, &p); chronotope_decide_detector(c.core, &p);
    CHECK(chronotope_resolve_and_commit(c.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_REJECTED_DUPLICATE);
    std::vector<chronotope_ledger_row> rows(64); uint32_t n = 0;
    chronotope_ledger(c.core, rows.data(), 64, &n);
    bool rec = false; for (uint32_t i = 0; i < n; ++i) if (rows[i].decision_bound == 3 && rows[i].receipt_observed_ns > 0) rec = true;
    CHECK(rec);
    // the external binder: the oracle fills the open field with the environment's truth (test only; counted as host action)
    Synthetic m; m.vx = 40;
    Ctx c2(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c2.core) { g_fail++; return; }
    m.render(500000000, f); admit(c2, f, 500000000, &id);
    chronotope_begin_decision(c2.core, id, &p); chronotope_decide_detector(c2.core, &p);
    m.render(1000000000, f); admit(c2, f, 1000000000);
    SynthBox bd = m.box_at(1000000000);
    CHECK(chronotope_bind_external(c2.core, bd.cx(), bd.cy(), CHRONOTOPE_BIND_ORACLE) == CHRONOTOPE_OK);
    CHECK(chronotope_commit_only(c2.core, &r) == CHRONOTOPE_OK && r.outcome == CHRONOTOPE_CS_COMMITTED && r.binding == CHRONOTOPE_BIND_ORACLE && r.x == bd.cx() && r.moved_px > 12.0);
    chronotope_audit_report au{}; chronotope_audit(c2.core, &au);
    CHECK(au.host_action_computations == 1 && au.binder_host_steps == 0 && !au.residency_ok);
    // the grounder classification: counted as a binder host step, not as a host algorithm
    Ctx c3(base_cfg(CHRONOTOPE_POL_LATE_BIND)); if (!c3.core) { g_fail++; return; }
    m.render(500000000, f); admit(c3, f, 500000000, &id);
    chronotope_begin_decision(c3.core, id, &p); chronotope_decide_detector(c3.core, &p);
    CHECK(chronotope_bind_external(c3.core, bd.cx(), bd.cy(), CHRONOTOPE_BIND_GROUNDER) == CHRONOTOPE_OK && chronotope_commit_only(c3.core, &r) == CHRONOTOPE_OK && r.binding == CHRONOTOPE_BIND_GROUNDER);
    chronotope_audit(c3.core, &au);
    CHECK(au.binder_host_steps == 1 && au.host_action_computations == 0);
    CHECK(chronotope_bind_external(c3.core, 1, 1, CHRONOTOPE_BIND_GROUNDER) == CHRONOTOPE_ERR);   // no open decision
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("chronotope %s\n", chronotope_version());
    t_grey_hash_detect();
    t_track_shift();
    t_track_flat_and_ambiguous();
    t_commit_frozen_and_immutable();
    t_late_bind_same_decision();
    t_version_guard();
    t_detector_at_d();
    t_restart_policies();
    t_pending_record();
    t_repeat_without_effect();
    t_audit_host_binder();
    t_ledger_and_gate();
    t_ack_aware_pending_and_external_binder();
    t_closed_loop();
    t_latency();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail;
}
