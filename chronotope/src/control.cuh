// control.cuh — where the agent becomes intelligent (design §8): verify the pending action, choose,
// validate, commit. Every function here runs as a single-thread kernel on the control stream so that
// the state is mutated in one place, in stream order, with no host in the loop between O and D.
#pragma once
#include <cuda_runtime.h>
#include "state.cuh"

namespace chronotope::control {

// perceive -> update state -> verify previous action (one admitted observation); appends a ledger row when relevant
void admit_update(AgentState* st, Memory* mem, const ObservationEnvelope* env, const uint64_t* hash,
                  const unsigned long long* change, const int* box4, const ArgmaxPair* am, const PeakStats* pk,
                  const TemplateStats* ts, const int* patch_changed, Config cfg, int have_prev, int n_pixels,
                  int64_t host_now_ns, int Wv, DecisionTrace* ring, uint32_t* ring_count, LedgerRow* ledger,
                  uint32_t* ledger_count, cudaStream_t s);
// B: a decision begins on the newest admitted observation (the frame at B is pinned by the caller); marks the row
void begin(AgentState* st, uint64_t decision_id, uint64_t obs_id, int64_t host_b_ns, LedgerRow* ledger,
           const uint32_t* ledger_count, Config cfg, cudaStream_t s);
// the resident planner: click the centre of the detected target in the frame at B (the direct Gate-4 path)
void decide_detector(AgentState* st, Config cfg, cudaStream_t s);
// the external planner's answer enters the core as the typed pre-commit state
void set_proposal(AgentState* st, Action a, cudaStream_t s);
// resolve the open field on the newest frame under the policy; decide what the commit will do
void resolve(AgentState* st, const ArgmaxPair* am, const PeakStats* pk, const TemplateStats* ts, const int* box_newest,
             Config cfg, int use_ncc, int Wv, cudaStream_t s);
// §9.5 in one atomic logical operation, on the device (with the staleness gate and the ack-aware duplicate rule)
void commit(AgentState* st, Memory* mem, CommittedAction* slots, Config cfg, DecisionTrace* ring, uint32_t* ring_count,
            int64_t host_now_ns, int recommit_test, LedgerRow* ledger, const uint32_t* ledger_count, cudaStream_t s);
// an EXTERNAL binding (an in-process grounder model, or the environment's oracle in tests): the open field filled from outside
void bind_external(AgentState* st, int x, int y, uint32_t binding, cudaStream_t s);
// a HOST-determined binding (the negative control; never on the resident path)
void override_binding(AgentState* st, int x, int y, cudaStream_t s);
// the receipt of a committed action became VISIBLE in an observation (the echo panel on the physical route)
void observe_receipt(Memory* mem, uint64_t decision_id, int64_t o_ns, LedgerRow* ledger, const uint32_t* ledger_count, Config cfg, cudaStream_t s);
void discard(AgentState* st, cudaStream_t s);
void reset(AgentState* st, Memory* mem, cudaStream_t s);

}  // namespace chronotope::control
