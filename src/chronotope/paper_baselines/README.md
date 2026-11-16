# Paper baselines for the core-five harness

These are independent Python host baselines. They do not change the accepted
legacy C++ detector, Agent S3 Worker, or native model backend. They are not the
GPU-history Chronotope system. No study is launched by importing these modules,
fetching their source, preparing a plan, or checking its contracts.

| Adapter | Reused mechanism | Explicit adaptation |
| --- | --- | --- |
| AOI visual | Authors' CLIP/pixel keyframe extractor, observation record, persistent narration, CU prompt and parser | 3 Hz published frames buffered during blocking inference; CLIP processes that buffer at the next decision. No audio/DOM; no concurrent perception claim. |
| MobileDreamer released selection | Authors' candidate generation/deduplication, outcome prediction prompt, selection/parsing methods, and OCR sketch formatter | Three candidates, one-step predictions, direct public task instruction and harness click/type/wait tools. Android reasoner/history plumbing is replaced. The OCR bridge maps current PaddleOCR result blocks to the original sketch format. |
| AAPT paper-derived | Flat bounded policy trees, grammar-constrained planner and observer, change gate, eligibility and deadline enforcement | Harness click/type/wait grammar, existing frame cadence, planning consumes the normal trial clock. No free preparation period. |

The sources are [AOI](https://github.com/19PINE-AI/aoi),
[MobileDreamer](https://github.com/UITron-hub/MobileDreamer), and
[AAPT, arXiv:2607.28399v1](https://arxiv.org/html/2607.28399v1).
Exact commits and file SHA256s are in `sources.lock.json`. AOI is pinned at
`5ee56ceee768fe6a3f3cfa9aaab6c81dfa31407d`; MobileDreamer at
`473c0bd83a47eeb0a39762942b22ebc9b23d7b73`. Upstream code is fetched into the
ignored `runs/paper_baseline_sources` directory, never copied into a new license
grant. AOI has an MIT license; the pinned MobileDreamer release has no license
file. Its Android `trajectory_rollout` base package is also absent. The adapter
therefore compiles selected, unchanged method ASTs from the verified source
instead of importing that missing base class or pretending to reproduce it.

MobileDreamer's README reports a depth-two paper setting, but its released
selection helpers implement one-step candidate outcomes. Report this adapter as
**MobileDreamer released one-step selection (adapted)**, not a reproduction of
the depth-two paper result. It requires the separately learned
[`UITron-hub/MobileDreamer`](https://huggingface.co/UITron-hub/MobileDreamer)
checkpoint at `7d229e206cb4dd7aaff4c801174d2292fd545d57`, subfolder
`merge_model`. Its adapter configuration names Qwen3-8B; the merged configuration
declares Qwen3ForCausalLM and float32. Actual tensor precision and effective
serving/KV precision must be audited separately. The example leaves those fields
null so it cannot silently run with an invented configuration. World-model calls
are serial, audited calls; upstream HTTP retries are not retained. The algorithm's
candidate-zero fallback for malformed selection text is retained. Infrastructure
errors, including errors upstream catches internally, leave the case unscored.

AAPT uses the same model object for its planner and observer. The observer gets
only eligible guards and a screenshot, a 64-token budget, and temperature 0;
it cannot create an action. The planner uses temperature 0.2 and one retry on
invalid tree output. Maximums are 10 nodes, depth 2, and 3 children per node.
Coverage must exceed the configured planner p95 budget by at least 500 ms.
Supply that budget and its reference from existing timing evidence; the runner
does not add calibration/qualification episodes. The change gate uses a
20-grayscale-level threshold, 0.5% changed pixels and a 500 ms static probe.
Planner branch confidence must be at least 0.45. Deadline age starts at the
selected screenshot's publication and is checked again before commitment.
The branch probabilities are model estimates, not asserted calibrated values.
Optional explicit `risk_horizons_ms` restrict action deadlines by risk; this
single-action adaptation does not implement open-loop action chunks.

All three consume public task text and screenshots only. Hidden case seeds,
target geometry, event labels and future schedules are not model inputs.
No detector corrects their generated coordinates or values. Their prompts and
computations differ from Agent S3, so this is a comparison of configurations,
not an isolated controller-overhead experiment. Align planner weights/revision,
runtime precision, KV settings and external services before comparing results.
The Qwen0.8B examples preserve the previously selected FP16 planner/vision/KV
identity; they do not assert that the example endpoint currently serves it.

## Setup and CPU validation

From the repository root:

```bash
.venv/bin/python studies/paper_baselines.py fetch-sources
.venv/bin/python -m pytest tests/test_paper_baselines.py -q
```

The core adapters need numpy and Pillow. AOI additionally needs the authors'
OpenAI CLIP package and ViT-B/16 weights; the example uses CPU CLIP explicitly.
MobileDreamer needs the upstream PaddleOCR dependencies, PP-DocLayoutV2 and
PaddleOCR-VL-0.9B artifacts, plus a separately served world model. Keep these
dependencies in an isolated environment and pin/check their artifacts before
evaluation. Tests replace only model transports, the CLIP encoder and OCR
inference; they exercise real upstream extraction/selection code. Tests fetch
no weights and send no network requests. Upstream integration tests skip if the
source checkouts have not been fetched.

## Prepare from existing cases

Copy and complete a configuration from `configs/paper_baselines/*.example.json`.
Supply the existing public task instruction in a text file; never put hidden
target coordinates, seed-specific code values or event times in that instruction.
For example, preparation of one existing Notification window is:

```bash
.venv/bin/python studies/paper_baselines.py prepare \
  --cases runs/notification_qwen08_edgellm0111_20261031/experiment/cases/notification_w1000000000.jsonl \
  --config /absolute/path/to/completed-config.json \
  --task notification \
  --instruction-file /absolute/path/to/public-task-instruction.txt \
  --out runs/paper_baseline_notification_new
.venv/bin/python studies/paper_baselines.py check \
  --out runs/paper_baseline_notification_new
```

This freezes the exact case file, configuration, local source hashes and a
Live/Replay/Paused job per case. It adds no cases, prerequisites or successor
jobs. The existing Notification distributional renderer and evidence hashes are
retained. Moving Balloon accepts only the established `moving_target` drift
scenario. Code, Button and Trading retain their original deadlines/change times.
Typed values are harness actions in these new host adapters; no native typing
support is added. Other distributional overlays are rejected rather than ignored.
Large models cannot use sub-second Notification windows.

`check` validates manifests/source contracts only. It does not verify that model
services are running or that the claimed checkpoint/precision matches their
launch provenance. Complete that setup audit before an actual evaluation.
The explicit execution command is:

```bash
.venv/bin/python studies/paper_baselines.py run \
  --out runs/paper_baseline_notification_new
```

Execution refuses an existing execution directory, measures real latency and
records actual completions in `execution/records.json`. It preserves request text,
image hashes, token usage, raw responses and request durations in per-case call
audits, including failed requests. Replay uses its own matching Live trace;
the placeholder hash in the frozen plan is replaced only with that trace's hash
in the executed spec. Model-format failures remain no-action evaluation outcomes.
Infrastructure failures stop with retained evidence and an explicit status; they
do not count as model failures. There is no automatic resume, service replacement,
competence gate, or chained model batch.
