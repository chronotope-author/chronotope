"""Distributional gym evaluation (opt-in; cards/D1-distributional-gym.md).

Nothing in this package is imported by the existing harness, studies or scorer. It reuses run_trial, the
scripted and VLM controllers, the trace format and stats.py; it adds frozen model-independent episode manifests,
a single receipt-time scorer and the aggregate tables of the request.
"""
from .episode import (CONDITIONS, FAILURE_MODES, SCORER, ControllerSpec, EpisodeOutcome, check_fairness, load_outcomes,
                      run_episode, score_trace, trace_path_for, verify_trace_evidence)
from .manifest import (EVENT_NS, MANIFEST_SCHEMA, ROW_FIELDS, TASK_FAMILIES, EpisodeRow, GenConfig, ManifestSet,
                       build_manifest, fixed_gates_ns, generate, generate_row, load_empirical_windows, proxy_scenario,
                       verify_row_against_manifest)
from .metrics import (deadline_curve, episode_success, fit_logistic, negative_control_check, pairwise, percentiles,
                      slack_curve, speed_curve, summarize_controller, wilson_interval)
from .report import build_report, dump_summary, negative_controls_report
from .scenarios import UPDATE_TYPES, DistMovingTargetManifest, DistRotatingCodeManifest

__all__ = [n for n in dir() if not n.startswith("_")]
