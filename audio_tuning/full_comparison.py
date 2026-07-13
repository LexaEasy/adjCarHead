from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import NOMINAL_FREQUENCIES_HZ
from device_profile import DeviceProfile
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from scoring import confidence_weights, observed_frequency_mask, score_response
from spatial_positions import SPATIAL_SCHEMA_VERSION
from tuning_state import TuningState, require_transition


FULL_INVARIANTS = (
    "input_device",
    "output_device",
    "sample_rate",
    "volume_note",
    "device_profile_id",
    "device_profile_schema",
    "microphone_profile_id",
    "processing_settings",
    "measurement_mode",
    "system_profile_hash",
    "microphone_profile_hash",
    "source_signal_id",
    "inverse_filter_id",
    "ess_parameters",
    "clock_correction",
    "channel_selection",
    "channel_routing_verified",
    "analysis_schema_version",
)


def _load(path: Path, purpose: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("spatial_schema_version") != SPATIAL_SCHEMA_VERSION
        or payload.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION
        or payload.get("spatial_aggregate_complete") is not True
        or payload.get("session_purpose") != purpose
    ):
        raise ValueError(f"Invalid {purpose} full spatial result")
    expected_state = f"full_{purpose}_measured"
    if payload.get("tuning_state") != expected_state:
        raise ValueError(f"Full {purpose} result has an invalid tuning state")
    if payload.get("position_order") is None or len(payload["position_order"]) != 6:
        raise ValueError("Full comparison requires six positions")
    return payload


def write_full_comparison(
    out: Path,
    baseline_path: Path,
    candidate_path: Path,
    profile: DeviceProfile,
) -> Path:
    baseline = _load(baseline_path, "baseline")
    candidate = _load(candidate_path, "candidate")
    base_invariants = baseline.get("measurement_invariants")
    candidate_invariants = candidate.get("measurement_invariants")
    if not isinstance(base_invariants, dict) or not isinstance(candidate_invariants, dict):
        raise ValueError("Full comparison requires measurement invariants")
    changed = [
        field for field in FULL_INVARIANTS if base_invariants.get(field) != candidate_invariants.get(field)
    ]
    if changed:
        raise ValueError("Full comparison invariants changed: " + ", ".join(changed))
    base_eq = base_invariants.get("eq_settings")
    candidate_eq = candidate_invariants.get("eq_settings")
    if not isinstance(base_eq, dict) or not isinstance(candidate_eq, dict) or set(base_eq) != set(candidate_eq):
        raise ValueError("Full comparison requires matching DSP control sets")

    target = np.asarray(baseline.get("target_db"), dtype=np.float64)
    baseline_curve = np.asarray(baseline.get("aligned_mean_db"), dtype=np.float64)
    candidate_curve = np.asarray(candidate.get("aligned_mean_db"), dtype=np.float64)
    baseline_spatial = np.asarray(baseline.get("standard_deviation_db"), dtype=np.float64)
    candidate_spatial = np.asarray(candidate.get("standard_deviation_db"), dtype=np.float64)
    if any(value.shape != (31,) for value in (target, baseline_curve, candidate_curve, baseline_spatial, candidate_spatial)):
        raise ValueError("Full comparison requires 31-band spatial results")
    baseline_native = score_response(
        baseline_curve, target, profile, baseline["quality"], baseline_spatial, base_eq
    )
    comparison_mask = np.asarray(baseline_native.target_optimization_mask, dtype=bool)
    comparison_weights = np.asarray(baseline_native.confidence_weights, dtype=np.float64)
    candidate_observed = observed_frequency_mask(
        candidate_curve,
        profile,
        confidence_weights(candidate["quality"]),
    )
    lost = comparison_mask & ~candidate_observed
    baseline_score = score_response(
        baseline_curve,
        target,
        profile,
        baseline["quality"],
        baseline_spatial,
        base_eq,
        comparison_mask,
        comparison_weights,
    )
    candidate_score = score_response(
        candidate_curve,
        target,
        profile,
        candidate["quality"],
        candidate_spatial,
        candidate_eq,
        comparison_mask,
        comparison_weights,
    )
    improvement = baseline_score.total_cost - candidate_score.total_cost
    zone_delta = {
        name: candidate_score.zone_errors_db[name] - value
        for name, value in baseline_score.zone_errors_db.items()
        if value is not None and candidate_score.zone_errors_db.get(name) is not None
    }
    passed = improvement > 0.3 and max(zone_delta.values(), default=0.0) <= 0.5 and not np.any(lost)
    if passed:
        require_transition(TuningState.FULL_CANDIDATE_MEASURED, TuningState.FULL_COMPARISON_PASSED)
        require_transition(TuningState.FULL_COMPARISON_PASSED, TuningState.LISTENING_CONFIRMATION_REQUIRED)
    result = {
        "comparison_type": "full_baseline_vs_candidate",
        "baseline_result": str(baseline_path.resolve()),
        "candidate_result": str(candidate_path.resolve()),
        "channel_selection": base_invariants["channel_selection"],
        "comparison_mask_source": "baseline",
        "lost_reliable_band_count": int(np.sum(lost)),
        "lost_reliable_band_frequencies_hz": np.asarray(NOMINAL_FREQUENCIES_HZ)[lost].tolist(),
        "frequency_coverage_regression": bool(np.any(lost)),
        "baseline_score": baseline_score.to_dict(),
        "candidate_score": candidate_score.to_dict(),
        "score_improvement": improvement,
        "zone_error_delta_db": zone_delta,
        "technical_state": TuningState.FULL_COMPARISON_PASSED.value if passed else TuningState.FULL_CANDIDATE_MEASURED.value,
        "tuning_state": TuningState.LISTENING_CONFIRMATION_REQUIRED.value if passed else TuningState.FULL_CANDIDATE_MEASURED.value,
        "verdict": "technically_better_candidate" if passed else "candidate_rejected_or_inconclusive",
        "final_verdict_allowed": False,
        "final_dsp_eligible": False,
        "requires_listening_confirmation": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "full_comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
