from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from device_profile import DeviceProfile
from quality_constraints import artifact_scalar, artifact_values
from scoring import ScoreResult


@dataclass(frozen=True)
class CompletionAssessment:
    status: str
    assessed_profile: str
    soft_targets_met: bool
    meaningful_improvement_threshold_db: float | None
    score_improvement_db: float
    repeatability_percentile_75_db: float | None
    spatial_percentile_75_db: float | None
    persistent_defect_proxy_db: float
    maximum_eq_gain_db: float
    total_eq_correction_db: float
    positive_eq_gain_db: float
    active_eq_filter_count: int
    deep_eq_filter_count: int
    measured_headroom_db: float | None
    phase_sum_improvement_db: float | None
    phase_check_status: str
    requires_listening_confirmation: bool
    missing_evidence: tuple[str, ...]
    unmet_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: np.ndarray | None, mask: np.ndarray) -> float | None:
    if values is None:
        return None
    selected = values[mask & np.isfinite(values)]
    return float(np.percentile(selected, 75)) if len(selected) else None


def assess_completion(
    score: ScoreResult,
    profile: DeviceProfile,
    spatial_std_db: np.ndarray,
    eq_settings: dict[str, object],
    quality: dict[str, object],
    score_improvement_db: float,
    worst_zone_regression_db: float,
    frequency_coverage_regression: bool,
    assessed_profile: str,
) -> CompletionAssessment:
    limits = profile.completion_limits
    mask = np.asarray(score.target_optimization_mask, dtype=bool)
    spatial = np.asarray(spatial_std_db, dtype=np.float64)
    if spatial.shape != mask.shape:
        raise ValueError("Completion assessment requires spatial data for every score band")
    repeatability = artifact_values(
        profile, "repeatability", "band_standard_deviation_db"
    )
    repeatability_p75 = _percentile(repeatability, mask)
    spatial_p75 = _percentile(spatial, mask)
    missing = []
    if repeatability_p75 is None:
        missing.append("repeatability_artifact")
    if spatial_p75 is None:
        missing.append("spatial_stability")

    phase_status = profile.phase_alignment_status
    phase_sum_improvement = None
    if profile.phase_alignment_eligible:
        phase_sum_improvement = artifact_scalar(
            profile, "phase_alignment", "summed_level_improvement_db"
        )
        if phase_sum_improvement is None:
            missing.append("phase_sum_measurement")
        elif phase_sum_improvement >= limits.get("minimum_phase_sum_improvement_db", 1.0):
            phase_status = "verified"
        else:
            phase_status = "failed"

    gains = np.asarray([float(value) for value in eq_settings.values()], dtype=np.float64)
    maximum_gain = float(np.max(np.abs(gains))) if len(gains) else 0.0
    total_correction = float(np.sum(np.abs(gains)))
    positive_gain = float(np.sum(np.maximum(gains, 0.0)))
    active_filter_count = int(np.sum(np.abs(gains) > 1e-9))
    deep_filter_count = int(
        np.sum(np.abs(gains) > limits.get("maximum_eq_gain_db", 6.0))
    )
    peak = quality.get("worst_peak_dbfs", quality.get("peak_dbfs"))
    headroom = -float(peak) if peak is not None else None
    if headroom is None:
        missing.append("measured_headroom")

    threshold = None
    if repeatability_p75 is not None:
        threshold = max(
            limits.get("minimum_measured_score_improvement_db", 0.3),
            repeatability_p75,
        )
    unmet = []
    checks = (
        (score.mean_absolute_error_db <= limits.get("target_mean_absolute_error_db", 3.0), "mae"),
        (
            score.median_absolute_error_db
            <= limits.get("target_median_absolute_error_db", 2.5),
            "median_absolute_error",
        ),
        (
            score.percentile_75_absolute_error_db
            <= limits.get("target_percentile_75_absolute_error_db", 3.5),
            "percentile_75_absolute_error",
        ),
        (
            score.maximum_absolute_error_db
            <= limits.get("maximum_persistent_defect_db", 8.0),
            "persistent_defect_proxy",
        ),
        (deep_filter_count == 0, "deep_eq_filters"),
        (
            worst_zone_regression_db
            <= limits.get("maximum_zone_regression_db", 0.5),
            "zone_regression",
        ),
        (not frequency_coverage_regression, "frequency_coverage"),
    )
    unmet.extend(name for passed, name in checks if not passed)
    if repeatability_p75 is not None and repeatability_p75 > limits.get(
        "maximum_repeatability_percentile_75_db", 1.0
    ):
        unmet.append("repeatability")
    if spatial_p75 is not None and spatial_p75 > limits.get(
        "maximum_spatial_percentile_75_db", 2.0
    ):
        unmet.append("spatial_stability")
    if headroom is not None and headroom < limits.get("minimum_measured_headroom_db", 6.0):
        unmet.append("headroom")
    if phase_sum_improvement is not None and phase_sum_improvement < limits.get(
        "minimum_phase_sum_improvement_db", 1.0
    ):
        unmet.append("phase_sum_improvement")

    plateau = threshold is not None and score_improvement_db <= threshold
    soft_targets_met = not missing and not unmet
    if missing:
        status = "insufficient_evidence"
    elif soft_targets_met and plateau:
        status = "stop_recommended"
    else:
        status = "continue"
    return CompletionAssessment(
        status=status,
        assessed_profile=assessed_profile,
        soft_targets_met=soft_targets_met,
        meaningful_improvement_threshold_db=threshold,
        score_improvement_db=score_improvement_db,
        repeatability_percentile_75_db=repeatability_p75,
        spatial_percentile_75_db=spatial_p75,
        persistent_defect_proxy_db=score.maximum_absolute_error_db,
        maximum_eq_gain_db=maximum_gain,
        total_eq_correction_db=total_correction,
        positive_eq_gain_db=positive_gain,
        active_eq_filter_count=active_filter_count,
        deep_eq_filter_count=deep_filter_count,
        measured_headroom_db=headroom,
        phase_sum_improvement_db=phase_sum_improvement,
        phase_check_status=phase_status,
        requires_listening_confirmation=True,
        missing_evidence=tuple(missing),
        unmet_criteria=tuple(unmet),
    )
