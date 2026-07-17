from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config import NOMINAL_FREQUENCIES_HZ
from device_profile import DeviceProfile
from quality_constraints import artifact_values, quality_constraint_masks
from response_metrics import calculate_response_error_metrics


ZONES_HZ = {
    "Subbass_diagnostic": (20.0, 50.0, False),
    "Bass": (50.0, 100.0, True),
    "Midbass": (100.0, 250.0, True),
    "Lower_mid": (250.0, 500.0, True),
    "Mid": (500.0, 2000.0, True),
    "Presence": (2000.0, 5000.0, True),
    "Treble": (5000.0, 10000.0, True),
    "Air_diagnostic": (10000.0, 16000.0, False),
}

EVALUATION_RANGES_HZ = {
    "Bass_100_315": (100.0, 315.0, True),
    "Lower_mid_400_800": (400.0, 800.0, True),
    "Mid_1000_2000": (1000.0, 2000.0, True),
    "Critical_2500_5000": (2500.0, 5000.0, True),
    "Treble_6300_10000": (6300.0, 10000.0, True),
    "Air_12500_16000_diagnostic": (12500.0, 16000.0, False),
}


@dataclass(frozen=True)
class ScoreResult:
    total_cost: float
    target_error_db: float
    mean_absolute_error_db: float
    median_absolute_error_db: float
    percentile_75_absolute_error_db: float
    maximum_absolute_error_db: float
    maximum_positive_deviation_db: float
    maximum_negative_deviation_db: float
    mean_signed_deviation_db: float
    minimum_response_db: float
    maximum_response_db: float
    points_within_3_db: int
    band_count: int
    broad_peak_error_db: float
    deficit_error_db: float
    spatial_variance_penalty: float
    positive_gain_penalty: float
    confidence_penalty: float
    instability_penalty: float
    experimental_distortion_limited_band_count: int
    instability_limited_band_count: int
    observed_low_hz: float
    observed_high_hz: float
    optimization_low_hz: float
    optimization_high_hz: float
    observed_frequency_mask: list[bool]
    target_optimization_mask: list[bool]
    confidence_weights: list[float]
    zone_errors_db: dict[str, float | None]
    evaluation_range_metrics: dict[str, dict[str, float | int] | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def confidence_weights(quality: dict[str, object] | None) -> np.ndarray:
    if quality is None or quality.get("band_confidence_weight") is None:
        return np.full(31, 0.5, dtype=np.float64)
    weights = np.asarray(quality["band_confidence_weight"], dtype=np.float64)
    if weights.shape != (31,):
        raise ValueError("Band confidence must contain 31 values")
    return weights


def observed_frequency_mask(
    response_db: np.ndarray,
    profile: DeviceProfile,
    weights: np.ndarray,
) -> np.ndarray:
    frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ, dtype=np.float64)
    lower_limit = profile.sweep_start_hz if profile.has_subwoofer else max(40.0, profile.sweep_start_hz)
    upper_limit = min(18_000.0, profile.sweep_end_hz)
    reference = (frequencies >= 250.0) & (frequencies <= 4000.0) & (weights > 0)
    if not np.any(reference):
        reference = (frequencies >= 250.0) & (frequencies <= 4000.0)
    reference_level = float(np.median(response_db[reference]))
    range_mask = (frequencies >= lower_limit) & (frequencies <= upper_limit)
    rolloff_mask = response_db >= reference_level - 18.0
    return range_mask & rolloff_mask & (weights > 0)


def target_optimization_mask(
    observed: np.ndarray,
    profile: DeviceProfile,
    quality: dict[str, object] | None = None,
) -> np.ndarray:
    frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ, dtype=np.float64)
    microphone_low, microphone_high = profile.microphone_profile.target_optimization_range_hz
    distortion_ok, repeatability_ok, level_ok = quality_constraint_masks(profile, quality)
    return (
        observed
        & (frequencies >= max(50.0, microphone_low))
        & (frequencies <= min(10_000.0, microphone_high))
        & distortion_ok
        & repeatability_ok
        & level_ok
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if not np.any(weights > 0):
        return None
    return float(np.average(values, weights=weights))


def score_response(
    response_db: np.ndarray,
    target_db: np.ndarray,
    profile: DeviceProfile,
    quality: dict[str, object] | None = None,
    spatial_std_db: np.ndarray | None = None,
    eq_settings: dict[str, float] | None = None,
    comparison_mask: np.ndarray | None = None,
    comparison_weights: np.ndarray | None = None,
) -> ScoreResult:
    response = np.asarray(response_db, dtype=np.float64)
    target = np.asarray(target_db, dtype=np.float64)
    if response.shape != (31,) or target.shape != (31,):
        raise ValueError("Scoring requires 31 third-octave bands")
    frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ, dtype=np.float64)
    confidence = confidence_weights(quality)
    observed = observed_frequency_mask(response, profile, confidence)
    microphone_low, microphone_high = profile.microphone_profile.target_optimization_range_hz
    base_target = (
        observed
        & (frequencies >= max(50.0, microphone_low))
        & (frequencies <= min(10_000.0, microphone_high))
    )
    scored = target_optimization_mask(observed, profile, quality)
    score_weights = confidence
    if comparison_mask is not None:
        scored = np.asarray(comparison_mask, dtype=bool)
        if scored.shape != (31,) or not np.any(scored):
            raise ValueError("Comparison mask must contain at least one of 31 bands")
    if comparison_weights is not None:
        score_weights = np.asarray(comparison_weights, dtype=np.float64)
        if score_weights.shape != (31,) or np.any(score_weights < 0):
            raise ValueError("Comparison weights must contain 31 non-negative values")
    residual = response - target
    peaks = np.maximum(residual, 0.0)
    deficits = np.maximum(-residual, 0.0)
    broad_deficit = np.zeros(31, dtype=bool)
    for index in np.flatnonzero(scored & (deficits > 0.5)):
        neighbors = [neighbor for neighbor in (index - 1, index + 1) if 0 <= neighbor < 31]
        broad_deficit[index] = any(scored[neighbor] and deficits[neighbor] > 0.5 for neighbor in neighbors)
    deficit_weight = np.where(broad_deficit, 0.75, 0.25)
    correction_cost = 1.25 * peaks + deficit_weight * deficits
    absolute_error = np.abs(residual)
    response_metrics = calculate_response_error_metrics(response, target, scored)
    target_error = _weighted_mean(correction_cost[scored], score_weights[scored])
    if target_error is None:
        raise ValueError("No reliable bands remain in the usable frequency range")
    broad_peak_error = _weighted_mean(peaks[scored], score_weights[scored]) or 0.0
    deficit_error = _weighted_mean(deficits[scored], score_weights[scored]) or 0.0
    zone_errors: dict[str, float | None] = {}
    for name, (lower, upper, optimized) in ZONES_HZ.items():
        base_mask = scored if optimized else observed
        mask = base_mask & (frequencies >= lower) & (frequencies < upper)
        zone_weights = score_weights[mask]
        zone_errors[name] = _weighted_mean(absolute_error[mask], zone_weights)
        if not optimized and zone_errors[name] is not None:
            zone_errors[name] = round(float(zone_errors[name]), 6)
    evaluation_ranges: dict[str, dict[str, float | int] | None] = {}
    for name, (lower, upper, optimized) in EVALUATION_RANGES_HZ.items():
        base_mask = scored if optimized else observed
        range_mask = base_mask & (frequencies >= lower) & (frequencies <= upper)
        evaluation_ranges[name] = (
            calculate_response_error_metrics(response, target, range_mask).to_dict()
            if np.any(range_mask)
            else None
        )
    spatial_penalty = 0.0
    if spatial_std_db is not None:
        spatial = np.asarray(spatial_std_db, dtype=np.float64)
        mean_spatial = _weighted_mean(spatial[scored], score_weights[scored])
        spatial_penalty = 0.35 * (mean_spatial or 0.0)
    positive_penalty = 0.0
    if eq_settings:
        positive_penalty = 0.03 * sum(max(0.0, value) for value in eq_settings.values())
    low_confidence_ratio = 1.0 - float(np.mean(score_weights[scored]))
    confidence_penalty = low_confidence_ratio * 2.0
    distortion_ok, repeatability_ok, level_ok = quality_constraint_masks(profile, quality)
    instability_penalty = 0.0
    repeatability = artifact_values(profile, "repeatability", "band_standard_deviation_db")
    level_shape = artifact_values(profile, "level_linearity", "band_shape_maximum_deviation_db")
    for values in (repeatability, level_shape):
        if values is not None:
            valid = base_target & np.isfinite(values)
            if np.any(valid):
                instability_penalty += 0.25 * float(np.mean(values[valid]))
    observed_frequencies = frequencies[observed]
    optimized_frequencies = frequencies[scored]
    return ScoreResult(
        total_cost=(
            target_error
            + spatial_penalty
            + positive_penalty
            + confidence_penalty
            + instability_penalty
        ),
        target_error_db=target_error,
        **response_metrics.to_dict(),
        broad_peak_error_db=broad_peak_error,
        deficit_error_db=deficit_error,
        spatial_variance_penalty=spatial_penalty,
        positive_gain_penalty=positive_penalty,
        confidence_penalty=confidence_penalty,
        instability_penalty=instability_penalty,
        experimental_distortion_limited_band_count=int(np.sum(base_target & ~distortion_ok)),
        instability_limited_band_count=int(
            np.sum(base_target & (~repeatability_ok | ~level_ok))
        ),
        observed_low_hz=float(observed_frequencies[0]) if len(observed_frequencies) else float("nan"),
        observed_high_hz=float(observed_frequencies[-1]) if len(observed_frequencies) else float("nan"),
        optimization_low_hz=float(optimized_frequencies[0]),
        optimization_high_hz=float(optimized_frequencies[-1]),
        observed_frequency_mask=observed.tolist(),
        target_optimization_mask=scored.tolist(),
        confidence_weights=confidence.tolist(),
        zone_errors_db=zone_errors,
        evaluation_range_metrics=evaluation_ranges,
    )
