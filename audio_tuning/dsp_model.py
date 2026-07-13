from __future__ import annotations

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import DeviceProfile
from dsp_matrix import load_dsp_response_matrix
from scoring import confidence_weights, observed_frequency_mask, score_response
from targets import target_curve_db


def _quantize(value: float, step_db: float, limit_db: float) -> float:
    limited = float(np.clip(value, -limit_db, limit_db))
    return round(round(limited / step_db) * step_db, 6)


def _interpolate(source_hz: np.ndarray, values: np.ndarray) -> np.ndarray:
    destination = np.asarray(ANALYSIS_FREQUENCIES_HZ, dtype=np.float64)
    return np.interp(np.log(destination), np.log(source_hz), values)


def _matrix_context(measurement: dict[str, object]) -> dict[str, object]:
    return {
        field: measurement.get(field)
        for field in (
            "device_profile_id",
            "system_profile_hash",
            "microphone_profile_hash",
            "microphone_profile_id",
            "input_device",
            "output_device",
            "sample_rate",
            "analysis_schema_version",
            "source_signal_id",
        )
    }


def suggest_dsp(
    frequencies_hz: np.ndarray,
    response_db: np.ndarray,
    target_db: np.ndarray,
    optimization_mask: np.ndarray,
    profile: DeviceProfile,
    current_eq: dict[str, float],
    spatial_std_db: np.ndarray,
    quality: dict[str, object],
    measurement_context: dict[str, object],
) -> dict[str, object]:
    if not profile.dsp_recommendation_eligible or profile.dsp_matrix_file is None:
        raise ValueError(profile.dsp_recommendation_block_reason())
    control_ids = [control.control_id for control in profile.dsp_controls]
    context = _matrix_context(measurement_context)
    context["analysis_schema_version"] = measurement_context.get("analysis_schema_version")
    matrix = load_dsp_response_matrix(profile.dsp_matrix_file, set(control_ids), context)
    if matrix.device_profile_id != profile.device_id:
        raise ValueError("DSP matrix belongs to another device profile")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.shape != matrix.frequencies_hz.shape or not np.allclose(
        frequencies, matrix.frequencies_hz
    ):
        raise ValueError("DSP matrix frequency grid does not match the full measurement")
    influence = np.column_stack([matrix.response_per_db[item] for item in control_ids])
    mask = np.asarray(optimization_mask, dtype=bool)
    if mask.shape != frequencies.shape or np.sum(mask) < len(control_ids):
        raise ValueError("Not enough reliable bands for matrix-based DSP optimization")
    response = np.asarray(response_db, dtype=np.float64)
    target = np.asarray(target_db, dtype=np.float64)
    spatial = np.asarray(spatial_std_db, dtype=np.float64)
    weights = 1.0 / (1.0 + np.square(spatial[mask]))
    weighted_matrix = influence[mask] * np.sqrt(weights)[:, np.newaxis]
    weighted_error = (target[mask] - response[mask]) * np.sqrt(weights)
    ridge = 0.05
    augmented = np.vstack((weighted_matrix, np.sqrt(ridge) * np.eye(len(control_ids))))
    requested, *_ = np.linalg.lstsq(
        augmented,
        np.concatenate((weighted_error, np.zeros(len(control_ids)))),
        rcond=None,
    )

    max_delta = profile.quality_limits.get("max_delta_per_control_db", 0.3)
    max_controls = int(profile.quality_limits.get("max_controls_per_iteration", 6))
    proposed = np.zeros(len(control_ids), dtype=np.float64)
    reasons = ["Изменение ниже минимального шага"] * len(control_ids)
    for index, control in enumerate(profile.dsp_controls):
        column = influence[:, index]
        affected = mask & (np.abs(column) >= 0.25 * max(float(np.max(np.abs(column))), 1e-9))
        step = _quantize(float(requested[index]), control.step_db, max_delta)
        if step > 0 and np.any(affected) and float(np.mean(spatial[affected])) > 3.0:
            reasons[index] = "Подъём запрещён: пространственный разброс выше 3 dB"
            continue
        proposed[index] = step
        if step:
            reasons[index] = (
                "Измеренная матрица DSP; ошибка симметрии "
                f"{matrix.symmetry_error_db_per_db[control.control_id]:.2f} dB/dB"
            )
    ranked = sorted(
        np.flatnonzero(proposed),
        key=lambda index: (
            proposed[index] > 0,
            matrix.symmetry_error_db_per_db[control_ids[index]],
            -abs(proposed[index]),
        ),
    )
    for index in ranked[max_controls:]:
        proposed[index] = 0.0
        reasons[index] = "Отложено из-за лимита числа регуляторов на итерацию"

    new_eq = dict(current_eq)
    suggestions = []
    for index, control in enumerate(profile.dsp_controls):
        old = float(current_eq[control.control_id])
        new = float(np.clip(old + proposed[index], control.minimum_db, control.maximum_db))
        new = round(round(new / control.step_db) * control.step_db, 6)
        proposed[index] = new - old
        new_eq[control.control_id] = new
        suggestions.append(
            {"control": control.control_id, "label": control.label, "old": old, "new": new, "reason": reasons[index]}
        )

    predicted = response + influence @ proposed
    response_31 = _interpolate(frequencies, response)
    predicted_31 = _interpolate(frequencies, predicted)
    spatial_31 = _interpolate(frequencies, spatial)
    target_31 = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    current_score = score_response(response_31, target_31, profile, quality, spatial_31, current_eq)
    comparison_mask = np.asarray(current_score.target_optimization_mask, dtype=bool)
    comparison_weights = np.asarray(current_score.confidence_weights, dtype=np.float64)
    predicted_observed = observed_frequency_mask(
        predicted_31,
        profile,
        confidence_weights(quality),
    )
    predicted_score = score_response(
        predicted_31,
        target_31,
        profile,
        quality,
        spatial_31,
        new_eq,
        comparison_mask,
        comparison_weights,
    )
    current_score = score_response(
        response_31,
        target_31,
        profile,
        quality,
        spatial_31,
        current_eq,
        comparison_mask,
        comparison_weights,
    )
    score_improvement = current_score.total_cost - predicted_score.total_cost
    zone_limit = profile.quality_limits.get("max_predicted_zone_regression_db", 0.5)
    zone_regressions = [
        name
        for name, current in current_score.zone_errors_db.items()
        if current is not None
        and predicted_score.zone_errors_db.get(name) is not None
        and float(predicted_score.zone_errors_db[name]) - current > zone_limit
    ]
    lost = comparison_mask & ~predicted_observed
    total_positive = float(np.sum(np.maximum(proposed, 0.0)))
    peak_dbfs = float(quality.get("worst_peak_dbfs", quality.get("peak_dbfs", -6.0)))
    predicted_headroom = -peak_dbfs - total_positive
    rejections = []
    if score_improvement < profile.quality_limits.get("minimum_score_improvement", 0.1):
        rejections.append("insufficient_predicted_score_improvement")
    if zone_regressions:
        rejections.append("protected_zone_regression")
    if np.any(lost):
        rejections.append("frequency_coverage_regression")
    if predicted_score.broad_peak_error_db - current_score.broad_peak_error_db > 0.25:
        rejections.append("new_broad_peak")
    if total_positive > profile.quality_limits.get("max_total_positive_delta_db", 0.6):
        rejections.append("total_positive_gain_limit")
    if predicted_headroom < profile.quality_limits.get("minimum_predicted_headroom_db", 6.0):
        rejections.append("predicted_headroom_limit")
    return {
        "suggestions": suggestions,
        "proposed_delta_db": proposed.tolist(),
        "predicted_response_db": predicted.tolist(),
        "current_score": current_score.to_dict(),
        "predicted_score": predicted_score.to_dict(),
        "predicted_score_delta": score_improvement,
        "predicted_zone_regressions": zone_regressions,
        "frequency_coverage_regression": bool(np.any(lost)),
        "total_positive_gain_db": total_positive,
        "changed_control_count": int(np.sum(np.abs(proposed) > 1e-9)),
        "predicted_headroom_db": predicted_headroom,
        "recommendation_accepted": not rejections,
        "recommendation_rejection_reasons": rejections,
    }
