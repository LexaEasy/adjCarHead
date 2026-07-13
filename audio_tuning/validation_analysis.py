from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ, NOMINAL_FREQUENCIES_HZ
from device_profile import DeviceProfile
from response_alignment import align_response_to_target
from result_validation import require_valid_ess_result
from targets import target_curve_db


VALIDATION_SCHEMA_VERSION = "automotive_validation_result_v1"
COMMON_FIELDS = (
    "device_profile_id",
    "device_profile_schema",
    "microphone_profile_id",
    "input_device",
    "output_device",
    "sample_rate",
    "processing_settings",
    "measurement_mode",
    "mic_position_id",
    "channel_selection",
    "channel_routing_verified",
)


def load_ess_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Validation requires an accepted ESS result: {path}")
    require_valid_ess_result(payload, expected_mode="quick")
    raw = np.asarray(payload.get("raw_response_db"), dtype=np.float64)
    if raw.shape != (31,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"Invalid ESS response: {path}")
    return payload


def _measurement(payload: dict[str, object]) -> dict[str, object]:
    value = payload.get("measurement")
    if not isinstance(value, dict):
        raise ValueError("Validation ESS results require measurement metadata")
    return value


def _require_common(
    payloads: list[dict[str, object]],
    include_volume: bool,
    include_eq: bool = True,
) -> None:
    for payload in payloads:
        require_valid_ess_result(payload, expected_mode="quick")
    fields = COMMON_FIELDS
    if include_volume:
        fields += ("volume_note",)
    if include_eq:
        fields += ("eq_settings",)
    reference = {field: _measurement(payloads[0]).get(field) for field in fields}
    if any(value is None for value in reference.values()):
        raise ValueError("Validation metadata is incomplete")
    for payload in payloads[1:]:
        current = {field: _measurement(payload).get(field) for field in fields}
        if current != reference:
            raise ValueError("Validation measurements do not share the required invariants")
    result_fields = (
        "analysis_schema_version",
        "analysis_method",
        "source_signal_id",
        "inverse_filter_id",
        "ess_parameters",
    )
    result_reference = {field: payloads[0].get(field) for field in result_fields}
    for payload in payloads[1:]:
        if {field: payload.get(field) for field in result_fields} != result_reference:
            raise ValueError("Validation ESS artifacts do not share the required invariants")
    if reference["measurement_mode"] != "quick":
        raise ValueError("One-time validation series must contain quick measurements")


def _raw(payload: dict[str, object]) -> np.ndarray:
    return np.asarray(payload["raw_response_db"], dtype=np.float64)


def _aligned_rows(payloads: list[dict[str, object]], profile: DeviceProfile) -> np.ndarray:
    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    return np.vstack(
        [align_response_to_target(_raw(payload), target).aligned_response_db for payload in payloads]
    )


def analyze_repeatability(
    payloads: list[dict[str, object]],
    profile: DeviceProfile,
) -> dict[str, object]:
    if len(payloads) < 3:
        raise ValueError("Repeatability validation requires at least three quick results")
    _require_common(payloads, include_volume=True)
    rows = _aligned_rows(payloads, profile)
    mean = np.mean(rows, axis=0)
    std = np.std(rows, axis=0, ddof=1)
    maximum_deviation = np.max(np.abs(rows - mean), axis=0)
    frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ)
    evaluated = (frequencies >= 100.0) & (frequencies <= 10_000.0)
    limit = profile.quality_limits.get("max_repeatability_std_db", 1.0)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "kind": "repeatability",
        "device_profile_id": profile.device_id,
        "accepted": bool(np.all(std[evaluated] <= limit)),
        "measurement_count": len(payloads),
        "frequencies_hz": ANALYSIS_FREQUENCIES_HZ,
        "band_standard_deviation_db": std.tolist(),
        "band_maximum_deviation_db": maximum_deviation.tolist(),
        "evaluation_mask": evaluated.tolist(),
        "limit_db": limit,
    }


def analyze_level_linearity(
    ranked_payloads: list[tuple[float, dict[str, object]]],
    profile: DeviceProfile,
) -> dict[str, object]:
    if len(ranked_payloads) < 3:
        raise ValueError("Level validation requires at least three ranked quick results")
    ranked = sorted(ranked_payloads, key=lambda item: item[0])
    payloads = [payload for _, payload in ranked]
    _require_common(payloads, include_volume=False)
    rows = _aligned_rows(payloads, profile)
    mean = np.mean(rows, axis=0)
    shape_deviation = np.max(np.abs(rows - mean), axis=0)
    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    offsets = [align_response_to_target(_raw(payload), target).offset_db for payload in payloads]
    monotonic = bool(np.all(np.diff(offsets) > 0.5))
    frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ)
    evaluated = (frequencies >= 100.0) & (frequencies <= 10_000.0)
    limit = profile.quality_limits.get("max_level_shape_deviation_db", 1.0)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "kind": "level_linearity",
        "device_profile_id": profile.device_id,
        "accepted": bool(monotonic and np.all(shape_deviation[evaluated] <= limit)),
        "level_ranks": [rank for rank, _ in ranked],
        "relative_level_offsets_db": offsets,
        "level_response_monotonic": monotonic,
        "frequencies_hz": ANALYSIS_FREQUENCIES_HZ,
        "band_shape_maximum_deviation_db": shape_deviation.tolist(),
        "evaluation_mask": evaluated.tolist(),
        "limit_db": limit,
    }
