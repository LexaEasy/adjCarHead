from __future__ import annotations

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import DeviceProfile
from dsp_matrix import DSP_MATRIX_SCHEMA_VERSION
from validation_analysis import _measurement, _raw, _require_common
from result_validation import ess_validation_manifest


def _matrix_curve(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray] | None:
    smoothed = payload.get("smoothed_response")
    if not isinstance(smoothed, dict):
        return None
    frequencies = np.asarray(smoothed.get("frequencies_hz"), dtype=np.float64)
    response = np.asarray(smoothed.get("raw_response_db"), dtype=np.float64)
    if frequencies.ndim != 1 or response.shape != frequencies.shape or not np.all(np.isfinite(response)):
        raise ValueError("Invalid smoothed ESS response for DSP characterization")
    return frequencies, response


def characterize_dsp(
    baseline: dict[str, object],
    variants: list[tuple[str, float, dict[str, object], float, dict[str, object]]],
    profile: DeviceProfile,
) -> dict[str, object]:
    all_payloads = [baseline] + [item for variant in variants for item in (variant[2], variant[4])]
    _require_common(all_payloads, include_volume=True, include_eq=False)
    baseline_curve = _matrix_curve(baseline)
    if baseline_curve is None:
        if len(profile.dsp_controls) > 31:
            raise ValueError("DSP profiles with more than 31 controls require smoothed ESS responses")
        matrix_frequencies = np.asarray(ANALYSIS_FREQUENCIES_HZ)
        baseline_raw = _raw(baseline)
    else:
        matrix_frequencies, baseline_raw = baseline_curve
    baseline_eq = _measurement(baseline).get("eq_settings")
    if not isinstance(baseline_eq, dict):
        raise ValueError("DSP baseline requires EQ settings")
    controls = []
    expected_ids = {control.control_id for control in profile.dsp_controls}
    if len(variants) != len(expected_ids) or {variant[0] for variant in variants} != expected_ids:
        raise ValueError("DSP validation must include every profile control exactly once")
    symmetry_limit = profile.quality_limits.get("max_dsp_symmetry_error_db_per_db", 0.5)
    for control_id, plus_delta, plus, minus_delta, minus in variants:
        if plus_delta <= 0 or minus_delta >= 0:
            raise ValueError("DSP characterization requires positive plus and negative minus deltas")
        for delta, payload in ((plus_delta, plus), (minus_delta, minus)):
            actual = _measurement(payload).get("eq_settings")
            expected = {str(key): float(value) for key, value in baseline_eq.items()}
            expected[control_id] = expected.get(control_id, 0.0) + delta
            if not isinstance(actual, dict) or set(actual) != set(expected) or any(
                abs(float(actual[key]) - value) > 1e-6 for key, value in expected.items()
            ):
                raise ValueError(f"DSP variant changes controls other than declared: {control_id}")
        plus_curve = _matrix_curve(plus)
        minus_curve = _matrix_curve(minus)
        if baseline_curve is None:
            plus_raw, minus_raw = _raw(plus), _raw(minus)
        else:
            if plus_curve is None or minus_curve is None:
                raise ValueError("Every DSP variant requires a smoothed ESS response")
            if not np.allclose(plus_curve[0], matrix_frequencies) or not np.allclose(
                minus_curve[0], matrix_frequencies
            ):
                raise ValueError("DSP characterization frequency grids must match")
            plus_raw, minus_raw = plus_curve[1], minus_curve[1]
        plus_effect = (plus_raw - baseline_raw) / plus_delta
        minus_effect = (minus_raw - baseline_raw) / minus_delta
        response = 0.5 * (plus_effect + minus_effect)
        symmetry = float(np.sqrt(np.mean(np.square(plus_effect - minus_effect))))
        controls.append(
            {
                "id": control_id,
                "response_per_db": response.tolist(),
                "boost_response_per_db": plus_effect.tolist(),
                "cut_response_per_db": minus_effect.tolist(),
                "symmetry_error_db_per_db": symmetry,
                "accepted": symmetry <= symmetry_limit,
                "failure_reason": None if symmetry <= symmetry_limit else "boost_cut_asymmetry",
                "units": "db_response_per_db_control",
                "plus_delta_db": plus_delta,
                "minus_delta_db": minus_delta,
                "baseline_measurement_id": baseline.get("measurement_id"),
                "plus_measurement_id": plus.get("measurement_id"),
                "minus_measurement_id": minus.get("measurement_id"),
            }
        )
    measurement = _measurement(baseline)
    source_ids = [str(payload.get("measurement_id")) for payload in all_payloads]
    if any(not value or value == "None" for value in source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("DSP characterization requires unique measurement ids")
    return {
        "schema_version": DSP_MATRIX_SCHEMA_VERSION,
        "device_profile_id": profile.device_id,
        "system_profile_hash": measurement.get("system_profile_hash"),
        "microphone_profile_hash": measurement.get("microphone_profile_hash"),
        "microphone_profile_id": measurement.get("microphone_profile_id"),
        "input_device": measurement.get("input_device"),
        "output_device": measurement.get("output_device"),
        "sample_rate": measurement.get("sample_rate"),
        "analysis_schema_version": baseline.get("analysis_schema_version"),
        "source_signal_id": baseline.get("source_signal_id"),
        "source_measurement_ids": source_ids,
        "source_validation_manifests": [ess_validation_manifest(payload) for payload in all_payloads],
        "baseline_measurement_id": baseline.get("measurement_id"),
        "frequencies_hz": matrix_frequencies.tolist(),
        "controls": controls,
        "accepted": all(bool(control["accepted"]) for control in controls),
        "symmetry_limit_db_per_db": symmetry_limit,
    }
