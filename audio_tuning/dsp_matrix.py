from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

DSP_MATRIX_SCHEMA_VERSION = "dsp_response_matrix_v1"


@dataclass(frozen=True)
class DspResponseMatrix:
    source_path: Path
    device_profile_id: str
    frequencies_hz: np.ndarray
    response_per_db: dict[str, np.ndarray]
    symmetry_error_db_per_db: dict[str, float]


def load_dsp_response_matrix(
    path: Path,
    expected_control_ids: set[str] | None = None,
    expected_context: dict[str, object] | None = None,
) -> DspResponseMatrix:
    resolved = path.resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != DSP_MATRIX_SCHEMA_VERSION:
        raise ValueError(f"Expected DSP matrix schema {DSP_MATRIX_SCHEMA_VERSION}")
    if data.get("accepted") is not True:
        raise ValueError("DSP matrix has not passed characterization")
    required_metadata = (
        "device_profile_id",
        "system_profile_hash",
        "microphone_profile_hash",
        "microphone_profile_id",
        "input_device",
        "output_device",
        "sample_rate",
        "analysis_schema_version",
        "source_signal_id",
        "baseline_measurement_id",
        "source_measurement_ids",
    )
    if any(data.get(field) in (None, "", []) for field in required_metadata):
        raise ValueError("DSP matrix metadata is incomplete")
    if expected_context is not None:
        changed = [
            field
            for field, expected in expected_context.items()
            if expected is not None and data.get(field) != expected
        ]
        if changed:
            raise ValueError("DSP matrix context mismatch: " + ", ".join(changed))
    frequencies = np.asarray(data.get("frequencies_hz"), dtype=np.float64)
    if (
        frequencies.ndim != 1
        or len(frequencies) < 31
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.diff(frequencies) > 0)
    ):
        raise ValueError("DSP matrix requires an increasing frequency grid")
    controls = data.get("controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("DSP matrix requires controls")
    response: dict[str, np.ndarray] = {}
    symmetry: dict[str, float] = {}
    for item in controls:
        if not isinstance(item, dict):
            raise ValueError("DSP matrix control must be an object")
        control_id = str(item.get("id", ""))
        if control_id in response:
            raise ValueError(f"Duplicate DSP matrix control: {control_id}")
        if item.get("accepted") is not True:
            raise ValueError(f"DSP control {control_id} has not passed characterization")
        if item.get("units") != "db_response_per_db_control":
            raise ValueError(f"DSP control {control_id} has invalid units")
        if not float(item.get("plus_delta_db", 0.0)) > 0.0 or not float(
            item.get("minus_delta_db", 0.0)
        ) < 0.0:
            raise ValueError(f"DSP control {control_id} has invalid characterization deltas")
        measurement_fields = (
            "baseline_measurement_id",
            "plus_measurement_id",
            "minus_measurement_id",
        )
        if any(not item.get(field) for field in measurement_fields):
            raise ValueError(f"DSP control {control_id} has incomplete measurement references")
        values = np.asarray(item.get("response_per_db"), dtype=np.float64)
        if not control_id or values.shape != frequencies.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid DSP matrix control: {control_id}")
        for field in ("boost_response_per_db", "cut_response_per_db"):
            branch = np.asarray(item.get(field), dtype=np.float64)
            if branch.shape != frequencies.shape or not np.all(np.isfinite(branch)):
                raise ValueError(f"DSP control {control_id} has invalid {field}")
        symmetry_value = float(item.get("symmetry_error_db_per_db", np.nan))
        if not np.isfinite(symmetry_value):
            raise ValueError(f"DSP control {control_id} has invalid symmetry error")
        response[control_id] = values
        symmetry[control_id] = symmetry_value
    if expected_control_ids is not None and set(response) != expected_control_ids:
        raise ValueError("DSP matrix controls do not match the device profile")
    return DspResponseMatrix(
        source_path=resolved,
        device_profile_id=str(data.get("device_profile_id", "")),
        frequencies_hz=frequencies,
        response_per_db=response,
        symmetry_error_db_per_db=symmetry,
    )
