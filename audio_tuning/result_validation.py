from __future__ import annotations

from dataclasses import dataclass

from frequency_bands import ANALYSIS_SCHEMA_VERSION


CHANNEL_SELECTIONS = {"left", "right", "stereo"}


@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    errors: tuple[str, ...]

    def require(self) -> None:
        if self.errors:
            raise ValueError("Invalid ESS result: " + "; ".join(self.errors))


def validate_ess_result(
    result: dict[str, object],
    *,
    expected_mode: str | None = None,
    require_clock_correction: bool = True,
    require_current_schema: bool = True,
    require_quality_acceptance: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    if result.get("method") != "ess" or result.get("analysis_method") != "ess_deconvolution":
        errors.append("analysis_method")
    if require_current_schema and result.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION:
        errors.append("analysis_schema_version")
    if require_clock_correction and result.get("clock_drift_compensated") is not True:
        errors.append("clock_drift_compensated")
    if result.get("timing_markers_valid") is not True:
        errors.append("timing_markers_valid")
    for field in ("measurement_id", "source_signal_id", "inverse_filter_id"):
        if not isinstance(result.get(field), str) or not result.get(field):
            errors.append(field)
    if result.get("active_ess_complete") is not True:
        errors.append("active_ess_complete")
    if result.get("dropout_analysis_scope") != "active_ess_only":
        errors.append("dropout_analysis_scope")

    ess = result.get("ess_parameters")
    if not isinstance(ess, dict):
        errors.append("ess_parameters")
    else:
        try:
            valid_ess = all(
                float(ess.get(field, 0.0)) > 0.0 for field in ("duration_s", "start_hz", "end_hz")
            )
        except (TypeError, ValueError):
            valid_ess = False
        if not valid_ess:
            errors.append("ess_parameters")

    quality = result.get("quality")
    if not isinstance(quality, dict):
        errors.append("quality")
    elif require_quality_acceptance and (
        quality.get("accepted") is not True or bool(quality.get("hard_failures"))
    ):
        errors.append("quality_acceptance")

    measurement = result.get("measurement")
    if not isinstance(measurement, dict):
        errors.append("measurement")
    else:
        for field in (
            "device_profile_id",
            "microphone_profile_id",
            "input_device",
            "output_device",
            "sample_rate",
            "channel_selection",
        ):
            if measurement.get(field) is None:
                errors.append(field)
        if expected_mode is not None and measurement.get("measurement_mode") != expected_mode:
            errors.append("measurement_mode")
        if measurement.get("channel_selection") not in CHANNEL_SELECTIONS:
            errors.append("channel_selection")
        if not (measurement.get("mic_position_id") or measurement.get("mic_position_note")):
            errors.append("mic_position")
        if not isinstance(measurement.get("sample_rate"), int) or measurement.get("sample_rate", 0) <= 0:
            errors.append("sample_rate")
    return ValidationReport(not errors, tuple(dict.fromkeys(errors)))


def require_valid_ess_result(
    result: dict[str, object],
    *,
    expected_mode: str | None = None,
    require_quality_acceptance: bool = True,
) -> None:
    validate_ess_result(
        result,
        expected_mode=expected_mode,
        require_quality_acceptance=require_quality_acceptance,
    ).require()
