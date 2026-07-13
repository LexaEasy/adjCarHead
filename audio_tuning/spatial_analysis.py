from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import numpy as np

from config import TARGET_DB
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from response_alignment import ResponseAlignment, align_response_to_target
from result_validation import require_valid_ess_result
from spatial_positions import SPATIAL_POSITION_KEYS, SPATIAL_SCHEMA_VERSION
from spatial_quality import aggregate_spatial_quality
from spatial_smoothing import SmoothedSpatialAccumulator


INVARIANT_FIELDS = (
    "eq_settings",
    "input_device",
    "output_device",
    "sample_rate",
    "volume_note",
    "device_profile_id",
    "device_profile_schema",
    "microphone_profile_id",
    "processing_settings",
    "measurement_mode",
    "session_purpose",
    "analysis_schema_version",
    "system_profile_hash",
    "microphone_profile_hash",
    "source_signal_id",
    "inverse_filter_id",
    "ess_parameters",
    "clock_correction",
    "channel_selection",
    "channel_routing_verified",
)


@dataclass(frozen=True)
class SpatialResult:
    session_id: str
    raw_mean_db: np.ndarray
    aligned_mean_db: np.ndarray
    standard_deviation_db: np.ndarray
    p10_db: np.ndarray
    p90_db: np.ndarray
    position_raw_db: dict[str, np.ndarray]
    alignment: ResponseAlignment
    invariants: dict[str, object]
    target_name: str
    target_db: np.ndarray
    quality: dict[str, object]
    smoothed_frequencies_hz: np.ndarray | None
    smoothed_raw_mean_db: np.ndarray | None
    smoothed_aligned_mean_db: np.ndarray | None
    smoothed_standard_deviation_db: np.ndarray | None
    source_measurement_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "spatial_schema_version": SPATIAL_SCHEMA_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "session_id": self.session_id,
            "position_order": SPATIAL_POSITION_KEYS,
            "raw_mean_db": self.raw_mean_db.tolist(),
            "aligned_mean_db": self.aligned_mean_db.tolist(),
            "standard_deviation_db": self.standard_deviation_db.tolist(),
            "p10_db": self.p10_db.tolist(),
            "p90_db": self.p90_db.tolist(),
            "position_raw_db": {
                position: values.tolist() for position, values in self.position_raw_db.items()
            },
            "response_alignment": self.alignment.diagnostics(),
            "measurement_invariants": self.invariants,
            "target_profile": self.target_name,
            "target_db": self.target_db.tolist(),
            "quality": self.quality,
            "smoothed_response": {
                "fractional_octave": 6,
                "frequencies_hz": self.smoothed_frequencies_hz.tolist(),
                "raw_mean_db": self.smoothed_raw_mean_db.tolist(),
                "aligned_mean_db": self.smoothed_aligned_mean_db.tolist(),
                "standard_deviation_db": self.smoothed_standard_deviation_db.tolist(),
            }
            if self.smoothed_frequencies_hz is not None
            else None,
            "impulse_responses_averaged": False,
            "spatial_aggregate_complete": True,
            "source_measurement_ids": list(self.source_measurement_ids),
        }


def load_spatial_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Spatial result must be a JSON object: {path}")
    return payload


def aggregate_spatial_payloads(
    payloads: list[dict[str, object]],
    target_db: np.ndarray | None = None,
    target_name: str = "warm_driver",
) -> SpatialResult:
    if len(payloads) != len(SPATIAL_POSITION_KEYS):
        raise ValueError(f"Exactly {len(SPATIAL_POSITION_KEYS)} ESS results are required")

    by_position: dict[str, np.ndarray] = {}
    position_timestamps: dict[str, datetime] = {}
    session_id: str | None = None
    invariants: dict[str, object] | None = None
    qualities: list[dict[str, object]] = []
    measurement_ids: list[str] = []
    smoothed = SmoothedSpatialAccumulator()
    for payload in payloads:
        require_valid_ess_result(payload, expected_mode="full")
        measurement_ids.append(str(payload["measurement_id"]))
        quality = payload.get("quality")
        if not isinstance(quality, dict) or quality.get("clipped") or quality.get("accepted") is False:
            raise ValueError("Clipped or missing-quality spatial results are not accepted")
        qualities.append(quality)
        measurement = payload.get("measurement")
        if not isinstance(measurement, dict):
            raise ValueError("Every spatial result must include measurement metadata")
        current_session = measurement.get("spatial_session_id")
        position = measurement.get("spatial_position")
        if not isinstance(current_session, str) or not current_session:
            raise ValueError("Spatial session id is required")
        if position not in SPATIAL_POSITION_KEYS:
            raise ValueError(f"Invalid spatial position: {position}")
        if position in by_position:
            raise ValueError(f"Duplicate spatial position: {position}")
        date_time = measurement.get("date_time")
        if not isinstance(date_time, str) or not date_time:
            raise ValueError("Every spatial position must include a recording timestamp")
        try:
            recorded_at = datetime.fromisoformat(date_time)
        except ValueError as error:
            raise ValueError("Spatial recording timestamp must use ISO 8601") from error
        if session_id is None:
            session_id = current_session
        elif current_session != session_id:
            raise ValueError("All spatial results must have the same session id")
        current_invariants = {field: measurement.get(field) for field in INVARIANT_FIELDS}
        if current_invariants["measurement_mode"] != "full":
            raise ValueError("Spatial aggregation accepts only full-mode measurements")
        if current_invariants["input_device"] is None or current_invariants["output_device"] is None:
            raise ValueError("Spatial measurements require explicit input and output devices")
        if not current_invariants["volume_note"]:
            raise ValueError("Spatial measurements require a non-empty volume note")
        if invariants is None:
            invariants = current_invariants
        elif current_invariants != invariants:
            raise ValueError("EQ, devices, sample rate and volume must match across positions")
        raw_response = np.asarray(payload.get("raw_response_db"), dtype=np.float64)
        if raw_response.shape != (31,) or not np.all(np.isfinite(raw_response)):
            raise ValueError(f"Invalid raw response for position: {position}")
        by_position[str(position)] = raw_response
        position_timestamps[str(position)] = recorded_at
        smoothed.add(payload, str(position))

    missing = set(SPATIAL_POSITION_KEYS) - set(by_position)
    if missing:
        raise ValueError(f"Missing spatial positions: {', '.join(sorted(missing))}")
    chronological_order = tuple(
        position for position, _ in sorted(position_timestamps.items(), key=lambda item: item[1])
    )
    if chronological_order != SPATIAL_POSITION_KEYS:
        raise ValueError("Spatial positions were not recorded in the required order")
    ordered = np.vstack([by_position[position] for position in SPATIAL_POSITION_KEYS])
    raw_mean = np.mean(ordered, axis=0)
    selected_target = np.asarray(TARGET_DB if target_db is None else target_db, dtype=np.float64)
    if selected_target.shape != (31,):
        raise ValueError("Spatial target must contain 31 bands")
    alignment = align_response_to_target(raw_mean, selected_target)
    smoothed_frequencies, smoothed_raw_mean, smoothed_aligned_mean, smoothed_std = smoothed.aggregate(
        alignment.offset_db,
        len(SPATIAL_POSITION_KEYS),
    )
    return SpatialResult(
        session_id=session_id or "",
        raw_mean_db=raw_mean,
        aligned_mean_db=alignment.aligned_response_db,
        standard_deviation_db=np.std(ordered, axis=0, ddof=1),
        p10_db=np.percentile(ordered, 10, axis=0),
        p90_db=np.percentile(ordered, 90, axis=0),
        position_raw_db={position: by_position[position] for position in SPATIAL_POSITION_KEYS},
        alignment=alignment,
        invariants=invariants or {},
        target_name=target_name,
        target_db=selected_target,
        quality=aggregate_spatial_quality(qualities),
        smoothed_frequencies_hz=smoothed_frequencies,
        smoothed_raw_mean_db=smoothed_raw_mean,
        smoothed_aligned_mean_db=smoothed_aligned_mean,
        smoothed_standard_deviation_db=smoothed_std,
        source_measurement_ids=tuple(measurement_ids),
    )
