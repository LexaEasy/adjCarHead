from __future__ import annotations

from datetime import datetime

import numpy as np

from config import ANALYSIS_FREQUENCIES_HZ
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from spatial_positions import SPATIAL_POSITION_KEYS, SPATIAL_SCHEMA_VERSION
from targets import target_curve_db
from result_validation import ess_validation_manifest


def make_ess_result(
    profile: object,
    raw: np.ndarray,
    *,
    mode: str = "quick",
    position: str = "center_between_ears",
    session_id: str = "session",
    date_time: str | None = None,
    volume: str = "reference",
    eq: dict[str, float] | None = None,
    channel: str = "stereo",
    routing_verified: bool = True,
    measurement_id: str | None = None,
) -> dict[str, object]:
    smoothed_frequencies = np.geomspace(40.0, 18_000.0, 64)
    smoothed_response = np.interp(
        np.log(smoothed_frequencies),
        np.log(np.asarray(ANALYSIS_FREQUENCIES_HZ)),
        raw,
    )
    identifier = measurement_id or f"recording:sha256:{session_id}-{position}-{channel}"
    ess_parameters = {"duration_s": 5.0, "start_hz": 40.0, "end_hz": 18_000.0}
    return {
        "method": "ess",
        "analysis_method": "ess_deconvolution",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "clock_drift_compensated": True,
        "timing_markers_valid": True,
        "measurement_id": identifier,
        "source_signal_id": "ess:sha256:source",
        "inverse_filter_id": "ess_inverse:sha256:inverse",
        "active_ess_start_sample": 4_800,
        "active_ess_end_sample": 244_800,
        "active_ess_duration_s": 5.0,
        "active_ess_complete": True,
        "dropout_analysis_scope": "active_ess_only",
        "ess_parameters": ess_parameters,
        "raw_response_db": raw.tolist(),
        "quality": {
            "accepted": True,
            "clipped": False,
            "hard_failures": [],
            "warnings": [],
            "peak_dbfs": -12.0,
            "band_confidence_weight": [1.0] * 31,
            "early_h2_h3_ratio_percent": [1.0] * 31,
        },
        "smoothed_response": {
            "frequencies_hz": smoothed_frequencies.tolist(),
            "raw_response_db": smoothed_response.tolist(),
        },
        "measurement": {
            "profile_name": f"preset_{position}",
            "date_time": date_time or datetime(2026, 7, 13, 16, 0).isoformat(),
            "device_profile_id": profile.device_id,
            "device_profile_schema": profile.schema_version,
            "microphone_profile_id": profile.microphone_profile.profile_id,
            "input_device": 1,
            "output_device": 2,
            "sample_rate": 48_000,
            "volume_note": volume,
            "processing_settings": profile.processing,
            "measurement_mode": mode,
            "session_purpose": "baseline",
            "eq_settings": eq if eq is not None else profile.default_eq(),
            "mic_position_note": position,
            "mic_position_id": position,
            "spatial_session_id": session_id if mode == "full" else None,
            "spatial_position": position if mode == "full" else None,
            "channel_selection": channel,
            "channel_routing_verified": routing_verified,
            "system_profile_hash": "system-profile-hash",
            "microphone_profile_hash": "microphone-profile-hash",
            "source_signal_id": "ess:sha256:source",
            "inverse_filter_id": "ess_inverse:sha256:inverse",
            "ess_parameters": ess_parameters,
            "clock_correction": "required",
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        },
    }


def make_spatial_result(
    profile: object,
    curve: np.ndarray,
    *,
    purpose: str,
    channel: str = "stereo",
    session_id: str | None = None,
    routing_verified: bool = True,
) -> dict[str, object]:
    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    invariants = make_ess_result(
        profile,
        curve,
        mode="full",
        position="left_ear",
        channel=channel,
        routing_verified=routing_verified,
    )["measurement"]
    invariants["session_purpose"] = purpose
    source_results = [
        make_ess_result(
            profile,
            curve,
            mode="full",
            position=position,
            session_id=session_id or f"{purpose}-{channel}",
            date_time=f"2026-07-13T16:{index:02d}:00",
            channel=channel,
            routing_verified=routing_verified,
            measurement_id=f"{purpose}-{channel}-{index}",
        )
        for index, position in enumerate(SPATIAL_POSITION_KEYS, start=1)
    ]
    for source in source_results:
        source["measurement"]["session_purpose"] = purpose
    return {
        "spatial_schema_version": SPATIAL_SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "session_id": session_id or f"{purpose}-{channel}",
        "session_purpose": purpose,
        "tuning_state": f"full_{purpose}_measured",
        "position_order": SPATIAL_POSITION_KEYS,
        "raw_mean_db": curve.tolist(),
        "aligned_mean_db": curve.tolist(),
        "standard_deviation_db": [0.5] * 31,
        "target_db": target.tolist(),
        "measurement_invariants": invariants,
        "quality": {
            "accepted": True,
            "hard_failures": [],
            "band_confidence_weight": [1.0] * 31,
            "early_h2_h3_ratio_percent": [1.0] * 31,
            "worst_peak_dbfs": -12.0,
        },
        "spatial_aggregate_complete": True,
        "source_measurement_ids": [f"{purpose}-{channel}-{index}" for index in range(1, 7)],
        "source_validation_manifests": [ess_validation_manifest(value) for value in source_results],
    }
