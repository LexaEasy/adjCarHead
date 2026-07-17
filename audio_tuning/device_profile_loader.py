from __future__ import annotations

import json
from pathlib import Path
import re

from device_profile import (
    DSP_MODEL_STATUSES,
    LEGACY_PROFILE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    DeviceProfile,
    DspControl,
)
from microphone_profile import load_microphone_profile, microphone_profile_from_legacy
from targets import target_names


def _required_object(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Device profile field must be an object: {key}")
    return value


def _load_controls(data: object, required: bool) -> tuple[DspControl, ...]:
    if data is None and not required:
        return ()
    if not isinstance(data, list) or (required and not data):
        raise ValueError("Device profile requires DSP controls")
    controls = tuple(
        DspControl(
            control_id=str(item["id"]),
            label=str(item.get("label", item["id"])),
            center_hz=float(item["center_hz"]),
            q=float(item["q"]) if item.get("q") is not None else None,
            minimum_db=float(item.get("minimum_db", -9.0)),
            maximum_db=float(item.get("maximum_db", 9.0)),
            step_db=float(item.get("step_db", 0.1)),
        )
        for item in data
        if isinstance(item, dict)
    )
    if required and not controls:
        raise ValueError("Device profile requires valid DSP controls")
    if len({control.control_id for control in controls}) != len(controls):
        raise ValueError("DSP control ids must be unique")
    if any(
        control.step_db <= 0
        or control.center_hz <= 0
        or control.minimum_db > control.maximum_db
        or (control.q is not None and control.q <= 0)
        for control in controls
    ):
        raise ValueError("DSP centers, Q, limits and steps must be valid")
    return controls


def load_device_profile_data(path: Path) -> DeviceProfile:
    resolved = path.resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Device profile must be a JSON object")
    schema_version = str(data.get("schema_version", ""))
    if schema_version not in {LEGACY_PROFILE_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION}:
        raise ValueError(f"Expected device profile schema {PROFILE_SCHEMA_VERSION} or legacy v1")
    device_id = str(data.get("device_id", ""))
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", device_id) is None:
        raise ValueError("Device id must contain only lowercase letters, digits, _ or -")
    target_name = str(data.get("target", ""))
    if target_name not in target_names():
        raise ValueError(f"Unknown target profile: {target_name}")
    sweep = _required_object(data, "sweep")
    durations = _required_object(sweep, "duration_s")
    start_hz, end_hz = float(sweep["start_hz"]), float(sweep["end_hz"])
    quick_duration, full_duration = float(durations["quick"]), float(durations["full"])
    if not 10.0 <= start_hz < end_hz <= 22_000.0:
        raise ValueError("Sweep range must be within 10-22000 Hz")
    if quick_duration <= 0 or full_duration <= 0 or quick_duration > full_duration:
        raise ValueError("Sweep durations must satisfy 0 < quick <= full")

    if schema_version == LEGACY_PROFILE_SCHEMA_VERSION:
        microphone = microphone_profile_from_legacy(
            _required_object(data, "microphone"), resolved, device_id
        )
        controls = _load_controls(data.get("dsp_controls"), required=True)
        control_model: dict[str, object] = {
            "status": "specified",
            "band_count": len(controls),
            "center_frequencies_hz": [control.center_hz for control in controls],
            "legacy_v1": True,
        }
    else:
        reference = data.get("microphone_profile")
        if not isinstance(reference, str) or not reference:
            raise ValueError("System profile v2 requires microphone_profile")
        microphone_path = Path(reference)
        if not microphone_path.is_absolute():
            microphone_path = resolved.parent / microphone_path
        microphone = load_microphone_profile(microphone_path)
        control_model = _required_object(data, "dsp_control_model")
        status = str(control_model.get("status", ""))
        if status not in DSP_MODEL_STATUSES:
            raise ValueError(f"Unknown DSP control model status: {status}")
        controls = _load_controls(data.get("dsp_controls"), required=status != "uncharacterized")
        band_count = int(control_model.get("band_count", len(controls)))
        if band_count <= 0 or len(controls) > band_count:
            raise ValueError("Invalid DSP band count")
        if status != "uncharacterized" and len(controls) != band_count:
            raise ValueError("Specified DSP model must define every control")

    artifacts = data.get("validation_artifacts", {})
    limits = data.get("quality_limits", {})
    completion_limits = data.get("completion_limits", {})
    return DeviceProfile(
        source_path=resolved,
        schema_version=schema_version,
        device_id=device_id,
        name=str(data["name"]),
        has_subwoofer=bool(data.get("has_subwoofer", False)),
        target_name=target_name,
        sweep_start_hz=start_hz,
        sweep_end_hz=end_hz,
        quick_duration_s=quick_duration,
        full_duration_s=full_duration,
        volume=_required_object(data, "volume"),
        processing=_required_object(data, "processing"),
        microphone_profile=microphone,
        validation={str(key): bool(value) for key, value in _required_object(data, "validation").items()},
        validation_artifacts={str(key): value for key, value in artifacts.items()}
        if isinstance(artifacts, dict)
        else {},
        quality_limits={str(key): float(value) for key, value in limits.items()}
        if isinstance(limits, dict)
        else {},
        completion_limits={
            str(key): float(value) for key, value in completion_limits.items()
        }
        if isinstance(completion_limits, dict)
        else {},
        input_signal_path=dict(data.get("input_signal_path", {}))
        if isinstance(data.get("input_signal_path", {}), dict)
        else {},
        speaker_topology=dict(data.get("speaker_topology", {}))
        if isinstance(data.get("speaker_topology", {}), dict)
        else {},
        measurement_reference=dict(data.get("measurement_reference", {}))
        if isinstance(data.get("measurement_reference", {}), dict)
        else {},
        crossover_policy=dict(data.get("crossover_policy", {}))
        if isinstance(data.get("crossover_policy", {}), dict)
        else {},
        delays=_required_object(data, "delays"),
        head_unit=dict(data.get("head_unit", {})) if isinstance(data.get("head_unit", {}), dict) else {},
        dsp_control_model={str(key): value for key, value in control_model.items()},
        dsp_controls=controls,
    )
