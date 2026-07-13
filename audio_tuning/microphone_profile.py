from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


MICROPHONE_SCHEMA_VERSION = "measurement_microphone_v1"


def _frequency_range(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Microphone field must contain [low, high]: {field}")
    low, high = (float(item) for item in value)
    if not 1.0 <= low < high <= 100_000.0:
        raise ValueError(f"Invalid microphone frequency range: {field}")
    return low, high


@dataclass(frozen=True)
class MicrophoneProfile:
    source_path: Path | None
    profile_id: str
    name: str
    connection: str
    transducer_type: str
    polar_pattern: str
    published_frequency_range_hz: tuple[float, float]
    target_optimization_range_hz: tuple[float, float]
    orientation: dict[str, object]
    calibration: dict[str, object]
    specifications: dict[str, object]
    sources: dict[str, object]

    @property
    def calibration_file(self) -> Path | None:
        raw = self.calibration.get("file")
        if not raw:
            return None
        path = Path(str(raw))
        if path.is_absolute() or self.source_path is None:
            return path
        return self.source_path.parent / path

    @property
    def calibrated(self) -> bool:
        path = self.calibration_file
        return bool(self.calibration.get("individual")) and path is not None and path.exists()

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": MICROPHONE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_path": str(self.source_path) if self.source_path else None,
            "name": self.name,
            "connection": self.connection,
            "transducer_type": self.transducer_type,
            "polar_pattern": self.polar_pattern,
            "published_frequency_range_hz": list(self.published_frequency_range_hz),
            "target_optimization_range_hz": list(self.target_optimization_range_hz),
            "orientation": self.orientation,
            "calibration": {**self.calibration, "active": self.calibrated},
            "specifications": self.specifications,
        }


def load_microphone_profile(path: Path) -> MicrophoneProfile:
    resolved = path.resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != MICROPHONE_SCHEMA_VERSION:
        raise ValueError(f"Expected microphone profile schema {MICROPHONE_SCHEMA_VERSION}")
    profile_id = str(data.get("microphone_id", ""))
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", profile_id) is None:
        raise ValueError("Microphone id must contain only lowercase letters, digits, _ or -")
    published = _frequency_range(data.get("published_frequency_range_hz"), "published_frequency_range_hz")
    optimization = _frequency_range(
        data.get("target_optimization_range_hz", list(published)),
        "target_optimization_range_hz",
    )
    if optimization[0] < published[0] or optimization[1] > published[1]:
        raise ValueError("Microphone optimization range must stay inside the published range")
    calibration = data.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Microphone calibration must be an object")
    orientation = data.get("orientation")
    if not isinstance(orientation, dict):
        raise ValueError("Microphone orientation must be an object")
    specifications = data.get("specifications", {})
    sources = data.get("sources", {})
    if not isinstance(specifications, dict) or not isinstance(sources, dict):
        raise ValueError("Microphone specifications and sources must be objects")
    return MicrophoneProfile(
        source_path=resolved,
        profile_id=profile_id,
        name=str(data["name"]),
        connection=str(data.get("connection", "unknown")),
        transducer_type=str(data.get("transducer_type", "unknown")),
        polar_pattern=str(data.get("polar_pattern", "unknown")),
        published_frequency_range_hz=published,
        target_optimization_range_hz=optimization,
        orientation={str(key): value for key, value in orientation.items()},
        calibration={str(key): value for key, value in calibration.items()},
        specifications={str(key): value for key, value in specifications.items()},
        sources={str(key): value for key, value in sources.items()},
    )


def microphone_profile_from_legacy(
    data: dict[str, object],
    source_path: Path,
    device_id: str,
) -> MicrophoneProfile:
    calibration = {
        "file": data.get("calibration_file"),
        "individual": bool(data.get("calibration_file")),
        "absolute_spl_capable": False,
    }
    return MicrophoneProfile(
        source_path=source_path,
        profile_id=f"legacy_{device_id}_microphone",
        name=str(data.get("name", "legacy microphone")),
        connection=str(data.get("connection", "unknown")),
        transducer_type=str(data.get("transducer_type", "unknown")),
        polar_pattern=str(data.get("polar_pattern", "unknown")),
        published_frequency_range_hz=(20.0, 20_000.0),
        target_optimization_range_hz=(20.0, 20_000.0),
        orientation={"legacy_note": data.get("orientation", "not specified")},
        calibration=calibration,
        specifications={"legacy_inline_profile": True},
        sources={},
    )
