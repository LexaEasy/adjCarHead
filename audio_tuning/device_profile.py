from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from microphone_profile import MicrophoneProfile


LEGACY_PROFILE_SCHEMA_VERSION = "car_audio_device_v1"
PROFILE_SCHEMA_VERSION = "car_audio_system_v2"
DSP_MODEL_STATUSES = {"uncharacterized", "specified", "characterized"}


@dataclass(frozen=True)
class DspControl:
    control_id: str
    label: str
    center_hz: float
    q: float | None
    minimum_db: float
    maximum_db: float
    step_db: float


@dataclass(frozen=True)
class DeviceProfile:
    source_path: Path
    schema_version: str
    device_id: str
    name: str
    has_subwoofer: bool
    target_name: str
    sweep_start_hz: float
    sweep_end_hz: float
    quick_duration_s: float
    full_duration_s: float
    volume: dict[str, object]
    processing: dict[str, object]
    microphone_profile: MicrophoneProfile
    validation: dict[str, bool]
    validation_artifacts: dict[str, object]
    quality_limits: dict[str, float]
    delays: dict[str, object]
    head_unit: dict[str, object]
    dsp_control_model: dict[str, object]
    dsp_controls: tuple[DspControl, ...]

    @property
    def microphone(self) -> dict[str, object]:
        return self.microphone_profile.metadata()

    @property
    def calibration_file(self) -> Path | None:
        return self.microphone_profile.calibration_file

    @property
    def calibrated(self) -> bool:
        return self.microphone_profile.calibrated

    @property
    def dsp_recommendation_eligible(self) -> bool:
        required_checks = (
            "microphone_processing_disabled",
            "repeatability_verified",
            "volume_linearity_verified",
            "dsp_controls_characterized",
        )
        return (
            self.dsp_control_model.get("status") == "characterized"
            and bool(self.dsp_controls)
            and self.dsp_matrix_file is not None
            and self.dsp_matrix_file.exists()
            and all(self.validation.get(check, False) for check in required_checks)
        )

    @property
    def volume_reference_ready(self) -> bool:
        return self.volume.get("status") != "pending_user_reference"

    @property
    def dsp_matrix_file(self) -> Path | None:
        raw = self.dsp_control_model.get("response_matrix_file")
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_absolute() else self.source_path.parent / path

    def validation_artifact_path(self, key: str) -> Path | None:
        raw = self.validation_artifacts.get(key)
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_absolute() else self.source_path.parent / path

    def dsp_recommendation_block_reason(self) -> str | None:
        if self.dsp_recommendation_eligible:
            return None
        reasons = []
        if self.dsp_control_model.get("status") != "characterized" or not self.dsp_controls:
            reasons.append("не описаны и не охарактеризованы все DSP-регуляторы")
        if self.dsp_matrix_file is None or not self.dsp_matrix_file.exists():
            reasons.append("нет измеренной матрицы воздействия DSP")
        labels = {
            "microphone_processing_disabled": "не подтверждено отключение обработки микрофона",
            "repeatability_verified": "не подтверждена повторяемость quick-замеров",
            "volume_linearity_verified": "не подтверждена линейность тракта по громкости",
            "dsp_controls_characterized": "не подтверждена характеризация DSP",
        }
        reasons.extend(label for key, label in labels.items() if not self.validation.get(key, False))
        return "Точная DSP-рекомендация заблокирована: " + "; ".join(reasons) + "."

    def volume_note(self) -> str:
        return json.dumps(self.volume, ensure_ascii=False, sort_keys=True)

    def default_eq(self) -> dict[str, float]:
        return {control.control_id: 0.0 for control in self.dsp_controls}

    def parse_eq(self, raw: str | None) -> dict[str, float]:
        settings = self.default_eq()
        if not raw:
            return settings
        if not self.dsp_controls:
            raise ValueError(
                f"DSP controls for {self.device_id} are not characterized; use an empty --eq value"
            )
        controls = {control.control_id: control for control in self.dsp_controls}
        for item in raw.split(","):
            key, value_text = item.split("=", maxsplit=1)
            key = key.strip()
            if key not in controls:
                raise ValueError(f"Unknown DSP control for {self.device_id}: {key}")
            value = float(value_text.strip())
            control = controls[key]
            if not control.minimum_db <= value <= control.maximum_db:
                raise ValueError(f"DSP value outside limits for {key}: {value}")
            steps = value / control.step_db
            if abs(steps - round(steps)) > 1e-6:
                raise ValueError(f"DSP value for {key} must use {control.step_db:g} dB steps")
            settings[key] = value
        return settings

    def equipment_metadata(self) -> dict[str, object]:
        return {
            "system_profile": {
                "schema_version": self.schema_version,
                "profile_id": self.device_id,
                "profile_path": str(self.source_path),
                "name": self.name,
                "has_subwoofer": self.has_subwoofer,
                "head_unit": self.head_unit,
                "volume": self.volume,
                "processing": self.processing,
                "validation": self.validation,
                "validation_artifacts": self.validation_artifacts,
                "quality_limits": self.quality_limits,
                "dsp_control_model": self.dsp_control_model,
            },
            "microphone_profile": self.microphone_profile.metadata(),
        }


def load_device_profile(path: Path) -> DeviceProfile:
    from device_profile_loader import load_device_profile_data

    return load_device_profile_data(path)
