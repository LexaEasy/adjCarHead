from __future__ import annotations

import json

import numpy as np

from device_profile import DeviceProfile


def band_values(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 31:
        return None
    return np.asarray([np.nan if item is None else float(item) for item in value])


def artifact_values(
    profile: DeviceProfile,
    artifact_key: str,
    field: str,
) -> np.ndarray | None:
    path = profile.validation_artifact_path(artifact_key)
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return band_values(data.get(field)) if isinstance(data, dict) else None


def artifact_scalar(
    profile: DeviceProfile,
    artifact_key: str,
    field: str,
) -> float | None:
    path = profile.validation_artifact_path(artifact_key)
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get(field) if isinstance(data, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def quality_constraint_masks(
    profile: DeviceProfile,
    quality: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distortion_ok = np.ones(31, dtype=bool)
    repeatability_ok = np.ones(31, dtype=bool)
    level_ok = np.ones(31, dtype=bool)
    distortion = band_values(quality.get("early_h2_h3_ratio_percent")) if quality else None
    if distortion is not None:
        limit = profile.quality_limits.get("experimental_early_h2_h3_hard_limit_percent", 50.0)
        valid = np.isfinite(distortion)
        distortion_ok[valid] = distortion[valid] <= limit
    repeatability = artifact_values(
        profile,
        "repeatability",
        "band_standard_deviation_db",
    )
    if repeatability is not None:
        limit = profile.quality_limits.get("max_repeatability_std_db", 1.0)
        repeatability_ok[np.isfinite(repeatability)] = repeatability[np.isfinite(repeatability)] <= limit
    level_shape = artifact_values(
        profile,
        "level_linearity",
        "band_shape_maximum_deviation_db",
    )
    if level_shape is not None:
        limit = profile.quality_limits.get("max_level_shape_deviation_db", 1.0)
        level_ok[np.isfinite(level_shape)] = level_shape[np.isfinite(level_shape)] <= limit
    return distortion_ok, repeatability_ok, level_ok
