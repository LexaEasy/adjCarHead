from __future__ import annotations

import numpy as np


ANALYSIS_SCHEMA_VERSION = "iso266_third_octave_v2"
REFERENCE_FREQUENCY_HZ = 1000.0

NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ = (
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
)

# ISO 266 base-ten series around the 1000 Hz reference, n = -17 ... 13.
EXACT_THIRD_OCTAVE_FREQUENCIES_HZ = tuple(
    REFERENCE_FREQUENCY_HZ * 10.0 ** (index / 10.0) for index in range(-17, 14)
)


def third_octave_edges(center_hz: float) -> tuple[float, float]:
    half_step = 10.0**0.05
    return center_hz / half_step, center_hz * half_step


def band_power_levels(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    *,
    reducer: str,
) -> np.ndarray:
    levels = []
    for center_hz in EXACT_THIRD_OCTAVE_FREQUENCIES_HZ:
        lower_hz, upper_hz = third_octave_edges(center_hz)
        mask = (frequencies_hz >= lower_hz) & (frequencies_hz < upper_hz)
        if reducer == "sum":
            value = float(np.sum(power[mask]))
        elif reducer == "mean":
            value = float(np.mean(power[mask])) if np.any(mask) else 0.0
        else:
            raise ValueError(f"Unsupported band reducer: {reducer}")
        levels.append(value)
    return np.asarray(levels, dtype=np.float64)


def smooth_fractional_octave(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    *,
    fraction: int = 6,
    point_count: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (frequencies_hz > 0.0) & np.isfinite(power) & (power >= 0.0)
    frequencies = frequencies_hz[valid]
    spectrum_power = power[valid]
    centers = np.geomspace(20.0, 20_000.0, point_count)
    half_width = 2.0 ** (1.0 / (2.0 * fraction))
    smoothed_power = np.empty_like(centers)
    for index, center_hz in enumerate(centers):
        mask = (frequencies >= center_hz / half_width) & (
            frequencies < center_hz * half_width
        )
        smoothed_power[index] = np.mean(spectrum_power[mask]) if np.any(mask) else np.nan
    levels_db = 10.0 * np.log10(np.maximum(smoothed_power, 1e-20))
    return centers, levels_db
