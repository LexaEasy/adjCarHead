from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


TARGET_FREQUENCIES_HZ = np.asarray(
    [20, 31.5, 50, 80, 120, 200, 250, 300, 500, 1000, 1500, 2000,
     2500, 3150, 4000, 6000, 8000, 10000, 12500, 16000, 20000],
    dtype=np.float64,
)

TARGET_PROFILES_DB = {
    "neutral_driver": np.asarray(
        [4, 5.5, 6, 5, 4, 2.5, 1.8, 1.3, 0.4, 0, -0.4, -0.8,
         -1.2, -1.6, -2.1, -2.8, -3.5, -4.2, -4.9, -5.8, -7],
        dtype=np.float64,
    ),
    "warm_driver": np.asarray(
        [5, 7, 8, 7, 5.5, 3, 2.17, 1.5, 0.5, 0, -0.5, -1.06,
         -1.5, -1.99, -2.5, -3.2, -4, -4.8, -5.5, -6.5, -8],
        dtype=np.float64,
    ),
}


def target_names() -> tuple[str, ...]:
    return tuple(TARGET_PROFILES_DB)


def target_curve_db(name: str, frequencies_hz: list[float] | np.ndarray) -> np.ndarray:
    if name not in TARGET_PROFILES_DB:
        raise ValueError(f"Unknown target profile: {name}")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if np.any(frequencies <= 0):
        raise ValueError("Target frequencies must be positive")
    interpolator = PchipInterpolator(
        np.log10(TARGET_FREQUENCIES_HZ),
        TARGET_PROFILES_DB[name],
    )
    return np.asarray(interpolator(np.log10(frequencies)), dtype=np.float64)
