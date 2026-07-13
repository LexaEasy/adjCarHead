from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len, rfft, rfftfreq

from dsp import EPS
from frequency_bands import EXACT_THIRD_OCTAVE_FREQUENCIES_HZ, band_power_levels


@dataclass(frozen=True)
class EarlyHarmonicEstimate:
    h2_percent: tuple[float | None, ...]
    h3_percent: tuple[float | None, ...]
    combined_percent: tuple[float | None, ...]
    diagnostics: dict[str, object]


def _window_bounds(
    length: int,
    center: int,
    sample_rate: int,
    duration_s: float = 0.12,
    pre_peak_s: float = 0.01,
) -> tuple[int, int]:
    duration = int(round(duration_s * sample_rate))
    pre = int(round(pre_peak_s * sample_rate))
    start = max(0, center - pre)
    return start, min(length, start + duration)


def _window(samples: np.ndarray, bounds: tuple[int, int], sample_rate: int) -> np.ndarray:
    start, end = bounds
    window = samples[start:end].copy()
    pre = min(int(round(0.01 * sample_rate)), len(window))
    if pre > 1:
        window[:pre] *= np.linspace(0.0, 1.0, pre, endpoint=False)
    tail = min(int(round(0.02 * sample_rate)), len(window) // 4)
    if tail > 1:
        window[-tail:] *= np.linspace(1.0, 0.0, tail)
    return window


def _band_ratio(
    harmonic_power: np.ndarray,
    linear_power: np.ndarray,
    order: int,
    end_hz: float,
    sample_rate: int,
) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for index, center_hz in enumerate(EXACT_THIRD_OCTAVE_FREQUENCIES_HZ):
        available = order * center_hz < 0.95 * sample_rate / 2.0 and center_hz <= end_hz
        if not available or linear_power[index] <= EPS:
            values.append(None)
        else:
            values.append(float(100.0 * np.sqrt(harmonic_power[index] / linear_power[index])))
    return tuple(values)


def estimate_early_h2_h3(
    deconvolved: np.ndarray,
    linear_peak: int,
    sample_rate: int,
    sweep_duration_s: float,
    start_hz: float,
    end_hz: float,
) -> EarlyHarmonicEstimate:
    log_ratio = np.log(end_hz / start_hz)
    centers = {
        order: linear_peak
        - int(round(sweep_duration_s * np.log(order) / log_ratio * sample_rate))
        for order in (2, 3)
    }
    bounds = {
        "linear": _window_bounds(len(deconvolved), linear_peak, sample_rate),
        "h2": _window_bounds(len(deconvolved), centers[2], sample_rate),
        "h3": _window_bounds(len(deconvolved), centers[3], sample_rate),
    }
    windows = {name: _window(deconvolved, value, sample_rate) for name, value in bounds.items()}
    fft_size = next_fast_len(max(65_536, *(len(value) for value in windows.values())))
    frequencies = rfftfreq(fft_size, 1.0 / sample_rate)
    powers = {
        name: band_power_levels(
            frequencies,
            np.square(np.abs(rfft(value, n=fft_size))),
            reducer="mean",
        )
        for name, value in windows.items()
    }
    h2 = _band_ratio(powers["h2"], powers["linear"], 2, end_hz, sample_rate)
    h3 = _band_ratio(powers["h3"], powers["linear"], 3, end_hz, sample_rate)
    combined = tuple(
        None if left is None and right is None else float(np.sqrt((left or 0.0) ** 2 + (right or 0.0) ** 2))
        for left, right in zip(h2, h3)
    )
    noise_start = min(len(deconvolved), linear_peak + int(round(0.2 * sample_rate)))
    noise_end = min(len(deconvolved), noise_start + int(round(0.12 * sample_rate)))
    noise = _window(deconvolved, (noise_start, noise_end), sample_rate)
    noise_power = band_power_levels(
        frequencies,
        np.square(np.abs(rfft(noise, n=fft_size))),
        reducer="mean",
    )
    snr = {
        name: (10.0 * np.log10(np.maximum(powers[name], EPS) / np.maximum(noise_power, EPS))).tolist()
        for name in ("h2", "h3")
    }
    return EarlyHarmonicEstimate(
        h2_percent=h2,
        h3_percent=h3,
        combined_percent=combined,
        diagnostics={
            "expected_peak_samples": {"linear": linear_peak, "h2": centers[2], "h3": centers[3]},
            "window_bounds_samples": {name: list(value) for name, value in bounds.items()},
            "window_duration_s": 0.12,
            "harmonic_snr_db": snr,
            "noise_reference_bounds_samples": [noise_start, noise_end],
        },
    )
