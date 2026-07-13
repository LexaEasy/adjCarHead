from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import welch

from config import ANALYSIS_FREQUENCIES_HZ
from frequency_bands import band_power_levels

EPS = 1e-20


@dataclass(frozen=True)
class AudioQuality:
    duration_s: float
    peak: float
    rms_dbfs: float
    silence_ratio: float
    clipped: bool


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float64)
    return np.mean(samples, axis=1, dtype=np.float64)


def apply_fade(samples: np.ndarray, sample_rate: int, fade_s: float) -> np.ndarray:
    fade_len = min(int(sample_rate * fade_s), len(samples) // 2)
    if fade_len <= 0:
        return samples
    envelope = np.ones(len(samples), dtype=np.float64)
    fade = np.linspace(0.0, 1.0, fade_len)
    envelope[:fade_len] = fade
    envelope[-fade_len:] = fade[::-1]
    return samples * envelope


def quality_metrics(samples: np.ndarray, sample_rate: int) -> AudioQuality:
    mono = to_mono(samples)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    rms_dbfs = float(20.0 * np.log10(max(rms, EPS)))
    silence_ratio = float(np.mean(np.abs(mono) < 1e-4)) if len(mono) else 1.0
    return AudioQuality(
        duration_s=float(len(mono) / sample_rate),
        peak=peak,
        rms_dbfs=rms_dbfs,
        silence_ratio=silence_ratio,
        clipped=peak >= 0.999,
    )


def third_octave_levels(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = to_mono(samples)
    nperseg = min(65_536, len(mono))
    if nperseg < 256:
        raise ValueError("Audio file is too short for spectrum analysis")
    noverlap = min(32_768, nperseg // 2)
    freqs, power = welch(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )

    band_power = band_power_levels(freqs, power, reducer="sum")
    levels_db = 10.0 * np.log10(np.maximum(band_power, EPS))
    return levels_db


def psd_ratio_estimate(
    source: np.ndarray,
    recorded: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    src = to_mono(source)
    rec = to_mono(recorded)
    n = min(len(src), len(rec))
    if n < 256:
        raise ValueError("Source and recorded signals are too short")
    src = src[:n]
    rec = rec[:n]
    nperseg = min(65_536, n)
    noverlap = min(32_768, nperseg // 2)
    freqs, src_power = welch(
        src,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )
    _, rec_power = welch(
        rec,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )

    ratio = np.divide(rec_power, src_power, out=np.ones_like(rec_power), where=src_power > EPS)
    band_ratio = band_power_levels(freqs, ratio, reducer="mean")
    levels_db = 10.0 * np.log10(np.maximum(band_ratio, EPS))
    return levels_db


def zone_errors(response_db: np.ndarray, ideal_db: np.ndarray) -> dict[str, float]:
    freq = np.array(ANALYSIS_FREQUENCIES_HZ, dtype=np.float64)
    grid = np.geomspace(freq[0], freq[-1], 512)
    response_i = PchipInterpolator(np.log10(freq), response_db)(np.log10(grid))
    ideal_i = PchipInterpolator(np.log10(freq), ideal_db)(np.log10(grid))
    abs_dev = np.abs(response_i - ideal_i)
    zones = {
        "Bass": grid <= 300,
        "Mid": (grid > 300) & (grid <= 4000),
        "Treble": grid > 4000,
        "Full": np.ones_like(grid, dtype=bool),
    }
    return {name: float(np.mean(abs_dev[mask])) for name, mask in zones.items()}
