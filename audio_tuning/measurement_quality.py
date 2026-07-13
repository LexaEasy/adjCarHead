from __future__ import annotations

import numpy as np

from dsp import EPS, quality_metrics, third_octave_levels, to_mono


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, EPS)))


def _dropout_ratio(samples: np.ndarray, sample_rate: int) -> float:
    mono = to_mono(samples)
    frame_length = max(1, int(round(0.1 * sample_rate)))
    frame_count = len(mono) // frame_length
    if frame_count < 4:
        return 0.0
    frames = mono[: frame_count * frame_length].reshape(frame_count, frame_length)
    levels = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), EPS))
    threshold = float(np.median(levels) - 20.0)
    return float(np.mean(levels < threshold))


def _noise_metrics(
    signal: np.ndarray,
    noise: np.ndarray | None,
    sample_rate: int,
) -> tuple[float | None, list[float] | None, list[float], list[str]]:
    if noise is None or len(noise) < 256:
        return None, None, [0.5] * 31, ["unknown"] * 31
    noise_mono = to_mono(noise)
    noise_rms = float(np.sqrt(np.mean(noise_mono**2)))
    signal_levels = third_octave_levels(signal, sample_rate)
    noise_levels = third_octave_levels(noise_mono, sample_rate)
    snr = signal_levels - noise_levels
    weights = np.where(snr >= 20.0, 1.0, np.where(snr >= 12.0, 0.5, 0.0))
    labels = np.where(snr >= 20.0, "high", np.where(snr >= 12.0, "medium", "low"))
    return _dbfs(noise_rms), snr.tolist(), weights.tolist(), labels.tolist()


def assess_measurement(
    recorded: np.ndarray,
    sample_rate: int,
    signal_segment: np.ndarray | None = None,
    noise_segment: np.ndarray | None = None,
) -> dict[str, object]:
    basic = quality_metrics(recorded, sample_rate)
    mono = to_mono(recorded)
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    peak_dbfs = _dbfs(basic.peak)
    crest_factor_db = peak_dbfs - basic.rms_dbfs
    analysis_signal = signal_segment if signal_segment is not None else recorded
    noise_floor, snr, weights, labels = _noise_metrics(
        analysis_signal,
        noise_segment,
        sample_rate,
    )
    dropout_ratio = _dropout_ratio(analysis_signal, sample_rate)
    failures = []
    warnings = []
    if basic.clipped:
        failures.append("clipping")
    if dropout_ratio > 0.05:
        failures.append("dropout")
    if basic.rms_dbfs < -55.0:
        warnings.append("very_low_level")
    if peak_dbfs > -3.0 and not basic.clipped:
        warnings.append("low_digital_headroom")
    if peak_dbfs < -30.0:
        warnings.append("low_peak_level")
    if snr is None:
        warnings.append("noise_floor_unavailable")
    elif sum(weight > 0 for weight in weights) < 12:
        warnings.append("low_band_snr")
    return {
        **basic.__dict__,
        "peak_dbfs": peak_dbfs,
        "crest_factor_db": crest_factor_db,
        "noise_floor_dbfs": noise_floor,
        "band_snr_db": snr,
        "band_confidence_weight": weights,
        "band_confidence": labels,
        "dropout_ratio": dropout_ratio,
        "hard_failures": failures,
        "warnings": warnings,
        "accepted": not failures,
        "absolute_spl_reliable": False,
        "recommended_peak_window_dbfs": [-12.0, -6.0],
        "analog_overload_separable": False,
    }
