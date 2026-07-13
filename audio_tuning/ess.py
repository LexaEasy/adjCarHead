from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len, rfft, rfftfreq
from scipy.signal import fftconvolve

from dsp import EPS, apply_fade, to_mono
from ess_distortion import estimate_early_h2_h3
from frequency_bands import band_power_levels, smooth_fractional_octave


@dataclass(frozen=True)
class EssResult:
    response_db: np.ndarray
    smoothed_frequencies_hz: np.ndarray
    smoothed_response_db: np.ndarray
    impulse_response: np.ndarray
    peak_offset_samples: int
    alignment_delay_s: float
    reference_peak_index: int
    recorded_peak_index: int
    early_h2_ratio_percent: tuple[float | None, ...] | None
    early_h3_ratio_percent: tuple[float | None, ...] | None
    early_h2_h3_ratio_percent: tuple[float | None, ...] | None
    distortion_diagnostics: dict[str, object] | None


def generate_ess(
    duration_s: float,
    sample_rate: int,
    start_hz: float,
    end_hz: float,
    level: float,
    fade_s: float,
) -> np.ndarray:
    if duration_s <= 0 or start_hz <= 0 or end_hz <= start_hz:
        raise ValueError("Invalid ESS parameters")
    if end_hz >= sample_rate / 2:
        raise ValueError("ESS end frequency must be below Nyquist")

    sample_count = int(round(duration_s * sample_rate))
    time_s = np.arange(sample_count, dtype=np.float64) / sample_rate
    log_ratio = np.log(end_hz / start_hz)
    phase = 2.0 * np.pi * start_hz * duration_s / log_ratio
    phase *= np.exp(time_s * log_ratio / duration_s) - 1.0
    return apply_fade(np.sin(phase) * level, sample_rate, fade_s)


def build_inverse_filter(
    sweep: np.ndarray,
    sample_rate: int,
    start_hz: float,
    end_hz: float,
) -> np.ndarray:
    mono = to_mono(sweep)
    duration_s = len(mono) / sample_rate
    decay_time_s = duration_s / np.log(end_hz / start_hz)
    time_s = np.arange(len(mono), dtype=np.float64) / sample_rate
    inverse = mono[::-1] * np.exp(-time_s / decay_time_s)

    reference = fftconvolve(mono, inverse, mode="full")
    peak = float(np.max(np.abs(reference)))
    if peak <= EPS:
        raise ValueError("Cannot normalize an empty ESS inverse filter")
    return inverse / peak


def _band_levels(freqs: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    power = np.square(magnitude)
    band_power = band_power_levels(freqs, power, reducer="mean")
    return 10.0 * np.log10(np.maximum(band_power, EPS))


def _gate_impulse_response(
    deconvolved: np.ndarray,
    peak_index: int,
    sample_rate: int,
    pre_peak_s: float,
    ir_duration_s: float,
) -> tuple[np.ndarray, int]:
    pre_samples = min(int(round(pre_peak_s * sample_rate)), peak_index)
    start = peak_index - pre_samples
    end = min(len(deconvolved), peak_index + int(round(ir_duration_s * sample_rate)))
    impulse = deconvolved[start:end].copy()

    if pre_samples > 1:
        impulse[:pre_samples] *= np.linspace(0.0, 1.0, pre_samples, endpoint=False)
    tail_samples = min(int(round(0.05 * sample_rate)), len(impulse) // 4)
    if tail_samples > 1:
        phase = np.linspace(0.0, np.pi, tail_samples)
        impulse[-tail_samples:] *= 0.5 * (1.0 + np.cos(phase))
    return impulse, pre_samples


def deconvolve_ess(
    source: np.ndarray,
    recorded: np.ndarray,
    inverse_filter: np.ndarray,
    sample_rate: int,
    pre_peak_s: float = 0.02,
    ir_duration_s: float = 1.0,
    sweep_start_hz: float | None = None,
    sweep_end_hz: float | None = None,
) -> EssResult:
    src = to_mono(source)
    rec = to_mono(recorded)
    inverse = to_mono(inverse_filter)
    if min(len(src), len(rec), len(inverse)) < 256:
        raise ValueError("ESS source, recording and inverse filter are required")

    reference = fftconvolve(src, inverse, mode="full")
    deconvolved = fftconvolve(rec, inverse, mode="full")
    reference_peak = int(np.argmax(np.abs(reference)))
    recorded_peak = int(np.argmax(np.abs(deconvolved)))
    impulse, peak_offset = _gate_impulse_response(
        deconvolved,
        recorded_peak,
        sample_rate,
        pre_peak_s,
        ir_duration_s,
    )

    fft_size = next_fast_len(max(65_536, len(impulse)))
    spectrum = np.abs(rfft(impulse, n=fft_size))
    freqs = rfftfreq(fft_size, 1.0 / sample_rate)
    smoothed_frequencies, smoothed_response = smooth_fractional_octave(
        freqs,
        np.square(spectrum),
        fraction=6,
    )
    distortion = None
    if sweep_start_hz is not None and sweep_end_hz is not None:
        distortion = estimate_early_h2_h3(
            deconvolved,
            recorded_peak,
            sample_rate,
            len(src) / sample_rate,
            sweep_start_hz,
            sweep_end_hz,
        )
    return EssResult(
        response_db=_band_levels(freqs, spectrum),
        smoothed_frequencies_hz=smoothed_frequencies,
        smoothed_response_db=smoothed_response,
        impulse_response=impulse,
        peak_offset_samples=peak_offset,
        alignment_delay_s=(recorded_peak - reference_peak) / sample_rate,
        reference_peak_index=reference_peak,
        recorded_peak_index=recorded_peak,
        early_h2_ratio_percent=distortion.h2_percent if distortion else None,
        early_h3_ratio_percent=distortion.h3_percent if distortion else None,
        early_h2_h3_ratio_percent=distortion.combined_percent if distortion else None,
        distortion_diagnostics=distortion.diagnostics if distortion else None,
    )
