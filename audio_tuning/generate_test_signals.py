from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp

from config import (
    DEFAULT_SIGNAL_LEVEL,
    FADE_SECONDS,
    ANALYSIS_FREQUENCIES_HZ,
    SAMPLE_RATE,
    TIMING_MARKER_DURATION_SECONDS,
    TIMING_MARKER_END_HZ,
    TIMING_MARKER_FADE_SECONDS,
    TIMING_MARKER_LEVEL,
    TIMING_MARKER_START_HZ,
)
from dsp import apply_fade
from ess import build_inverse_filter, generate_ess
from frequency_bands import (
    ANALYSIS_SCHEMA_VERSION,
    EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
    NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ,
)
from timing import generate_timing_marker


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), sample_rate, subtype="FLOAT")


def log_sweep(duration_s: float, sample_rate: int, start_hz: float, end_hz: float) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = chirp(t, f0=start_hz, f1=end_hz, t1=duration_s, method="logarithmic")
    return apply_fade(signal * DEFAULT_SIGNAL_LEVEL, sample_rate, FADE_SECONDS)


def pink_noise(duration_s: float, sample_rate: int, start_hz: float, end_hz: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = int(sample_rate * duration_s)
    white = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[(freqs < start_hz) | (freqs > end_hz)] = 0.0
    signal = np.fft.irfft(spectrum * scale, n=n)
    signal = signal / max(np.max(np.abs(signal)), 1e-12)
    return apply_fade(signal * DEFAULT_SIGNAL_LEVEL, sample_rate, FADE_SECONDS)


def stepped_sine(step_s: float, sample_rate: int, start_hz: float, end_hz: float) -> np.ndarray:
    parts = []
    for freq in ANALYSIS_FREQUENCIES_HZ:
        if not start_hz <= freq <= end_hz:
            continue
        t = np.linspace(0.0, step_s, int(sample_rate * step_s), endpoint=False)
        tone = np.sin(2.0 * np.pi * freq * t) * DEFAULT_SIGNAL_LEVEL
        parts.append(apply_fade(tone, sample_rate, 0.02))
    return np.concatenate(parts)


def calibration_tone(duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = np.sin(2.0 * np.pi * 1000.0 * t) * DEFAULT_SIGNAL_LEVEL
    return apply_fade(tone, sample_rate, FADE_SECONDS)


def generate_all(
    out: Path,
    sweep_duration: float,
    noise_duration: float,
    start_hz: float = 40.0,
    end_hz: float = 18_000.0,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    ess_sweep = generate_ess(
        sweep_duration,
        SAMPLE_RATE,
        start_hz,
        end_hz,
        DEFAULT_SIGNAL_LEVEL,
        FADE_SECONDS,
    )
    ess_inverse = build_inverse_filter(ess_sweep, SAMPLE_RATE, start_hz, end_hz)
    timing_marker = generate_timing_marker(
        TIMING_MARKER_DURATION_SECONDS,
        SAMPLE_RATE,
        TIMING_MARKER_START_HZ,
        TIMING_MARKER_END_HZ,
        TIMING_MARKER_LEVEL,
        TIMING_MARKER_FADE_SECONDS,
    )
    files = [
        (out / "ess_sweep.wav", ess_sweep),
        (out / "ess_inverse.wav", ess_inverse),
        (out / "timing_marker.wav", timing_marker),
        (out / "log_sweep.wav", log_sweep(sweep_duration, SAMPLE_RATE, start_hz, end_hz)),
        (out / "pink_noise.wav", pink_noise(noise_duration, SAMPLE_RATE, start_hz, end_hz)),
        (out / "stepped_sine.wav", stepped_sine(1.5, SAMPLE_RATE, start_hz, end_hz)),
        (out / "calibration_1000hz.wav", calibration_tone(5.0, SAMPLE_RATE)),
    ]
    for path, samples in files:
        write_wav(path, samples, SAMPLE_RATE)
    metadata_path = out / "ess_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "method": "exponential_sine_sweep",
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                "sample_rate": SAMPLE_RATE,
                "duration_s": sweep_duration,
                "start_hz": start_hz,
                "end_hz": end_hz,
                "level": DEFAULT_SIGNAL_LEVEL,
                "fade_s": FADE_SECONDS,
                "sweep_file": "ess_sweep.wav",
                "inverse_filter_file": "ess_inverse.wav",
                "timing_marker_file": "timing_marker.wav",
                "nominal_third_octave_frequencies_hz": NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ,
                "exact_third_octave_centers_hz": EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [path for path, _ in files] + [metadata_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate automotive audio test signals.")
    parser.add_argument("--out", type=Path, default=Path("data/test_signals"))
    parser.add_argument("--sweep-duration", type=float, default=8.0)
    parser.add_argument("--noise-duration", type=float, default=8.0)
    parser.add_argument("--start-hz", type=float, default=40.0)
    parser.add_argument("--end-hz", type=float, default=18_000.0)
    args = parser.parse_args()

    written = generate_all(
        args.out,
        args.sweep_duration,
        args.noise_duration,
        args.start_hz,
        args.end_hz,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
