from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import chirp, fftconvolve, resample_poly

from dsp import EPS, apply_fade, to_mono


@dataclass(frozen=True)
class TimingLayout:
    sample_rate: int
    total_samples: int
    marker_duration_samples: int
    start_marker_sample: int
    end_marker_sample: int
    ess_start_sample: int
    ess_end_sample: int
    post_roll_samples: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TimingLayout:
        return cls(**{field: int(value[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ClockCorrection:
    corrected_recording: np.ndarray
    clock_ratio: float
    clock_drift_ppm: float
    start_marker_recorded_sample: int
    end_marker_recorded_sample: int
    start_marker_score: float
    end_marker_score: float
    resample_up: int
    resample_down: int
    ess_pre_context_samples: int
    valid_start_sample: int
    valid_end_sample: int

    def diagnostics(self) -> dict[str, object]:
        return {
            "clock_ratio": self.clock_ratio,
            "clock_drift_ppm": self.clock_drift_ppm,
            "start_marker_recorded_sample": self.start_marker_recorded_sample,
            "end_marker_recorded_sample": self.end_marker_recorded_sample,
            "start_marker_score": self.start_marker_score,
            "end_marker_score": self.end_marker_score,
            "resample_up": self.resample_up,
            "resample_down": self.resample_down,
            "ess_pre_context_samples": self.ess_pre_context_samples,
            "valid_start_sample": self.valid_start_sample,
            "valid_end_sample": self.valid_end_sample,
        }


def generate_timing_marker(
    duration_s: float,
    sample_rate: int,
    start_hz: float,
    end_hz: float,
    level: float,
    fade_s: float,
) -> np.ndarray:
    if duration_s <= 0 or start_hz <= 0 or end_hz <= start_hz:
        raise ValueError("Invalid timing marker parameters")
    if end_hz >= sample_rate / 2:
        raise ValueError("Timing marker end frequency must be below Nyquist")
    time_s = np.arange(int(round(duration_s * sample_rate))) / sample_rate
    marker = chirp(time_s, f0=start_hz, f1=end_hz, t1=duration_s, method="logarithmic")
    return apply_fade(marker * level, sample_rate, fade_s)


def build_timed_playback(
    signal: np.ndarray,
    marker: np.ndarray,
    sample_rate: int,
    pre_roll_s: float,
    marker_guard_s: float,
    post_roll_s: float,
    final_roll_s: float,
) -> tuple[np.ndarray, TimingLayout]:
    audio = signal[:, np.newaxis] if signal.ndim == 1 else signal
    marker_mono = to_mono(marker)
    marker_audio = np.repeat(marker_mono[:, np.newaxis], audio.shape[1], axis=1)

    def silence(duration_s: float) -> np.ndarray:
        return np.zeros((int(round(duration_s * sample_rate)), audio.shape[1]), dtype=np.float64)

    pre = silence(pre_roll_s)
    guard = silence(marker_guard_s)
    tail = silence(post_roll_s)
    final = silence(final_roll_s)
    start_marker_sample = len(pre)
    ess_start_sample = start_marker_sample + len(marker_audio) + len(guard)
    ess_end_sample = ess_start_sample + len(audio)
    end_marker_sample = ess_end_sample + len(tail)
    playback = np.vstack((pre, marker_audio, guard, audio, tail, marker_audio, final))
    layout = TimingLayout(
        sample_rate=sample_rate,
        total_samples=len(playback),
        marker_duration_samples=len(marker_audio),
        start_marker_sample=start_marker_sample,
        end_marker_sample=end_marker_sample,
        ess_start_sample=ess_start_sample,
        ess_end_sample=ess_end_sample,
        post_roll_samples=len(tail),
    )
    return playback, layout


def _normalized_marker_correlation(recorded: np.ndarray, marker: np.ndarray) -> np.ndarray:
    rec = to_mono(recorded)
    ref = to_mono(marker)
    if len(rec) <= len(ref):
        raise ValueError("Recording is too short for timing marker detection")
    correlation = fftconvolve(rec, ref[::-1], mode="valid")
    cumulative_power = np.concatenate(([0.0], np.cumsum(np.square(rec))))
    local_power = cumulative_power[len(ref) :] - cumulative_power[: -len(ref)]
    denominator = np.sqrt(np.maximum(local_power * np.sum(np.square(ref)), EPS))
    return np.abs(correlation) / denominator


def _find_marker(
    scores: np.ndarray,
    expected_sample: int,
    sample_rate: int,
    search_before_s: float = 1.0,
    search_after_s: float = 2.0,
) -> tuple[int, float]:
    start = max(0, expected_sample - int(round(search_before_s * sample_rate)))
    end = min(len(scores), expected_sample + int(round(search_after_s * sample_rate)))
    if end <= start:
        raise ValueError("Timing marker search window is empty")
    relative = int(np.argmax(scores[start:end]))
    index = start + relative
    return index, float(scores[index])


def correct_clock_drift(
    recorded: np.ndarray,
    marker: np.ndarray,
    layout: TimingLayout,
    minimum_marker_score: float = 0.08,
    maximum_abs_drift_ppm: float = 5_000.0,
    ess_pre_context_s: float = 0.1,
) -> ClockCorrection:
    rec = to_mono(recorded)
    scores = _normalized_marker_correlation(rec, marker)
    start_index, start_score = _find_marker(scores, layout.start_marker_sample, layout.sample_rate)
    end_index, end_score = _find_marker(scores, layout.end_marker_sample, layout.sample_rate)
    if min(start_score, end_score) < minimum_marker_score:
        raise ValueError(
            f"Timing marker confidence is too low: start={start_score:.3f}, end={end_score:.3f}"
        )

    source_span = layout.end_marker_sample - layout.start_marker_sample
    recorded_span = end_index - start_index
    if recorded_span <= 0:
        raise ValueError("Timing markers are reversed or overlap")
    clock_ratio = recorded_span / source_span
    drift_ppm = (clock_ratio - 1.0) * 1_000_000.0
    if abs(drift_ppm) > maximum_abs_drift_ppm:
        raise ValueError(f"Implausible clock drift: {drift_ppm:.1f} ppm")

    correction_ratio = source_span / recorded_span
    fraction = Fraction(correction_ratio).limit_denominator(20_000)
    resampled = resample_poly(rec, fraction.numerator, fraction.denominator, padtype="line")
    corrected_start = int(round(start_index * fraction.numerator / fraction.denominator))
    shift = layout.start_marker_sample - corrected_start
    aligned = np.zeros(layout.total_samples, dtype=np.float64)
    source_start = max(0, -shift)
    target_start = max(0, shift)
    copy_count = min(len(resampled) - source_start, len(aligned) - target_start)
    if copy_count <= 0:
        raise ValueError("Clock-corrected recording does not overlap the measurement timeline")
    aligned[target_start : target_start + copy_count] = resampled[source_start : source_start + copy_count]

    pre_context = min(int(round(ess_pre_context_s * layout.sample_rate)), layout.ess_start_sample)
    segment_start = layout.ess_start_sample - pre_context
    segment_end = min(layout.total_samples, layout.ess_end_sample + layout.post_roll_samples)
    valid_start = max(target_start, segment_start) - segment_start
    valid_end = min(target_start + copy_count, segment_end) - segment_start
    return ClockCorrection(
        corrected_recording=aligned[segment_start:segment_end],
        clock_ratio=clock_ratio,
        clock_drift_ppm=drift_ppm,
        start_marker_recorded_sample=start_index,
        end_marker_recorded_sample=end_index,
        start_marker_score=start_score,
        end_marker_score=end_score,
        resample_up=fraction.numerator,
        resample_down=fraction.denominator,
        ess_pre_context_samples=pre_context,
        valid_start_sample=max(0, valid_start),
        valid_end_sample=max(0, valid_end),
    )
