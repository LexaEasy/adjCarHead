from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frequency_bands import (
    EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
    NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ,
    third_octave_edges,
)
from targets import TARGET_FREQUENCIES_HZ, TARGET_PROFILES_DB, target_curve_db as build_target_curve

SAMPLE_RATE = 48_000
DEFAULT_SIGNAL_LEVEL = 0.25
FADE_SECONDS = 0.1
ESS_START_HZ = third_octave_edges(EXACT_THIRD_OCTAVE_FREQUENCIES_HZ[0])[0]
ESS_END_HZ = third_octave_edges(EXACT_THIRD_OCTAVE_FREQUENCIES_HZ[-1])[1]
DEFAULT_PRE_ROLL_SECONDS = 0.5
DEFAULT_POST_ROLL_SECONDS = 1.0
TIMING_MARKER_DURATION_SECONDS = 0.12
TIMING_MARKER_START_HZ = 2_000.0
TIMING_MARKER_END_HZ = 8_000.0
TIMING_MARKER_LEVEL = 0.20
TIMING_MARKER_FADE_SECONDS = 0.005
TIMING_MARKER_GUARD_SECONDS = 0.4
TIMING_FINAL_ROLL_SECONDS = 0.5

TARGET_CONTROL_FREQUENCIES_HZ = TARGET_FREQUENCIES_HZ.tolist()
TARGET_CONTROL_DB = TARGET_PROFILES_DB["warm_driver"].tolist()

ANALYSIS_FREQUENCIES_HZ = list(EXACT_THIRD_OCTAVE_FREQUENCIES_HZ)
NOMINAL_FREQUENCIES_HZ = list(NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ)


def target_curve_db(frequencies_hz: list[float] | np.ndarray) -> np.ndarray:
    return build_target_curve("warm_driver", frequencies_hz)


TARGET_DB = target_curve_db(ANALYSIS_FREQUENCIES_HZ).tolist()

EQ_BANDS_HZ = [60, 230, 910, 3600, 14000]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def recordings(self) -> Path:
        return self.data / "recordings"

    @property
    def outputs(self) -> Path:
        return self.data / "outputs"

    @property
    def test_signals(self) -> Path:
        return self.data / "test_signals"


def default_paths() -> Paths:
    return Paths(Path(__file__).resolve().parent)
