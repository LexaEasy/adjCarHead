from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frequencies: list[float] = []
    corrections: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if len(row) < 2:
                continue
            try:
                frequency = float(row[0].strip())
                correction = float(row[1].strip())
            except ValueError:
                continue
            if frequency > 0:
                frequencies.append(frequency)
                corrections.append(correction)
    if len(frequencies) < 2:
        raise ValueError(f"Microphone calibration requires at least two rows: {path}")
    order = np.argsort(frequencies)
    return np.asarray(frequencies)[order], np.asarray(corrections)[order]


def calibration_correction_db(path: Path, frequencies_hz: np.ndarray) -> np.ndarray:
    source_frequencies, source_corrections = load_calibration(path)
    return np.interp(
        np.log10(frequencies_hz),
        np.log10(source_frequencies),
        source_corrections,
        left=source_corrections[0],
        right=source_corrections[-1],
    )
