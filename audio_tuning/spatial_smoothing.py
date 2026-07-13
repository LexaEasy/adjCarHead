from __future__ import annotations

import numpy as np


class SmoothedSpatialAccumulator:
    def __init__(self) -> None:
        self.frequencies: np.ndarray | None = None
        self.rows: list[np.ndarray] = []
        self.available = True

    def add(self, payload: dict[str, object], position: str) -> None:
        smoothed = payload.get("smoothed_response")
        if not isinstance(smoothed, dict):
            self.available = False
            return
        frequencies = np.asarray(smoothed.get("frequencies_hz"), dtype=np.float64)
        response = np.asarray(smoothed.get("raw_response_db"), dtype=np.float64)
        if (
            frequencies.ndim != 1
            or response.shape != frequencies.shape
            or not np.all(np.isfinite(response))
        ):
            raise ValueError(f"Invalid smoothed response for position: {position}")
        if self.frequencies is None:
            self.frequencies = frequencies
        elif not np.allclose(frequencies, self.frequencies):
            raise ValueError("Smoothed frequency grids must match across positions")
        self.rows.append(response)

    def aggregate(
        self,
        offset_db: float,
        expected_count: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if not self.available or len(self.rows) != expected_count:
            return None, None, None, None
        stack = np.vstack(self.rows)
        raw_mean = np.mean(stack, axis=0)
        return (
            self.frequencies,
            raw_mean,
            raw_mean - offset_db,
            np.std(stack, axis=0, ddof=1),
        )
