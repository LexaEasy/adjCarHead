from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ResponseErrorMetrics:
    mean_absolute_error_db: float
    median_absolute_error_db: float
    percentile_75_absolute_error_db: float
    maximum_absolute_error_db: float
    maximum_positive_deviation_db: float
    maximum_negative_deviation_db: float
    mean_signed_deviation_db: float
    minimum_response_db: float
    maximum_response_db: float
    points_within_3_db: int
    band_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calculate_response_error_metrics(
    response_db: np.ndarray,
    target_db: np.ndarray,
    mask: np.ndarray,
) -> ResponseErrorMetrics:
    response = np.asarray(response_db, dtype=np.float64)
    target = np.asarray(target_db, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if response.shape != target.shape or selected.shape != response.shape:
        raise ValueError("Response, target and mask must have the same shape")
    selected_response = response[selected]
    residual = selected_response - target[selected]
    finite = np.isfinite(residual) & np.isfinite(selected_response)
    residual = residual[finite]
    selected_response = selected_response[finite]
    if not len(residual):
        raise ValueError("At least one finite response band is required")
    absolute = np.abs(residual)
    return ResponseErrorMetrics(
        mean_absolute_error_db=float(np.mean(absolute)),
        median_absolute_error_db=float(np.median(absolute)),
        percentile_75_absolute_error_db=float(np.percentile(absolute, 75)),
        maximum_absolute_error_db=float(np.max(absolute)),
        maximum_positive_deviation_db=float(max(0.0, np.max(residual))),
        maximum_negative_deviation_db=float(min(0.0, np.min(residual))),
        mean_signed_deviation_db=float(np.mean(residual)),
        minimum_response_db=float(np.min(selected_response)),
        maximum_response_db=float(np.max(selected_response)),
        points_within_3_db=int(np.sum(absolute <= 3.0)),
        band_count=int(len(residual)),
    )
