from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import NOMINAL_FREQUENCIES_HZ


ALIGNMENT_LOWER_HZ = 315.0
ALIGNMENT_UPPER_HZ = 3150.0
ALIGNMENT_METHOD = "median_mad_315_3150_hz"


@dataclass(frozen=True)
class ResponseAlignment:
    aligned_response_db: np.ndarray
    offset_db: float
    mad_db: float
    robust_sigma_db: float
    band_count: int
    outlier_count: int

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": ALIGNMENT_METHOD,
            "range_hz": [ALIGNMENT_LOWER_HZ, ALIGNMENT_UPPER_HZ],
            "offset_db": self.offset_db,
            "mad_db": self.mad_db,
            "robust_sigma_db": self.robust_sigma_db,
            "band_count": self.band_count,
            "outlier_count": self.outlier_count,
            "absolute_spl_reliable": False,
        }


def align_response_to_target(
    raw_response_db: np.ndarray,
    target_db: np.ndarray,
) -> ResponseAlignment:
    raw = np.asarray(raw_response_db, dtype=np.float64)
    target = np.asarray(target_db, dtype=np.float64)
    frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ, dtype=np.float64)
    if raw.shape != target.shape or raw.shape != frequencies.shape:
        raise ValueError("Response, target and frequency grid must have matching shapes")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(target)):
        raise ValueError("Response alignment requires finite values")

    mask = (frequencies >= ALIGNMENT_LOWER_HZ) & (frequencies <= ALIGNMENT_UPPER_HZ)
    residual = raw[mask] - target[mask]
    initial_offset = float(np.median(residual))
    deviations = np.abs(residual - initial_offset)
    mad_db = float(np.median(deviations))
    robust_sigma_db = 1.4826 * mad_db
    outlier_threshold_db = max(1.0, 3.0 * robust_sigma_db)
    inliers = deviations <= outlier_threshold_db
    offset_db = float(np.median(residual[inliers]))
    return ResponseAlignment(
        aligned_response_db=raw - offset_db,
        offset_db=offset_db,
        mad_db=mad_db,
        robust_sigma_db=robust_sigma_db,
        band_count=int(np.sum(mask)),
        outlier_count=int(np.sum(~inliers)),
    )
