from __future__ import annotations

import numpy as np


def aggregate_spatial_quality(qualities: list[dict[str, object]]) -> dict[str, object]:
    confidence_rows = [quality.get("band_confidence_weight") for quality in qualities]
    snr_rows = [quality.get("band_snr_db") for quality in qualities]
    valid_confidence = all(isinstance(row, list) and len(row) == 31 for row in confidence_rows)
    valid_snr = all(isinstance(row, list) and len(row) == 31 for row in snr_rows)
    confidence = np.min(np.asarray(confidence_rows), axis=0).tolist() if valid_confidence else None
    snr = np.min(np.asarray(snr_rows), axis=0).tolist() if valid_snr else None
    distortion_rows = [quality.get("early_h2_h3_ratio_percent") for quality in qualities]
    valid_distortion = all(isinstance(row, list) and len(row) == 31 for row in distortion_rows)
    distortion = None
    if valid_distortion:
        distortion = []
        for index in range(31):
            values = [float(row[index]) for row in distortion_rows if row[index] is not None]
            distortion.append(max(values) if values else None)
    failures = sorted(
        {str(item) for quality in qualities for item in quality.get("hard_failures", [])}
    )
    warnings = sorted(
        {str(item) for quality in qualities for item in quality.get("warnings", [])}
    )
    return {
        "accepted": not failures,
        "hard_failures": failures,
        "warnings": warnings,
        "band_confidence_weight": confidence,
        "band_snr_db": snr,
        "early_h2_h3_ratio_percent": distortion,
        "distortion_metric_status": "experimental",
        "worst_peak_dbfs": max(float(quality.get("peak_dbfs", -999.0)) for quality in qualities),
        "worst_dropout_ratio": max(float(quality.get("dropout_ratio", 0.0)) for quality in qualities),
        "position_count": len(qualities),
    }
