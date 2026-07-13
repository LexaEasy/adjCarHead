from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import NOMINAL_FREQUENCIES_HZ
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from spatial_positions import SPATIAL_SCHEMA_VERSION


def _load_channels(paths: list[Path]) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        invariants = payload.get("measurement_invariants") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("spatial_schema_version") != SPATIAL_SCHEMA_VERSION
            or payload.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION
            or payload.get("spatial_aggregate_complete") is not True
            or not isinstance(invariants, dict)
        ):
            raise ValueError(f"Invalid full channel result: {path}")
        channel = str(invariants.get("channel_selection"))
        if channel in results:
            raise ValueError(f"Duplicate full channel result: {channel}")
        results[channel] = payload
    if set(results) != {"left", "right", "stereo"}:
        raise ValueError("Channel comparison requires left, right and stereo full results")
    if len({str(payload.get("session_id")) for payload in results.values()}) != 3:
        raise ValueError("Channel results must have distinct session ids")
    return results


def write_channel_comparison(out: Path, paths: list[Path]) -> Path:
    results = _load_channels(paths)
    reference = results["left"]["measurement_invariants"]
    common_fields = (
        "eq_settings",
        "input_device",
        "output_device",
        "sample_rate",
        "volume_note",
        "device_profile_id",
        "microphone_profile_id",
        "processing_settings",
        "system_profile_hash",
        "microphone_profile_hash",
        "source_signal_id",
        "inverse_filter_id",
        "ess_parameters",
        "clock_correction",
        "analysis_schema_version",
    )
    for channel, payload in results.items():
        invariants = payload["measurement_invariants"]
        changed = [field for field in common_fields if invariants.get(field) != reference.get(field)]
        if changed:
            raise ValueError(f"Channel comparison invariants changed for {channel}: {', '.join(changed)}")
    raw = {key: np.asarray(value["raw_mean_db"], dtype=float) for key, value in results.items()}
    aligned = {key: np.asarray(value["aligned_mean_db"], dtype=float) for key, value in results.items()}
    spatial = {key: np.asarray(value["standard_deviation_db"], dtype=float) for key, value in results.items()}
    if any(value.shape != (31,) for rows in (raw, aligned, spatial) for value in rows.values()):
        raise ValueError("Channel comparison requires 31-band full results")
    interaction = aligned["stereo"] - 0.5 * (aligned["left"] + aligned["right"])
    routing_verified = all(
        bool(payload["measurement_invariants"].get("channel_routing_verified"))
        for payload in results.values()
    )
    distortion = {
        key: value.get("quality", {}).get("early_h2_h3_ratio_percent")
        for key, value in results.items()
    }
    table = pd.DataFrame(
        {
            "frequency_hz": NOMINAL_FREQUENCIES_HZ,
            "left_minus_right_level_db": raw["left"] - raw["right"],
            "left_minus_right_shape_db": aligned["left"] - aligned["right"],
            "left_spatial_std_db": spatial["left"],
            "right_spatial_std_db": spatial["right"],
            "stereo_spatial_std_db": spatial["stereo"],
            "stereo_interaction_db": interaction,
        }
    )
    result = {
        "comparison_type": "full_left_right_stereo",
        "source_results": {key: str(path.resolve()) for key, path in zip(("left", "right", "stereo"), paths)},
        "channel_routing_verified": routing_verified,
        "channel_diagnostics_allowed": routing_verified,
        "left_right_level_difference_db": (raw["left"] - raw["right"]).tolist(),
        "left_right_shape_difference_db": (aligned["left"] - aligned["right"]).tolist(),
        "confidence": {key: value.get("quality", {}).get("band_confidence_weight") for key, value in results.items()},
        "experimental_distortion": distortion,
        "spatial_standard_deviation_db": {key: value.tolist() for key, value in spatial.items()},
        "stereo_only_dip_frequencies_hz": np.asarray(NOMINAL_FREQUENCIES_HZ)[interaction < -3.0].tolist(),
        "stereo_only_peak_frequencies_hz": np.asarray(NOMINAL_FREQUENCIES_HZ)[interaction > 3.0].tolist(),
        "polarity_claim_allowed": False,
        "cross_run_delay_claim_allowed": False,
        "timing_note": "Absolute L/R delay between Bluetooth runs is not reliable.",
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "channel_comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(out / "channel_comparison.csv", index=False)
    with (out / "channel_comparison.md").open("w", encoding="utf-8") as report:
        report.write("# Сравнение полных L/R/stereo-сессий\n\n")
        report.write(f"Маршрутизация каналов подтверждена: **{routing_verified}**.\n\n")
        report.write("Полярность и абсолютная межканальная задержка по раздельным Bluetooth-прогонам не определяются.\n\n")
        report.write(table.to_markdown(index=False, floatfmt=".2f"))
    return path
