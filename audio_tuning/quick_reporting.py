from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ANALYSIS_FREQUENCIES_HZ, NOMINAL_FREQUENCIES_HZ
from device_profile import DeviceProfile
from response_alignment import align_response_to_target
from result_validation import require_valid_ess_result
from scoring import score_response
from targets import target_curve_db


COMPARISON_FIELDS = (
    "device_profile_id",
    "device_profile_schema",
    "microphone_profile_id",
    "input_device",
    "output_device",
    "sample_rate",
    "volume_note",
    "processing_settings",
    "measurement_mode",
    "mic_position_id",
    "channel_selection",
    "channel_routing_verified",
)

RESULT_COMPARISON_FIELDS = (
    "analysis_schema_version",
    "analysis_method",
    "clock_drift_compensated",
    "source_signal_id",
    "inverse_filter_id",
    "ess_parameters",
)


def _load_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Quick comparison requires an ESS result: {path}")
    require_valid_ess_result(payload, expected_mode="quick")
    return payload


def _score(
    payload: dict[str, object],
    profile: DeviceProfile,
    comparison_mask: np.ndarray | None = None,
    comparison_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    raw = np.asarray(payload.get("raw_response_db"), dtype=np.float64)
    if raw.shape != (31,):
        raise ValueError("Quick ESS result must contain 31 response bands")
    aligned = align_response_to_target(raw, target).aligned_response_db
    quality = payload.get("quality")
    if not isinstance(quality, dict) or quality.get("accepted") is False:
        raise ValueError("Quick ESS result failed measurement quality gates")
    measurement = payload["measurement"]
    eq_settings = measurement.get("eq_settings") if isinstance(measurement, dict) else None
    return aligned, score_response(
        aligned,
        target,
        profile,
        quality,
        eq_settings=eq_settings if isinstance(eq_settings, dict) else None,
        comparison_mask=comparison_mask,
        comparison_weights=comparison_weights,
    )


def _validate_comparison_invariants(
    baseline: dict[str, object],
    current: dict[str, object],
) -> None:
    baseline_measurement = baseline.get("measurement")
    current_measurement = current.get("measurement")
    if not isinstance(baseline_measurement, dict) or not isinstance(current_measurement, dict):
        raise ValueError("Quick comparison requires measurement metadata in both ESS results")
    baseline_values = {field: baseline_measurement.get(field) for field in COMPARISON_FIELDS}
    current_values = {field: current_measurement.get(field) for field in COMPARISON_FIELDS}
    if any(value is None for value in baseline_values.values()):
        raise ValueError("Quick baseline has incomplete comparison metadata")
    if baseline_values != current_values:
        changed = [field for field in COMPARISON_FIELDS if baseline_values[field] != current_values[field]]
        raise ValueError(f"Quick comparison invariants changed: {', '.join(changed)}")
    changed_results = [
        field for field in RESULT_COMPARISON_FIELDS if baseline.get(field) != current.get(field)
    ]
    if changed_results:
        raise ValueError(f"Quick comparison result invariants changed: {', '.join(changed_results)}")


def _technical_verdict(
    baseline_score: object,
    current_score: object,
    frequency_coverage_regression: bool,
    minimum_improvement_db: float,
    maximum_zone_regression_db: float,
) -> tuple[str, float, dict[str, float]]:
    delta = baseline_score.total_cost - current_score.total_cost
    zone_delta = {
        name: float(current_score.zone_errors_db[name] - baseline_value)
        for name, baseline_value in baseline_score.zone_errors_db.items()
        if baseline_value is not None
        and current_score.zone_errors_db.get(name) is not None
        and not name.endswith("_diagnostic")
    }
    worst_zone_regression = max(zone_delta.values(), default=0.0)
    confidence_change = float(
        np.mean(current_score.confidence_weights) - np.mean(baseline_score.confidence_weights)
    )
    if frequency_coverage_regression:
        verdict = "candidate_rejected_frequency_coverage_regression"
    elif (
        delta > minimum_improvement_db
        and worst_zone_regression <= maximum_zone_regression_db
        and confidence_change >= -0.1
    ):
        verdict = "technically_better_candidate"
    elif delta < -minimum_improvement_db or worst_zone_regression > 1.0:
        verdict = "technically_worse"
    else:
        verdict = "technically_equivalent_or_inconclusive"
    return verdict, delta, zone_delta


def write_quick_outputs(
    out: Path,
    current_result_path: Path,
    profile: DeviceProfile,
    baseline_result_path: Path | None = None,
) -> Path:
    current_payload = _load_result(current_result_path)
    current_curve, current_score = _score(current_payload, profile)
    baseline_curve = None
    baseline_score = None
    lost_frequencies: list[float] = []
    verdict = "baseline_created"
    delta = None
    zone_delta = None
    if baseline_result_path is not None:
        baseline_payload = _load_result(baseline_result_path)
        _validate_comparison_invariants(baseline_payload, current_payload)
        baseline_curve, baseline_native_score = _score(baseline_payload, profile)
        comparison_mask = np.asarray(baseline_native_score.target_optimization_mask, dtype=bool)
        comparison_weights = np.asarray(baseline_native_score.confidence_weights, dtype=np.float64)
        current_curve, current_score = _score(
            current_payload, profile, comparison_mask, comparison_weights
        )
        lost = (
            comparison_mask
            & (comparison_weights > 0)
            & ~np.asarray(current_score.observed_frequency_mask, dtype=bool)
        )
        lost_frequencies = np.asarray(NOMINAL_FREQUENCIES_HZ, dtype=float)[lost].tolist()
        baseline_curve, baseline_score = _score(
            baseline_payload, profile, comparison_mask, comparison_weights
        )
        verdict, delta, zone_delta = _technical_verdict(
            baseline_score,
            current_score,
            bool(lost_frequencies),
            profile.completion_limits.get("minimum_measured_score_improvement_db", 0.3),
            profile.completion_limits.get("maximum_zone_regression_db", 0.5),
        )

    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    result = {
        "mode": "quick",
        "device_profile_id": profile.device_id,
        "device_profile_schema": profile.schema_version,
        "microphone_profile_id": profile.microphone_profile.profile_id,
        "equipment": profile.equipment_metadata(),
        "target_profile": profile.target_name,
        "current_ess_result": str(current_result_path.resolve()),
        "baseline_ess_result": str(baseline_result_path.resolve()) if baseline_result_path else None,
        "verdict": verdict,
        "verdict_scope": "technical_relative_candidate_requires_listening",
        "cost_improvement_vs_baseline": delta,
        "zone_error_delta_vs_baseline_db": zone_delta,
        "comparison_mask_source": "baseline",
        "lost_reliable_band_count": len(lost_frequencies),
        "lost_reliable_band_frequencies_hz": lost_frequencies,
        "frequency_coverage_regression": bool(lost_frequencies),
        "workflow_capabilities": profile.workflow_capabilities(),
        "completion_assessment": {
            "status": "insufficient_evidence",
            "reason": "full_six_position_comparison_required",
            "requires_listening_confirmation": True,
        },
        "current_score": current_score.to_dict(),
        "baseline_score": baseline_score.to_dict() if baseline_score else None,
        "final_dsp_eligible": False,
        "final_verdict_allowed": False,
        "requires_full_measurement": True,
        "requires_listening_confirmation": True,
        "tuning_state": "quick_candidate",
        "microphone_calibrated": profile.calibrated,
    }
    result_path = out / "quick_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    frequencies = np.asarray(ANALYSIS_FREQUENCIES_HZ)
    plt.figure(figsize=(11, 6))
    plt.semilogx(frequencies, target, "k--", label="Цель")
    if baseline_curve is not None:
        plt.semilogx(frequencies, baseline_curve, label="Baseline")
    plt.semilogx(frequencies, current_curve, marker="o", markersize=3, label="Текущий preset")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Frequency, Hz")
    plt.ylabel("dB после устойчивого выравнивания")
    plt.title(f"Быстрое сравнение: {profile.name}")
    plt.legend()
    plt.savefig(out / "quick_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    with (out / "quick_report.md").open("w", encoding="utf-8") as report:
        report.write(f"# Быстрый замер: {profile.name}\n\n")
        report.write(f"Вердикт: **{verdict}**. Один ESS, центральная позиция микрофона.\n\n")
        report.write(
            "Это только технический относительный кандидат. Итоговое улучшение подтверждается "
            "полным режимом и прослушиванием; состояние остаётся quick_candidate.\n\n"
        )
        if zone_delta is not None:
            report.write("## Изменение ошибок по зонам относительно baseline\n\n")
            report.write(pd.DataFrame([zone_delta]).to_markdown(index=False, floatfmt=".2f"))
            report.write("\n\n")
        report.write("## Текущая оценка\n\n")
        score_row = current_score.to_dict()
        score_row.pop("observed_frequency_mask")
        score_row.pop("target_optimization_mask")
        score_row.pop("confidence_weights")
        score_row.pop("zone_errors_db")
        score_row.pop("evaluation_range_metrics")
        report.write(pd.DataFrame([score_row]).to_markdown(index=False, floatfmt=".2f"))
        report.write("\n\n## Ошибка по зонам\n\n")
        report.write(pd.DataFrame([current_score.zone_errors_db]).to_markdown(index=False, floatfmt=".2f"))
        report.write("\n\n## Обязательные диапазоны\n\n")
        report.write(
            pd.DataFrame.from_dict(
                current_score.evaluation_range_metrics, orient="index"
            ).to_markdown(floatfmt=".2f")
        )
        report.write(
            f"\n\nНаблюдаемый диапазон: **{current_score.observed_low_hz:g}-"
            f"{current_score.observed_high_hz:g} Hz**. Точная оптимизация: "
            f"**{current_score.optimization_low_hz:g}-{current_score.optimization_high_hz:g} Hz**.\n"
        )
        if not profile.calibrated:
            report.write("\nМикрофон не калиброван: вывод является сравнительным, не абсолютным.\n")
    return result_path
