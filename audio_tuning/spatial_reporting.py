from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ANALYSIS_FREQUENCIES_HZ, NOMINAL_FREQUENCIES_HZ
from device_profile import DeviceProfile
from dsp_model import suggest_dsp
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from spatial_analysis import SpatialResult
from spatial_positions import SPATIAL_SCHEMA_VERSION, spatial_sequence_text
from scoring import score_response
from targets import target_curve_db


def write_spatial_outputs(
    out: Path,
    result: SpatialResult,
    profile: DeviceProfile | None = None,
    session_purpose: str = "baseline",
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    offset_db = result.alignment.offset_db
    p10_aligned = result.p10_db - offset_db
    p90_aligned = result.p90_db - offset_db
    table = pd.DataFrame(
        {
            "Spatial_Schema": SPATIAL_SCHEMA_VERSION,
            "Analysis_Schema": ANALYSIS_SCHEMA_VERSION,
            "Frequency_Hz": NOMINAL_FREQUENCIES_HZ,
            "Exact_Center_Hz": ANALYSIS_FREQUENCIES_HZ,
            "Target_dB": result.target_db,
            "Spatial_Raw_Mean_dB": result.raw_mean_db,
            "Spatial_Shape_Mean_dB": result.aligned_mean_db,
            "Spatial_StdDev_dB": result.standard_deviation_db,
            "Spatial_P10_Shape_dB": p10_aligned,
            "Spatial_P90_Shape_dB": p90_aligned,
        }
    )
    table.to_csv(out / "spatial_frequency_table.csv", index=False)
    frequencies = np.asarray(ANALYSIS_FREQUENCIES_HZ)
    plt.figure(figsize=(11, 6))
    plt.semilogx(frequencies, result.target_db, "k--", label="Целевая кривая")
    plt.fill_between(
        frequencies,
        p10_aligned,
        p90_aligned,
        alpha=0.2,
        label="Разброс P10-P90",
    )
    plt.semilogx(frequencies, result.aligned_mean_db, marker="o", label="Среднее 6 позиций")
    plt.title("Пространственно усреднённая форма АЧХ")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("dB после устойчивого выравнивания")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(out / "spatial_frequency_response.png", dpi=160, bbox_inches="tight")
    plt.close()

    stored_eq = result.invariants["eq_settings"]
    if not isinstance(stored_eq, dict):
        raise ValueError("Spatial session does not contain EQ settings")
    current_eq = profile.default_eq() if profile is not None else {}
    for band, value in stored_eq.items():
        key = str(band)
        if profile is not None and key not in current_eq:
            raise ValueError(f"Spatial session contains an unknown DSP control: {key}")
        current_eq[key] = float(value)
    score = None
    recommendation = None
    recommendation_eligible = bool(profile and profile.dsp_recommendation_eligible)
    response_payload = result.to_dict()
    response_payload["session_purpose"] = session_purpose
    response_payload["tuning_state"] = (
        "full_baseline_measured" if session_purpose == "baseline" else "full_candidate_measured"
    )
    response_payload["final_eq_eligible"] = False
    response_payload["dsp_recommendation_eligible"] = recommendation_eligible
    response_payload["final_eq_block_reason"] = (
        profile.dsp_recommendation_block_reason() if profile is not None else "Не передан профиль системы"
    )
    response_payload["equipment"] = profile.equipment_metadata() if profile is not None else None
    if profile is not None:
        score = score_response(
            result.aligned_mean_db,
            result.target_db,
            profile,
            result.quality,
            result.standard_deviation_db,
            current_eq,
        )
        if recommendation_eligible:
            dsp_frequencies = np.asarray(ANALYSIS_FREQUENCIES_HZ)
            dsp_response = result.aligned_mean_db
            dsp_target = result.target_db
            dsp_spatial = result.standard_deviation_db
            dsp_mask = np.asarray(score.target_optimization_mask, dtype=bool)
            if result.smoothed_frequencies_hz is not None:
                assert result.smoothed_aligned_mean_db is not None
                assert result.smoothed_standard_deviation_db is not None
                dsp_frequencies = result.smoothed_frequencies_hz
                dsp_response = result.smoothed_aligned_mean_db
                dsp_target = target_curve_db(profile.target_name, dsp_frequencies)
                analysis_frequencies = np.asarray(ANALYSIS_FREQUENCIES_HZ)
                nearest = np.argmin(
                    np.abs(
                        np.log(dsp_frequencies)[:, np.newaxis]
                        - np.log(analysis_frequencies)[np.newaxis, :]
                    ),
                    axis=1,
                )
                dsp_mask = np.asarray(score.target_optimization_mask, dtype=bool)[nearest]
                dsp_spatial = result.smoothed_standard_deviation_db
            context = {**result.invariants, "analysis_schema_version": ANALYSIS_SCHEMA_VERSION}
            recommendation = suggest_dsp(
                dsp_frequencies,
                dsp_response,
                dsp_target,
                dsp_mask,
                profile,
                current_eq,
                dsp_spatial,
                result.quality,
                context,
            )
    response_payload["score"] = score.to_dict() if score is not None else None
    response_payload["dsp_recommendation"] = recommendation
    response_path = out / "spatial_response.json"
    response_path.write_text(
        json.dumps(response_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out / "spatial_report.md").open("w", encoding="utf-8") as report:
        report.write("# Пространственный ESS-отчёт\n\n")
        report.write(f"Сессия: `{result.session_id}`. Полный комплект: 6 из 6 позиций.\n\n")
        report.write(spatial_sequence_text())
        report.write("\n\nИмпульсные характеристики между позициями не усреднялись.\n\n")
        report.write(f"Целевой профиль: `{result.target_name}`.\n\n")
        report.write("Задержки DSP сохранены как внешние метаданные и не оптимизируются.\n\n")
        if score is not None:
            report.write("## Итоговая оценка\n\n")
            report.write(
                pd.DataFrame([score.to_dict()])
                .drop(
                    columns=[
                        "observed_frequency_mask",
                        "target_optimization_mask",
                        "confidence_weights",
                        "zone_errors_db",
                    ]
                )
                .to_markdown(index=False, floatfmt=".2f")
            )
            report.write("\n\n## Ошибка по зонам\n\n")
            report.write(pd.DataFrame([score.zone_errors_db]).to_markdown(index=False, floatfmt=".2f"))
            report.write(
                f"\n\nНаблюдаемый диапазон: **{score.observed_low_hz:g}-"
                f"{score.observed_high_hz:g} Hz**. Точная оптимизация: "
                f"**{score.optimization_low_hz:g}-{score.optimization_high_hz:g} Hz**.\n"
            )
        report.write("\n\n## Пространственный разброс\n\n")
        report.write(
            table[["Frequency_Hz", "Spatial_StdDev_dB", "Spatial_P10_Shape_dB", "Spatial_P90_Shape_dB"]].to_markdown(
                index=False,
                floatfmt=".2f",
            )
        )
        report.write("\n\n## Контроль качества\n\n```json\n")
        report.write(json.dumps(result.quality, ensure_ascii=False, indent=2))
        report.write("\n```\n")
        report.write("\n## Технический следующий шаг DSP\n\n")
        if recommendation is None:
            reason = (
                profile.dsp_recommendation_block_reason()
                if profile is not None
                else "Не передан профиль устройства: автоматическая рекомендация отключена."
            )
            report.write(f"{reason}\n")
        elif recommendation["recommendation_accepted"]:
            changed = [item for item in recommendation["suggestions"] if item["old"] != item["new"]]
            report.write(pd.DataFrame(changed).to_markdown(index=False))
            report.write("\n\nРекомендация является только следующим техническим шагом и требует повторного full.\n")
        else:
            reasons = ", ".join(recommendation["recommendation_rejection_reasons"])
            report.write(f"Прогноз не прошёл защитные ограничения: `{reasons}`. Изменения не выдаются.\n")
        if profile is not None and not profile.calibrated:
            report.write("\nРезультат сравнительный: у микрофона нет активного калибровочного файла.\n")
        if profile is not None:
            missing = [key for key, verified in profile.validation.items() if not verified]
            if missing:
                report.write(f"\nНеподтверждённые одноразовые проверки: `{', '.join(missing)}`.\n")
    return response_path
