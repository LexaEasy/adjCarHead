from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ANALYSIS_FREQUENCIES_HZ,
    NOMINAL_FREQUENCIES_HZ,
    TARGET_DB,
)
from dsp import zone_errors
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from targets import target_curve_db as target_profile_curve_db


def build_table(
    profiles: dict[str, np.ndarray],
    default_name: str | None,
    raw_profiles: dict[str, np.ndarray] | None = None,
    target_db: np.ndarray | None = None,
) -> pd.DataFrame:
    selected_target = np.asarray(TARGET_DB if target_db is None else target_db)
    table = pd.DataFrame(
        {
            "Analysis_Schema": ANALYSIS_SCHEMA_VERSION,
            "Frequency_Hz": NOMINAL_FREQUENCIES_HZ,
            "Exact_Center_Hz": ANALYSIS_FREQUENCIES_HZ,
            "Ideal_dB": selected_target,
        }
    )
    raw_profiles = raw_profiles or {}
    default = profiles.get(default_name) if default_name else None
    for name, curve in profiles.items():
        if name in raw_profiles:
            table[f"{name}_Raw_dB"] = raw_profiles[name]
        table[f"{name}_dB"] = curve
        table[f"AbsDev_{name}_dB"] = np.abs(curve - selected_target)
        if default is not None and name != default_name:
            table[f"{name}_minus_default_dB"] = curve - default
            table[f"AbsDev_delta_vs_default_{name}_dB"] = table[f"AbsDev_{name}_dB"] - table[
                f"AbsDev_{default_name}_dB"
            ]
    return table


def plot_profiles(
    out: Path,
    profiles: dict[str, np.ndarray],
    continuous_profiles: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    target_db: np.ndarray | None = None,
    target_name: str = "warm_driver",
    device_name: str = "Аудиосистема",
) -> None:
    freqs = np.array(ANALYSIS_FREQUENCIES_HZ)
    ideal = np.asarray(TARGET_DB if target_db is None else target_db)
    continuous_profiles = continuous_profiles or {}
    plt.figure(figsize=(11, 6))
    plt.semilogx(freqs, ideal, "k--", label="Идеальный уровень")
    for name, curve in profiles.items():
        if name in continuous_profiles:
            smooth_freqs, smooth_curve = continuous_profiles[name]
            plt.semilogx(smooth_freqs, smooth_curve, label=f"{name}, 1/6 окт.")
            plt.semilogx(freqs, curve, "o", markersize=3, alpha=0.65, label=f"{name}, 1/3 окт.")
        else:
            plt.semilogx(freqs, curve, marker="o", label=name)
    plt.title(f"{device_name}: сравнение формы частотных кривых")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("dB после устойчивого выравнивания")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(out / "frequency_profiles.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.semilogx(freqs, np.zeros_like(freqs), "k--", label="Идеал: 0 dB")
    for name, curve in profiles.items():
        if name in continuous_profiles:
            smooth_freqs, smooth_curve = continuous_profiles[name]
            smooth_target = target_profile_curve_db(target_name, smooth_freqs)
            plt.semilogx(
                smooth_freqs,
                np.abs(smooth_curve - smooth_target),
                label=f"{name}, 1/6 окт.",
            )
            plt.semilogx(
                freqs,
                np.abs(curve - ideal),
                "o",
                markersize=3,
                alpha=0.65,
                label=f"{name}, 1/3 окт.",
            )
        else:
            plt.semilogx(freqs, np.abs(curve - ideal), marker="o", label=name)
    plt.title(f"{device_name}: модуль отклонения от цели")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("|Δ| от идеала, dB")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(out / "absolute_deviation.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_report(
    out: Path,
    table: pd.DataFrame,
    profiles: dict[str, np.ndarray],
    quality: dict[str, object],
    profile_alignments: dict[str, dict[str, object]],
    default_name: str,
    analysis_method: str,
    target_db: np.ndarray | None = None,
    device_name: str = "Аудиосистема",
) -> None:
    ideal = np.asarray(TARGET_DB if target_db is None else target_db)
    errors = {name: zone_errors(curve, ideal) for name, curve in profiles.items()}
    best = min(errors, key=lambda name: errors[name]["Full"])
    baseline_offset = float(profile_alignments[default_name]["offset_db"])
    alignment_rows = []
    for name, diagnostics in profile_alignments.items():
        offset_db = float(diagnostics["offset_db"])
        alignment_rows.append(
            {
                "Profile": name,
                "AlignmentOffset_dB": offset_db,
                "RelativeToBaseline_dB": offset_db - baseline_offset,
                "MAD_dB": float(diagnostics["mad_db"]),
                "Outliers": int(diagnostics["outlier_count"]),
                "AbsoluteSPLReliable": False,
            }
        )
    with (out / "report.md").open("w", encoding="utf-8") as file:
        file.write(f"# Отчёт по измерению: {device_name}\n\n")
        file.write(f"Схема частотного анализа: `{ANALYSIS_SCHEMA_VERSION}`.\n\n")
        file.write(f"Метод анализа пары source/recorded: `{analysis_method}`. Результат не считается полной безэховой АЧХ.\n\n")
        if analysis_method == "ess":
            file.write("ESS-деконволюция формирует импульсную характеристику и амплитудный отклик. Статус clock correction указан в timing diagnostics; фаза, абсолютная и межпрогонная Bluetooth-задержка пока не считаются достоверными.\n\n")
        elif analysis_method == "psd-ratio":
            file.write("PSD ratio является только сравнительной оценкой отношения спектральных мощностей, а не строгой transfer function.\n\n")
        file.write(f"Лучший профиль по Full shape MAE: **{best}**.\n\n")
        file.write("## Вывод\n\n")
        file.write("- Где стало лучше/хуже: для одного профиля сравнение недоступно; используйте несколько профилей или default.zip.\n")
        file.write("- Риск ложного улучшения: отражения, AGC, шумоподавление и изменение громкости могут исказить результат.\n")
        file.write("- Практический вывод: одиночная позиция используется для диагностики, но не формирует финальный EQ.\n\n")
        file.write("## Сводка по зонам\n\n")
        file.write(pd.DataFrame(errors).T.to_markdown(floatfmt=".2f"))
        file.write("\n\n## Выравнивание уровня\n\n")
        file.write(pd.DataFrame(alignment_rows).to_markdown(index=False, floatfmt=".2f"))
        file.write(
            "\n\n`AlignmentOffset_dB` является относительным смещением измерительного "
            "тракта, а не калиброванным SPL. Сравнивать его можно только при одинаковых "
            "методе, gain, громкости, расстоянии и положении микрофона.\n"
        )
        file.write("\n\n## Таблица частот\n\n")
        file.write(table.to_markdown(index=False, floatfmt=".2f"))
        file.write("\n\n## Контроль качества\n\n```json\n")
        file.write(json.dumps(quality, ensure_ascii=False, indent=2))
        file.write("\n```\n\n## Предупреждения\n\n")
        file.write("- Если RMS, длительность или положение микрофона отличаются между профилями, сравнение может быть некорректным.\n")
        file.write("- Если есть клиппинг или много тишины, запись нужно повторить.\n\n")
        file.write("## Рекомендация следующего EQ preset\n\n")
        file.write(
            "Финальная рекомендация отключена: сначала соберите все шесть фиксированных "
            "ESS-позиций и выполните `aggregate_spatial.py`."
        )
        file.write("\n\nЧто проверить на слух: бубнение, резкость вокала, усталость от ВЧ, потерю плотности баса.\n\n")
        file.write("Какие треки использовать: знакомые записи с вокалом, плотным басом, тарелками и натуральными инструментами.\n\n")
        file.write("Какие признаки ухудшения: лимитер, хрип, утомляющая яркость, провал голоса, размазанный бас.\n\n")
        file.write("Нужно ли повторить замер: да, если изменилась громкость, позиция микрофона или RMS/длительность заметно отличаются.\n")
