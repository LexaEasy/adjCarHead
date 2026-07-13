from __future__ import annotations

import numpy as np

from config import EQ_BANDS_HZ, NOMINAL_FREQUENCIES_HZ, TARGET_DB


def parse_eq(raw: str | None) -> dict[int, float]:
    settings = {band: 0 for band in EQ_BANDS_HZ}
    if not raw:
        return settings
    for item in raw.split(","):
        key, value = item.split("=", maxsplit=1)
        settings[int(key.strip())] = float(value.strip())
    return settings


def clamp_eq(value: float) -> float:
    return round(max(-9.0, min(9.0, value)), 1)


def mean_error(response: np.ndarray, lower: float, upper: float) -> float:
    freqs = np.array(NOMINAL_FREQUENCIES_HZ)
    ideal = np.array(TARGET_DB)
    mask = (freqs >= lower) & (freqs <= upper)
    if not np.any(mask):
        raise ValueError(f"No analysis bands between {lower} and {upper} Hz")
    return float(np.mean(ideal[mask] - response[mask]))


def reason_for(band: int, step: int) -> str:
    if step > 0:
        return {
            60: "80-125 Hz ниже цели; саббас 20-50 Hz не компенсируем агрессивно",
            230: "200-315 Hz ниже цели, можно мягко добавить тело",
            910: "500-1250 Hz явно ниже цели, можно осторожно поднять середину",
            3600: "1.6-5 kHz ниже цели, можно добавить ясность",
            14000: "8-16 kHz ниже цели, можно добавить воздух без резкости",
        }[band]
    if step < 0:
        return {
            60: "80-125 Hz выше цели; уменьшаем риск бубнения и лимитера",
            230: "200-315 Hz выше цели; можно убрать лишнюю мутность",
            910: "500-1250 Hz явно выше цели, можно осторожно ослабить середину",
            3600: "1.6-5 kHz выше цели; можно уменьшить резкость",
            14000: "8-16 kHz выше цели; можно уменьшить яркость и шипение",
        }[band]
    return "Зона близка к цели; оставляем без изменений"


def suggest_next(response: np.ndarray, current: dict[int, float]) -> list[dict[str, object]]:
    suggestions = []
    rules = [
        (60, mean_error(response, 80, 125)),
        (230, mean_error(response, 200, 315)),
        (910, mean_error(response, 500, 1250)),
        (3600, mean_error(response, 1600, 5000)),
        (14000, mean_error(response, 8000, 16000)),
    ]
    for band, error in rules:
        old = current.get(band, 0)
        step = 0.0
        if error > 3.0:
            step = 0.3
        elif error > 1.5:
            step = 0.2
        elif error > 0.7:
            step = 0.1
        elif error < -3.0:
            step = -0.3
        elif error < -1.5:
            step = -0.2
        elif error < -0.7:
            step = -0.1
        if band == 910 and abs(error) < 2.5:
            step = 0.0
        if band == 60 and mean_error(response, 125, 125) < -0.5:
            step = min(step, 0)
        reason = reason_for(band, step)
        if band == 910 and step == 0:
            reason = "500-1250 Hz около нормы; 910 Hz оставляем без изменений"
        if band == 60 and step == 0 and error > 1.5:
            reason = "125 Hz уже не ниже цели; 60 Hz не поднимаем агрессивно"
        suggestions.append(
            {
                "band": f"{band} Hz",
                "old": old,
                "new": clamp_eq(old + step),
                "reason": reason,
            }
        )
    return suggestions
