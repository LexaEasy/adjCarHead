from __future__ import annotations

from dataclasses import dataclass
import sys


SPATIAL_SCHEMA_VERSION = "six_fixed_ess_positions_v1"


@dataclass(frozen=True)
class SpatialPosition:
    key: str
    label: str
    instruction: str


SPATIAL_POSITIONS = (
    SpatialPosition("left_ear", "Левое ухо", "Поместите капсюль в точку левого уха."),
    SpatialPosition("right_ear", "Правое ухо", "Поместите капсюль в точку правого уха."),
    SpatialPosition(
        "front",
        "Впереди",
        "Поместите капсюль на 6 см впереди середины между ушами.",
    ),
    SpatialPosition(
        "back",
        "Позади",
        "Поместите капсюль на 6 см позади середины между ушами.",
    ),
    SpatialPosition(
        "up",
        "Выше",
        "Поместите капсюль на 6 см выше середины между ушами.",
    ),
    SpatialPosition(
        "down",
        "Ниже",
        "Поместите капсюль на 6 см ниже середины между ушами.",
    ),
)

SPATIAL_POSITION_KEYS = tuple(position.key for position in SPATIAL_POSITIONS)


def configure_utf8_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def spatial_sequence_text(current_key: str | None = None) -> str:
    if current_key is not None and current_key not in SPATIAL_POSITION_KEYS:
        raise ValueError(f"Unknown spatial position: {current_key}")
    lines = ["Последовательность шести фиксированных положений микрофона:"]
    for number, position in enumerate(SPATIAL_POSITIONS, start=1):
        marker = " <-- ТЕКУЩАЯ ПОЗИЦИЯ" if position.key == current_key else ""
        lines.append(f"{number}. {position.label}: {position.instruction}{marker}")
    lines.append(
        "Ориентацию капсюля, штатив, сиденье, громкость и gain между позициями не менять."
    )
    return "\n".join(lines)
