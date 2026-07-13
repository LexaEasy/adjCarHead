from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class TuningState(str, Enum):
    UNVALIDATED = "unvalidated"
    QUICK_CANDIDATE = "quick_candidate"
    FULL_BASELINE_MEASURED = "full_baseline_measured"
    FULL_CANDIDATE_MEASURED = "full_candidate_measured"
    FULL_COMPARISON_PASSED = "full_comparison_passed"
    LISTENING_CONFIRMATION_REQUIRED = "listening_confirmation_required"
    CONFIRMED_PRESET = "confirmed_preset"


ALLOWED_TRANSITIONS = {
    TuningState.UNVALIDATED: {TuningState.QUICK_CANDIDATE, TuningState.FULL_BASELINE_MEASURED},
    TuningState.QUICK_CANDIDATE: {TuningState.FULL_BASELINE_MEASURED},
    TuningState.FULL_BASELINE_MEASURED: {TuningState.FULL_CANDIDATE_MEASURED},
    TuningState.FULL_CANDIDATE_MEASURED: {TuningState.FULL_COMPARISON_PASSED},
    TuningState.FULL_COMPARISON_PASSED: {TuningState.LISTENING_CONFIRMATION_REQUIRED},
    TuningState.LISTENING_CONFIRMATION_REQUIRED: {TuningState.CONFIRMED_PRESET},
    TuningState.CONFIRMED_PRESET: set(),
}


def require_transition(current: TuningState, target: TuningState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid tuning state transition: {current.value} -> {target.value}")


def confirm_listening(
    comparison_path: Path,
    confirmed_by: str,
    listening_notes: str,
) -> Path:
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Full comparison must be a JSON object")
    current = TuningState(str(payload.get("tuning_state")))
    require_transition(current, TuningState.CONFIRMED_PRESET)
    if not confirmed_by.strip() or not listening_notes.strip():
        raise ValueError("Listening confirmation requires operator and notes")
    result = {
        **payload,
        "tuning_state": TuningState.CONFIRMED_PRESET.value,
        "listening_confirmed": True,
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
        "confirmed_by": confirmed_by.strip(),
        "listening_notes": listening_notes.strip(),
        "final_verdict_allowed": True,
        "final_dsp_eligible": True,
    }
    output = comparison_path.with_name("confirmed_preset.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
