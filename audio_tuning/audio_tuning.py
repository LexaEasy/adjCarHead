from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

from config import ANALYSIS_FREQUENCIES_HZ
from channel_comparison import write_channel_comparison
from device_profile import DeviceProfile, load_device_profile
from full_comparison import write_full_comparison
from generate_test_signals import generate_all
from quick_reporting import write_quick_outputs
from spatial_analysis import aggregate_spatial_payloads, load_spatial_payload
from spatial_positions import SPATIAL_POSITIONS, configure_utf8_console, spatial_sequence_text
from spatial_reporting import write_spatial_outputs
from targets import target_curve_db
from tuning_state import confirm_listening


ROOT = Path(__file__).resolve().parent


def add_measurement_arguments(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--device-profile", type=Path, required=required)
    parser.add_argument("--profile-name", required=required)
    parser.add_argument("--eq", default="")
    parser.add_argument("--input-device", type=int, required=required)
    parser.add_argument("--output-device", type=int, required=required)
    parser.add_argument("--split-streams", action="store_true")
    parser.add_argument("--channel", choices=("left", "right", "stereo"), default="stereo")
    parser.add_argument("--channel-routing-verified", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/outputs"))
    parser.add_argument("--dry-run", action="store_true", help="Prepare signals and print the protocol")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-mode automotive audio measurement tool.")
    parser.add_argument("--list-devices", action="store_true")
    subparsers = parser.add_subparsers(dest="mode")
    quick = subparsers.add_parser("quick", help="One-playback preset screening")
    add_measurement_arguments(quick)
    quick.add_argument("--yes", action="store_true", help="Skip the single playback confirmation")
    quick.add_argument("--baseline-result", type=Path)
    full = subparsers.add_parser("full", help="Six-position final DSP refinement")
    add_measurement_arguments(full, required=False)
    full.add_argument("--session-purpose", choices=("baseline", "candidate"), default="baseline")
    full.add_argument("--baseline-result", type=Path)
    full.add_argument("--confirm-listening-result", type=Path)
    full.add_argument("--confirmed-by")
    full.add_argument("--listening-notes")
    full.add_argument("--compare-channels", nargs=3, type=Path, metavar=("LEFT", "RIGHT", "STEREO"))
    full.set_defaults(yes=False)
    return parser


def prepare_signals(profile: DeviceProfile, mode: str) -> Path:
    duration = profile.quick_duration_s if mode == "quick" else profile.full_duration_s
    out = ROOT / "data" / "test_signals" / profile.device_id / mode
    generate_all(out, duration, duration, profile.sweep_start_hz, profile.sweep_end_hz)
    return out


def print_preflight(args: argparse.Namespace, profile: DeviceProfile, signal_dir: Path) -> None:
    duration = profile.quick_duration_s if args.mode == "quick" else profile.full_duration_s
    count = 1 if args.mode == "quick" else len(SPATIAL_POSITIONS)
    print(f"Устройство: {profile.name} ({profile.device_id})")
    print(f"Воспроизведение: output device {args.output_device}, канал {args.channel}")
    print(f"Запись: input device {args.input_device}, {profile.microphone.get('name', 'микрофон')}")
    print(
        "Ориентация микрофона: "
        f"{profile.microphone_profile.orientation.get('vehicle_rule', 'зафиксировать относительно автомобиля')}"
    )
    print(f"Сигнал: ESS {profile.sweep_start_hz:g}-{profile.sweep_end_hz:g} Hz, {duration:g} s")
    print(f"Количество воспроизведений: {count}")
    print(f"Уровни и состояние тракта: {profile.volume_note()}")
    print(f"Сигналы: {signal_dir}")
    print("На выходе: WAV, metadata, ESS-анализ, график и отчёт.")
    if args.mode == "quick":
        print("Быстрый режим не выдаёт финальный DSP-пресет.")
    else:
        print(f"Назначение full-сессии: {args.session_purpose}")
        print(spatial_sequence_text())
        block_reason = profile.dsp_recommendation_block_reason()
        if block_reason:
            print(block_reason)


def run_measurement(
    args: argparse.Namespace,
    profile: DeviceProfile,
    signal_dir: Path,
    run_id: str,
    position_key: str | None = None,
    session_id: str | None = None,
) -> Path:
    suffix = position_key or "center"
    name = f"{args.profile_name}_{suffix}"
    recording_out = ROOT / "data" / "recordings" / profile.device_id / run_id / suffix
    analysis_out = args.out / profile.device_id / args.mode / f"run_{run_id}_{suffix}"
    command = [
        sys.executable, str(ROOT / "measure_playrec.py"),
        "--profile-name", name,
        "--device-profile", str(args.device_profile.resolve()),
        "--measurement-mode", args.mode,
        "--eq", args.eq,
        "--signal", str(signal_dir / "ess_sweep.wav"),
        "--inverse-filter", str(signal_dir / "ess_inverse.wav"),
        "--timing-marker", str(signal_dir / "timing_marker.wav"),
        "--analysis-method", "ess", "--clock-correction", "required",
        "--out", str(recording_out), "--analysis-out", str(analysis_out),
        "--input-device", str(args.input_device), "--output-device", str(args.output_device),
        "--channel", args.channel,
        "--mic-position-note", "center_between_ears" if position_key is None else position_key,
        "--analyze",
    ]
    if args.split_streams:
        command.append("--split-streams")
    if args.channel_routing_verified:
        command.append("--channel-routing-verified")
    session_purpose = getattr(args, "session_purpose", None)
    if session_purpose is not None:
        command.extend(("--session-purpose", session_purpose))
    if position_key is not None and session_id is not None:
        command.extend(("--spatial-session-id", session_id, "--spatial-position", position_key))
    subprocess.run(command, check=True, cwd=ROOT)
    return analysis_out / "ess_response.json"


def confirm_playback(args: argparse.Namespace, text: str) -> None:
    if not args.yes:
        input(f"{text} Нажмите Enter для воспроизведения или Ctrl+C для остановки: ")


def run_quick(args: argparse.Namespace, profile: DeviceProfile, signal_dir: Path) -> None:
    confirm_playback(args, "Установите микрофон в центр между ушами водителя.")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = run_measurement(args, profile, signal_dir, run_id)
    out = result_path.parent
    write_quick_outputs(out, result_path, profile, args.baseline_result)
    print(out)


def run_full(args: argparse.Namespace, profile: DeviceProfile, signal_dir: Path) -> None:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_paths = []
    for index, position in enumerate(SPATIAL_POSITIONS, start=1):
        print(spatial_sequence_text(position.key))
        confirm_playback(args, f"Позиция {index}/6: {position.instruction}")
        result_paths.append(run_measurement(args, profile, signal_dir, session_id, position.key, session_id))
    payloads = [load_spatial_payload(path) for path in result_paths]
    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ)
    result = aggregate_spatial_payloads(payloads, target, profile.target_name)
    out = args.out / profile.device_id / "full" / f"run_{session_id}_summary"
    response_path = write_spatial_outputs(out, result, profile, args.session_purpose)
    (out / "source_results.json").write_text(
        json.dumps([str(path.resolve()) for path in result_paths], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.session_purpose == "candidate":
        if args.baseline_result is None:
            raise ValueError("Candidate full session requires --baseline-result")
        comparison_path = write_full_comparison(
            out,
            args.baseline_result.resolve(),
            response_path,
            profile,
        )
        print(comparison_path)
    else:
        print(response_path)


def main() -> None:
    configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args()
    if args.list_devices:
        subprocess.run([sys.executable, str(ROOT / "measure_playrec.py"), "--list-devices"], check=True, cwd=ROOT)
        return
    if args.mode not in {"quick", "full"}:
        parser.error("Choose quick or full, or use --list-devices")
    if not args.out.is_absolute():
        args.out = ROOT / args.out
    if args.mode == "full" and args.confirm_listening_result is not None:
        if not args.confirmed_by or not args.listening_notes:
            parser.error("Listening confirmation requires --confirmed-by and --listening-notes")
        output = confirm_listening(
            args.confirm_listening_result.resolve(),
            args.confirmed_by,
            args.listening_notes,
        )
        print(output)
        return
    if args.mode == "full" and args.compare_channels is not None:
        out = args.out / "channel_comparison" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        print(write_channel_comparison(out, [path.resolve() for path in args.compare_channels]))
        return
    required = ("device_profile", "profile_name", "input_device", "output_device")
    missing = [field for field in required if getattr(args, field, None) is None]
    if missing:
        parser.error("Measurement requires: " + ", ".join(f"--{field.replace('_', '-')}" for field in missing))
    args.device_profile = args.device_profile.resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.profile_name) is None:
        parser.error("--profile-name must contain only letters, digits, _ or -")
    if args.baseline_result is not None:
        args.baseline_result = args.baseline_result.resolve()
    profile = load_device_profile(args.device_profile)
    profile.parse_eq(args.eq)
    signal_dir = prepare_signals(profile, args.mode)
    print_preflight(args, profile, signal_dir)
    if args.dry_run:
        return
    if not profile.volume_reference_ready:
        raise SystemExit(
            "Рабочая громкость источника и магнитолы не зафиксирована в профиле; "
            "до её заполнения доступен только --dry-run."
        )
    if args.mode == "quick":
        run_quick(args, profile, signal_dir)
    else:
        run_full(args, profile, signal_dir)


if __name__ == "__main__":
    main()
