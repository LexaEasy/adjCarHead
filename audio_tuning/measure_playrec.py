from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

from artifact_identity import file_content_id, file_sha256
from config import (
    DEFAULT_POST_ROLL_SECONDS,
    DEFAULT_PRE_ROLL_SECONDS,
    TIMING_FINAL_ROLL_SECONDS,
    TIMING_MARKER_GUARD_SECONDS,
)
from device_profile import load_device_profile
from dsp import to_mono
from suggest_eq import parse_eq
from timing import build_timed_playback
from frequency_bands import ANALYSIS_SCHEMA_VERSION
from spatial_positions import SPATIAL_POSITION_KEYS, configure_utf8_console, spatial_sequence_text


CHANNEL_SELECTIONS = ("left", "right", "stereo")


def route_output(samples: np.ndarray, channel: str, output_channels: int) -> np.ndarray:
    if output_channels < 2:
        raise ValueError("Channel routing requires a two-channel output device")
    mono = to_mono(samples)
    routed = np.zeros((len(mono), output_channels), dtype=np.float64)
    if channel in {"left", "stereo"}:
        routed[:, 0] = mono
    if channel in {"right", "stereo"}:
        routed[:, 1] = mono
    return routed


def list_devices() -> None:
    print(sd.query_devices())


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Play a test signal and record microphone response.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--profile-name", default="preset")
    parser.add_argument("--device-profile", type=Path)
    parser.add_argument("--measurement-mode", choices=("quick", "full"))
    parser.add_argument("--session-purpose", choices=("baseline", "candidate"))
    parser.add_argument("--eq", default="")
    parser.add_argument("--signal", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/recordings/preset"))
    parser.add_argument("--analysis-out", type=Path)
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--output-channels", type=int, default=2)
    parser.add_argument("--channel", choices=CHANNEL_SELECTIONS, default="stereo")
    parser.add_argument("--channel-routing-verified", action="store_true")
    parser.add_argument("--split-streams", action="store_true")
    parser.add_argument("--analysis-method", choices=("ess", "psd-ratio"), default="ess")
    parser.add_argument("--inverse-filter", type=Path)
    parser.add_argument("--clock-correction", choices=("required", "off"), default="required")
    parser.add_argument("--timing-marker", type=Path)
    parser.add_argument("--pre-roll-s", type=float, default=DEFAULT_PRE_ROLL_SECONDS)
    parser.add_argument("--post-roll-s", type=float, default=DEFAULT_POST_ROLL_SECONDS)
    parser.add_argument("--volume-note", default="")
    parser.add_argument("--mic-position-note", default="")
    parser.add_argument("--spatial-session-id")
    parser.add_argument("--spatial-position", choices=SPATIAL_POSITION_KEYS)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return
    if not args.signal:
        raise SystemExit("--signal is required unless --list-devices is used")
    device_profile = load_device_profile(args.device_profile) if args.device_profile else None
    volume_note = args.volume_note or (device_profile.volume_note() if device_profile else "")
    spatial_requested = bool(args.spatial_session_id or args.spatial_position)
    if spatial_requested and not (args.spatial_session_id and args.spatial_position):
        raise SystemExit("--spatial-session-id and --spatial-position must be used together")
    if spatial_requested and (
        args.analysis_method != "ess" or args.clock_correction != "required" or not args.analyze
    ):
        raise SystemExit("Spatial positions require ESS, clock correction and --analyze")
    if spatial_requested and (
        args.input_device is None or args.output_device is None or not volume_note
    ):
        raise SystemExit("Spatial positions require explicit devices and --volume-note")
    inverse_filter = args.inverse_filter
    timing_marker_path = args.timing_marker
    clock_correction = args.clock_correction if args.analysis_method == "ess" else "off"
    if args.analyze and args.analysis_method == "ess":
        inverse_filter = inverse_filter or args.signal.with_name("ess_inverse.wav")
        if not inverse_filter.exists():
            raise SystemExit(f"ESS inverse filter not found: {inverse_filter}")
    if clock_correction == "required":
        timing_marker_path = timing_marker_path or args.signal.with_name("timing_marker.wav")
        if not timing_marker_path.exists():
            raise SystemExit(f"Timing marker not found: {timing_marker_path}")

    if args.spatial_position:
        print(spatial_sequence_text(args.spatial_position))
    print("Перед замером отключите AGC, шумоподавление и улучшайзеры микрофона.")
    print("Не меняйте громкость, положение микрофона и настройки аудиосистемы во время записи.")
    signal, sample_rate = sf.read(args.signal, always_2d=False)
    mono_signal = to_mono(signal)
    signal_playback = np.repeat(mono_signal[:, np.newaxis], args.output_channels, axis=1)
    if args.pre_roll_s < 0 or args.post_roll_s < 0:
        raise SystemExit("--pre-roll-s and --post-roll-s must be non-negative")
    pre_roll_s = args.pre_roll_s if args.analysis_method == "ess" else 0.0
    post_roll_s = args.post_roll_s if args.analysis_method == "ess" else 0.0
    timing_layout = None
    if clock_correction == "required":
        assert timing_marker_path is not None
        timing_marker, marker_rate = sf.read(timing_marker_path, always_2d=False)
        if marker_rate != sample_rate:
            raise SystemExit("Timing marker and ESS must have the same sample rate")
        playback, timing_layout = build_timed_playback(
            signal_playback,
            timing_marker,
            sample_rate,
            pre_roll_s,
            TIMING_MARKER_GUARD_SECONDS,
            post_roll_s,
            TIMING_FINAL_ROLL_SECONDS,
        )
    else:
        playback = np.pad(
            signal_playback,
            (
                (int(round(pre_roll_s * sample_rate)), int(round(post_roll_s * sample_rate))),
                (0, 0),
            ),
        )
    playback = route_output(playback, args.channel, args.output_channels)
    if args.split_streams:
        blocksize = 1024
        chunks = []
        captured_frames = 0

        def input_callback(indata, frames, _time, status) -> None:
            nonlocal captured_frames
            if status:
                print(status)
            if captured_frames < len(playback):
                need = len(playback) - captured_frames
                chunks.append(indata[: min(frames, need)].copy())
                captured_frames += min(frames, need)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            device=args.input_device,
            dtype="float32",
            blocksize=blocksize,
            callback=input_callback,
        ) as input_stream, sd.OutputStream(
            samplerate=sample_rate,
            channels=playback.shape[1],
            device=args.output_device,
            dtype="float32",
            blocksize=blocksize,
        ) as output_stream:
            for start in range(0, len(playback), blocksize):
                chunk = playback[start : start + blocksize].astype("float32")
                output_stream.write(chunk)
            deadline = time.monotonic() + 2.0
            while captured_frames < len(playback) and time.monotonic() < deadline:
                time.sleep(0.01)
        recorded = np.vstack(chunks)
    else:
        recorded = sd.playrec(
            playback,
            samplerate=sample_rate,
            channels=1,
            input_mapping=None,
            output_mapping=None,
            device=(args.input_device, args.output_device),
            blocking=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    wav_path = args.out / f"{args.profile_name}.wav"
    sf.write(wav_path, recorded, sample_rate, subtype="FLOAT")
    eq_settings = device_profile.parse_eq(args.eq) if device_profile else parse_eq(args.eq)
    metadata = {
        "profile_name": args.profile_name,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "eq_settings": eq_settings,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "input_device": args.input_device,
        "output_device": args.output_device,
        "sample_rate": sample_rate,
        "analysis_method": args.analysis_method,
        "clock_correction": clock_correction,
        "pre_roll_s": pre_roll_s,
        "post_roll_s": post_roll_s,
        "volume_note": volume_note,
        "mic_position_note": args.mic_position_note,
        "mic_position_id": args.spatial_position or args.mic_position_note,
        "spatial_session_id": args.spatial_session_id,
        "spatial_position": args.spatial_position,
        "measurement_mode": args.measurement_mode,
        "session_purpose": args.session_purpose,
        "channel_selection": args.channel,
        "channel_routing_verified": args.channel_routing_verified,
        "device_profile_path": str(args.device_profile) if args.device_profile else None,
        "device_profile_schema": device_profile.schema_version if device_profile else None,
        "device_profile_id": device_profile.device_id if device_profile else None,
        "target_profile": device_profile.target_name if device_profile else None,
        "microphone_profile_id": (
            device_profile.microphone_profile.profile_id if device_profile else None
        ),
        "microphone_calibrated": device_profile.calibrated if device_profile else False,
        "processing_settings": device_profile.processing if device_profile else None,
        "equipment": device_profile.equipment_metadata() if device_profile else None,
        "delay_settings": device_profile.delays if device_profile else None,
        "signal": str(args.signal),
        "source_signal_id": file_content_id(args.signal, "ess"),
        "inverse_filter": str(inverse_filter) if inverse_filter is not None else None,
        "inverse_filter_id": (
            file_content_id(inverse_filter, "ess_inverse") if inverse_filter is not None else None
        ),
        "system_profile_hash": file_sha256(device_profile.source_path) if device_profile else None,
        "microphone_profile_hash": (
            file_sha256(device_profile.microphone_profile.source_path)
            if device_profile and device_profile.microphone_profile.source_path
            else None
        ),
        "ess_parameters": {
            "duration_s": len(to_mono(signal)) / sample_rate,
            "start_hz": device_profile.sweep_start_hz if device_profile else None,
            "end_hz": device_profile.sweep_end_hz if device_profile else None,
        },
        "timing_marker": str(timing_marker_path) if timing_marker_path is not None else None,
        "timing_layout": timing_layout.to_dict() if timing_layout is not None else None,
        "recorded": str(wav_path),
    }
    metadata_path = args.out / f"{args.profile_name}.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.analyze:
        analysis_out = args.analysis_out
        if analysis_out is None and spatial_requested:
            analysis_out = args.out / "run_analysis"
        command = [
            sys.executable,
            str(Path(__file__).with_name("analyze_recordings.py")),
            "--source-signal",
            str(args.signal),
            "--recorded-signal",
            str(wav_path),
            "--analysis-method",
            args.analysis_method,
            "--clock-correction",
            clock_correction,
            "--profile-name",
            args.profile_name,
            "--eq",
            args.eq,
        ]
        if analysis_out is not None:
            command.extend(("--out", str(analysis_out)))
        if args.analysis_method == "ess":
            assert inverse_filter is not None
            command.extend(("--inverse-filter", str(inverse_filter)))
        if args.device_profile is not None:
            command.extend(("--device-profile", str(args.device_profile)))
        if clock_correction == "required":
            assert timing_marker_path is not None
            command.extend(("--timing-marker", str(timing_marker_path)))
            command.extend(("--measurement-metadata", str(metadata_path)))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
