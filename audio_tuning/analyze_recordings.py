from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from artifact_identity import file_content_id
from config import ANALYSIS_FREQUENCIES_HZ, TARGET_DB
from device_profile import load_device_profile
from dsp import psd_ratio_estimate, quality_metrics, third_octave_levels, to_mono
from ess import deconvolve_ess
from frequency_bands import (
    ANALYSIS_SCHEMA_VERSION,
    EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
    NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ,
)
from measurement_quality import assess_measurement
from microphone_calibration import calibration_correction_db
from reporting import build_table, plot_profiles, write_report
from response_alignment import align_response_to_target
from timing import TimingLayout, correct_clock_drift
from targets import target_curve_db


def timestamped_out(base: Path | None) -> Path:
    root = base or Path("data/outputs")
    out = root if root.name.startswith("run_") else root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, always_2d=False)
    return samples, int(sample_rate)


def locate_active_ess(recording: np.ndarray, source: np.ndarray) -> int:
    recorded_mono = to_mono(recording)
    source_mono = to_mono(source)
    if len(recorded_mono) < len(source_mono):
        return 0
    correlation = fftconvolve(recorded_mono, source_mono[::-1], mode="valid")
    return int(np.argmax(np.abs(correlation)))


def analyze_zip(
    zip_path: Path,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray, dict[str, object]]:
    rows = []
    raw_curves = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        for wav in sorted(Path(tmp).rglob("*.wav")):
            samples, sample_rate = read_wav(wav)
            raw_response = third_octave_levels(samples, sample_rate)
            raw_curves.append(raw_response)
            alignment = align_response_to_target(raw_response, np.asarray(TARGET_DB))
            quality = quality_metrics(samples, sample_rate)
            rows.append(
                {
                    "file": wav.name,
                    **quality.__dict__,
                    "response_alignment": alignment.diagnostics(),
                }
            )
    if not raw_curves:
        raise ValueError(f"No WAV files found in {zip_path}")
    raw_mean = np.mean(np.vstack(raw_curves), axis=0)
    alignment = align_response_to_target(raw_mean, np.asarray(TARGET_DB))
    return alignment.aligned_response_db, rows, raw_mean, alignment.diagnostics()


def analyze_pair(
    source_path: Path,
    recorded_path: Path,
    analysis_method: str,
    inverse_filter_path: Path | None,
    clock_correction_mode: str,
    timing_marker_path: Path | None,
    measurement_metadata_path: Path | None,
    device_profile_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, object], np.ndarray | None, np.ndarray]:
    source, source_rate = read_wav(source_path)
    recorded, recorded_rate = read_wav(recorded_path)
    if source_rate != recorded_rate:
        raise ValueError("Source and recorded WAV files must have the same sample rate")
    device_profile = load_device_profile(device_profile_path) if device_profile_path else None
    target_name = device_profile.target_name if device_profile else "warm_driver"
    target_db = target_curve_db(target_name, ANALYSIS_FREQUENCIES_HZ)
    metadata: dict[str, object] = {
        "method": analysis_method,
        "analysis_method": "ess_deconvolution" if analysis_method == "ess" else "psd_ratio",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "target_profile": target_name,
        "microphone_calibrated": bool(device_profile and device_profile.calibrated),
        "equipment": device_profile.equipment_metadata() if device_profile else None,
        "measurement_id": file_content_id(recorded_path, "recording"),
    }
    measurement_metadata: dict[str, object] | None = None
    if measurement_metadata_path is not None:
        measurement_metadata = json.loads(measurement_metadata_path.read_text(encoding="utf-8"))
        metadata["measurement"] = {
            key: measurement_metadata.get(key)
            for key in (
                "profile_name",
                "date_time",
                "eq_settings",
                "input_device",
                "output_device",
                "sample_rate",
                "volume_note",
                "mic_position_note",
                "mic_position_id",
                "spatial_session_id",
                "spatial_position",
                "measurement_mode",
                "session_purpose",
                "analysis_schema_version",
                "clock_correction",
                "channel_selection",
                "channel_routing_verified",
                "device_profile_id",
                "device_profile_schema",
                "target_profile",
                "microphone_profile_id",
                "microphone_calibrated",
                "processing_settings",
                "equipment",
                "delay_settings",
                "source_signal_id",
                "inverse_filter_id",
                "system_profile_hash",
                "microphone_profile_hash",
                "ess_parameters",
            )
        }
        if analysis_method == "ess":
            metadata["measurement"]["source_signal_id"] = file_content_id(source_path, "ess")
            if inverse_filter_path is not None:
                metadata["measurement"]["inverse_filter_id"] = file_content_id(
                    inverse_filter_path,
                    "ess_inverse",
                )
    if analysis_method == "psd-ratio":
        quality = assess_measurement(recorded, recorded_rate)
        metadata["quality"] = quality
        raw_spectrum = third_octave_levels(recorded, recorded_rate)
        raw_response = psd_ratio_estimate(source, recorded, recorded_rate)
        alignment = align_response_to_target(raw_response, target_db)
        metadata.update(
            {
                "raw_spectrum_db": raw_spectrum.tolist(),
                "raw_response_db": raw_response.tolist(),
                "aligned_response_db": alignment.aligned_response_db.tolist(),
                "response_alignment": alignment.diagnostics(),
                "description": "Welch power-spectrum ratio estimate; not a strict transfer function.",
                "phase_reliable": False,
                "group_delay_reliable": False,
            }
        )
        return alignment.aligned_response_db, metadata, None, raw_response

    if inverse_filter_path is None:
        raise ValueError("--inverse-filter is required for ESS analysis")
    inverse_filter, inverse_rate = read_wav(inverse_filter_path)
    if inverse_rate != recorded_rate:
        raise ValueError("ESS inverse filter must have the same sample rate as the recording")
    timing_diagnostics: dict[str, object]
    analysis_recording = recorded
    active_ess_start = 0
    expected_ess_samples = len(to_mono(source))
    available_ess_samples = min(len(to_mono(recorded)), expected_ess_samples)
    noise_segment: np.ndarray | None = None
    if clock_correction_mode == "required":
        if timing_marker_path is None or measurement_metadata_path is None:
            raise ValueError("ESS clock correction requires timing marker and measurement metadata")
        timing_marker, marker_rate = read_wav(timing_marker_path)
        if marker_rate != recorded_rate:
            raise ValueError("Timing marker must have the same sample rate as the recording")
        assert measurement_metadata is not None
        layout_data = measurement_metadata.get("timing_layout")
        if not isinstance(layout_data, dict):
            raise ValueError("Measurement metadata does not contain timing_layout")
        layout = TimingLayout.from_dict(layout_data)
        correction = correct_clock_drift(recorded, timing_marker, layout)
        analysis_recording = correction.corrected_recording
        active_ess_start = correction.ess_pre_context_samples
        active_ess_end = active_ess_start + expected_ess_samples
        available_ess_samples = max(
            0,
            min(active_ess_end, correction.valid_end_sample)
            - max(active_ess_start, correction.valid_start_sample),
        )
        pre_marker = to_mono(recorded)[: correction.start_marker_recorded_sample]
        trim = int(round(0.05 * recorded_rate))
        noise_segment = pre_marker[trim:-trim] if len(pre_marker) > 2 * trim else pre_marker
        timing_diagnostics = {
            "mode": "required",
            "clock_drift_compensated": True,
            "relative_ir_timing_reliable": True,
            "absolute_latency_reliable": False,
            "cross_run_delay_reliable": False,
            **correction.diagnostics(),
        }
    else:
        active_ess_start = locate_active_ess(recorded, source)
        available_ess_samples = min(
            expected_ess_samples,
            max(0, len(to_mono(recorded)) - active_ess_start),
        )
        timing_diagnostics = {
            "mode": "off",
            "clock_drift_compensated": False,
            "relative_ir_timing_reliable": False,
            "absolute_latency_reliable": False,
            "cross_run_delay_reliable": False,
        }
    active_ess_end = min(len(to_mono(analysis_recording)), active_ess_start + expected_ess_samples)
    active_ess_recording = to_mono(analysis_recording)[active_ess_start:active_ess_end]
    allowed_shortfall = int(round(0.02 * recorded_rate))
    active_ess_complete = available_ess_samples >= expected_ess_samples - allowed_shortfall
    quality = assess_measurement(
        recorded,
        recorded_rate,
        signal_segment=active_ess_recording,
        noise_segment=noise_segment,
    )
    if not active_ess_complete:
        failures = list(quality["hard_failures"])
        failures.append("incomplete_active_ess")
        quality["hard_failures"] = sorted(set(failures))
        quality["accepted"] = False
    metadata["quality"] = quality
    raw_spectrum = third_octave_levels(analysis_recording, recorded_rate)
    result = deconvolve_ess(
        source,
        analysis_recording,
        inverse_filter,
        recorded_rate,
        sweep_start_hz=device_profile.sweep_start_hz if device_profile else None,
        sweep_end_hz=device_profile.sweep_end_hz if device_profile else None,
    )
    if result.early_h2_h3_ratio_percent is not None:
        quality["early_h2_ratio_percent"] = list(result.early_h2_ratio_percent or ())
        quality["early_h3_ratio_percent"] = list(result.early_h3_ratio_percent or ())
        quality["early_h2_h3_ratio_percent"] = list(result.early_h2_h3_ratio_percent)
        quality["distortion_metric_method"] = "ess_early_h2_h3"
        quality["distortion_metric_status"] = "experimental"
        quality["harmonic_orders_included"] = [2, 3]
        quality["noise_compensated"] = False
        quality["absolute_thd_claim_allowed"] = False
        quality["distortion_diagnostics"] = result.distortion_diagnostics
        warning_limit = (
            device_profile.quality_limits.get("experimental_early_h2_h3_warning_percent", 10.0)
            if device_profile
            else 10.0
        )
        finite_distortion = [
            value for value in result.early_h2_h3_ratio_percent if value is not None
        ]
        if finite_distortion and max(finite_distortion) > warning_limit:
            quality["warnings"] = sorted(
                set(quality["warnings"]) | {"experimental_early_h2_h3_high"}
            )
    raw_response = result.response_db.copy()
    smoothed_response = result.smoothed_response_db.copy()
    calibration_file = device_profile.calibration_file if device_profile else None
    if calibration_file is not None:
        if not calibration_file.exists():
            raise ValueError(f"Microphone calibration file not found: {calibration_file}")
        raw_response += calibration_correction_db(
            calibration_file,
            np.asarray(ANALYSIS_FREQUENCIES_HZ),
        )
        smoothed_response += calibration_correction_db(
            calibration_file,
            result.smoothed_frequencies_hz,
        )
    alignment = align_response_to_target(raw_response, target_db)
    metadata.update(
        {
            "description": "ESS inverse-filter deconvolution with a gated linear impulse response.",
            "raw_spectrum_db": raw_spectrum.tolist(),
            "raw_response_db": raw_response.tolist(),
            "aligned_response_db": alignment.aligned_response_db.tolist(),
            "response_alignment": alignment.diagnostics(),
            "alignment_delay_s": result.alignment_delay_s,
            "reference_peak_index": result.reference_peak_index,
            "recorded_peak_index": result.recorded_peak_index,
            "impulse_peak_offset_samples": result.peak_offset_samples,
            "smoothed_response": {
                "fractional_octave": 6,
                "frequencies_hz": result.smoothed_frequencies_hz.tolist(),
                "raw_response_db": smoothed_response.tolist(),
                "aligned_response_db": (
                    smoothed_response - alignment.offset_db
                ).tolist(),
            },
            "timing": timing_diagnostics,
            "clock_drift_compensated": timing_diagnostics["clock_drift_compensated"],
            "timing_markers_valid": timing_diagnostics["clock_drift_compensated"],
            "source_signal_id": file_content_id(source_path, "ess"),
            "inverse_filter_id": file_content_id(inverse_filter_path, "ess_inverse"),
            "ess_parameters": (
                measurement_metadata.get("ess_parameters")
                if measurement_metadata
                else {
                    "duration_s": expected_ess_samples / recorded_rate,
                    "start_hz": device_profile.sweep_start_hz if device_profile else 1.0,
                    "end_hz": device_profile.sweep_end_hz if device_profile else recorded_rate / 2.0 - 1.0,
                }
            ),
            "active_ess_start_sample": active_ess_start,
            "active_ess_end_sample": active_ess_start + expected_ess_samples,
            "active_ess_duration_s": available_ess_samples / recorded_rate,
            "active_ess_expected_samples": expected_ess_samples,
            "active_ess_available_samples": available_ess_samples,
            "active_ess_complete": active_ess_complete,
            "dropout_analysis_scope": "active_ess_only",
            "phase_reliable": False,
            "group_delay_reliable": False,
        }
    )
    return (
        alignment.aligned_response_db,
        metadata,
        result.impulse_response,
        raw_response,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze automotive audio recordings.")
    parser.add_argument("--profiles", nargs="*", default=[], help="name=archive.zip")
    parser.add_argument("--source-signal", type=Path)
    parser.add_argument("--recorded-signal", type=Path)
    parser.add_argument("--analysis-method", choices=("ess", "psd-ratio"), default="ess")
    parser.add_argument("--inverse-filter", type=Path)
    parser.add_argument("--clock-correction", choices=("required", "off"), default="required")
    parser.add_argument("--timing-marker", type=Path)
    parser.add_argument("--measurement-metadata", type=Path)
    parser.add_argument("--device-profile", type=Path)
    parser.add_argument("--profile-name", default="measurement")
    parser.add_argument("--eq")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report_profile = load_device_profile(args.device_profile) if args.device_profile else None
    report_target_name = report_profile.target_name if report_profile else "warm_driver"
    report_target = target_curve_db(report_target_name, ANALYSIS_FREQUENCIES_HZ)
    report_device_name = report_profile.name if report_profile else "Аудиосистема"
    out = timestamped_out(args.out)
    profiles: dict[str, np.ndarray] = {}
    raw_profiles: dict[str, np.ndarray] = {}
    profile_alignments: dict[str, dict[str, object]] = {}
    quality: dict[str, object] = {}
    continuous_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    impulse_response: np.ndarray | None = None
    for item in args.profiles:
        name, raw_path = item.split("=", maxsplit=1)
        profiles[name], quality[name], raw_profiles[name], profile_alignments[name] = analyze_zip(
            Path(raw_path)
        )
    if args.source_signal and args.recorded_signal:
        profiles[args.profile_name], pair_meta, impulse_response, raw_profiles[args.profile_name] = analyze_pair(
            args.source_signal,
            args.recorded_signal,
            args.analysis_method,
            args.inverse_filter,
            args.clock_correction,
            args.timing_marker,
            args.measurement_metadata,
            args.device_profile,
        )
        quality[args.profile_name] = pair_meta
        profile_alignments[args.profile_name] = pair_meta["response_alignment"]
        smoothed = pair_meta.get("smoothed_response")
        if isinstance(smoothed, dict):
            continuous_profiles[args.profile_name] = (
                np.asarray(smoothed["frequencies_hz"], dtype=np.float64),
                np.asarray(smoothed["aligned_response_db"], dtype=np.float64),
            )
        artifact_name = "ess_response.json" if args.analysis_method == "ess" else "psd_ratio_response.json"
        (out / artifact_name).write_text(
            json.dumps(pair_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if args.analysis_method == "ess" and "timing" in pair_meta:
            (out / "timing_diagnostics.json").write_text(
                json.dumps(pair_meta["timing"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if impulse_response is not None:
            _, recorded_rate = read_wav(args.recorded_signal)
            sf.write(out / "impulse_response.wav", impulse_response, recorded_rate, subtype="FLOAT")
    if not profiles:
        raise SystemExit("Provide --profiles or --source-signal with --recorded-signal")
    default_name = "default" if "default" in profiles else next(iter(profiles))
    table = build_table(profiles, default_name, raw_profiles, report_target)
    table.to_csv(out / "frequency_table.csv", index=False)
    (out / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                "frequency_grid": "ISO 266 base-ten third-octave series",
                "nominal_frequencies_hz": NOMINAL_THIRD_OCTAVE_FREQUENCIES_HZ,
                "exact_center_frequencies_hz": EXACT_THIRD_OCTAVE_FREQUENCIES_HZ,
                "response_alignments": profile_alignments,
                "target_profile": report_target_name,
                "legacy_compatibility": "Scores without this schema version are not directly comparable.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_profiles(
        out,
        profiles,
        continuous_profiles,
        report_target,
        report_target_name,
        report_device_name,
    )
    report_method = args.analysis_method if args.source_signal and args.recorded_signal else "recording-spectrum"
    write_report(
        out,
        table,
        profiles,
        quality,
        profile_alignments,
        default_name,
        report_method,
        report_target,
        report_device_name,
    )
    print(out)


if __name__ == "__main__":
    main()
