from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config import ANALYSIS_FREQUENCIES_HZ
from device_profile import load_device_profile
from spatial_analysis import aggregate_spatial_payloads, load_spatial_payload
from spatial_positions import configure_utf8_console, spatial_sequence_text
from spatial_reporting import write_spatial_outputs
from targets import target_curve_db


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Aggregate six fixed-position ESS results.")
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/outputs/spatial"))
    parser.add_argument("--device-profile", type=Path)
    args = parser.parse_args()

    print(spatial_sequence_text())
    payloads = [load_spatial_payload(path) for path in args.results]
    profile = load_device_profile(args.device_profile) if args.device_profile else None
    target = target_curve_db(profile.target_name, ANALYSIS_FREQUENCIES_HZ) if profile else None
    result = aggregate_spatial_payloads(
        payloads,
        target_db=target,
        target_name=profile.target_name if profile else "warm_driver",
    )
    out = args.out / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    write_spatial_outputs(out, result, profile)
    print(out)


if __name__ == "__main__":
    main()
