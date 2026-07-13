from __future__ import annotations

import argparse
import json
from pathlib import Path

from device_profile import load_device_profile
from dsp_characterization import characterize_dsp
from validation_analysis import (
    analyze_level_linearity,
    analyze_repeatability,
    load_ess_result,
)


MANIFEST_SCHEMA_VERSION = "quick_validation_manifest_v1"


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one-time validation series made with quick mode.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SystemExit(f"Expected manifest schema {MANIFEST_SCHEMA_VERSION}")
    base = manifest_path.parent
    profile = load_device_profile(_resolve(base, data.get("device_profile")))
    kind = data.get("kind")
    if kind == "repeatability":
        paths = data.get("results")
        if not isinstance(paths, list):
            raise SystemExit("Repeatability manifest requires results")
        result = analyze_repeatability(
            [load_ess_result(_resolve(base, path)) for path in paths],
            profile,
        )
    elif kind == "level_linearity":
        items = data.get("measurements")
        if not isinstance(items, list):
            raise SystemExit("Level manifest requires measurements")
        result = analyze_level_linearity(
            [
                (float(item["level_rank"]), load_ess_result(_resolve(base, item["result"])))
                for item in items
                if isinstance(item, dict)
            ],
            profile,
        )
    elif kind == "dsp_characterization":
        controls = data.get("controls")
        if not isinstance(controls, list):
            raise SystemExit("DSP manifest requires controls")
        baseline = load_ess_result(_resolve(base, data.get("baseline_result")))
        variants = [
            (
                str(item["id"]),
                float(item["plus_delta_db"]),
                load_ess_result(_resolve(base, item["plus_result"])),
                float(item["minus_delta_db"]),
                load_ess_result(_resolve(base, item["minus_result"])),
            )
            for item in controls
            if isinstance(item, dict)
        ]
        result = characterize_dsp(baseline, variants, profile)
    else:
        raise SystemExit(f"Unknown validation kind: {kind}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
