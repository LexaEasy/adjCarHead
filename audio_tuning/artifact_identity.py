from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


AUDIO_SUFFIXES = {".aif", ".aiff", ".flac", ".wav"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with sf.SoundFile(path) as source:
        header = {
            "schema": "decoded_audio_v1",
            "sample_rate": source.samplerate,
            "channels": source.channels,
            "frames": source.frames,
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
        for block in source.blocks(blocksize=65_536, dtype="float32", always_2d=True):
            digest.update(np.ascontiguousarray(block, dtype="<f4").tobytes())
    return digest.hexdigest()


def file_content_id(path: Path, kind: str) -> str:
    digest = audio_content_sha256(path) if path.suffix.lower() in AUDIO_SUFFIXES else file_sha256(path)
    return f"{kind}:sha256:{digest}"


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
