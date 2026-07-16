from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from artifact_identity import file_content_id, file_sha256


class ArtifactIdentityTest(unittest.TestCase):
    def test_audio_id_ignores_container_only_bytes(self) -> None:
        samples = np.linspace(-0.5, 0.5, 4_800, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.wav"
            second = Path(directory) / "second.wav"
            sf.write(first, samples, 48_000, subtype="FLOAT")
            sf.write(second, samples, 48_000, subtype="FLOAT")
            second.write_bytes(second.read_bytes() + b"container-only-data")

            self.assertNotEqual(file_sha256(first), file_sha256(second))
            self.assertEqual(file_content_id(first, "ess"), file_content_id(second, "ess"))


if __name__ == "__main__":
    unittest.main()
