import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest


def _generate_tone(path: Path, seconds: float = 1.5, sr: int = 16000) -> None:
    """Write a short mono sine wave WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(int(seconds * sr)):
            value = int(0.1 * 32767 * math.sin(2 * math.pi * 440 * i / sr))
            wf.writeframes(struct.pack("<h", value))


def _hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()


@pytest.mark.skipif(not _hf_token(), reason="HF token required for pyannote model")
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not available on PATH")
def test_cli_smoke(tmp_path):
    input_wav = tmp_path / "input.wav"
    _generate_tone(input_wav)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "asr_cli.cli",
        str(input_wav),
        "--model",
        "tiny",
        "--num-speakers",
        "1",
        "--hf-token",
        _hf_token(),
        "--format",
        "json",
        "--out",
        str(out_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"CLI failed: {result.returncode}\nstdout:{result.stdout}\nstderr:{result.stderr}")

    expected_json = out_dir / "input.json"
    assert expected_json.exists(), "Expected JSON output was not created"
