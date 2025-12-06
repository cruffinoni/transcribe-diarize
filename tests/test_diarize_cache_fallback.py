import json
from pathlib import Path

import pytest

from asr_cli import cli
from asr_cli import diarization


def test_diarize_uses_cache_params_and_token(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")  # minimal placeholder
    cache_path = tmp_path / "input.tiny.abc.whisper.json"

    cache_payload = {
        "metadata": {
            "cached_audio": str(audio_path),
            "input_file": str(audio_path),
            "whisper_model": "tiny",
            "whisper_device": "cpu",
            "num_speakers_param": 2,
            "hf_token": "CACHEDTOKEN",
        },
        "result": {"segments": []},
    }
    cache_path.write_text(json.dumps(cache_payload))

    calls = {}

    class DummyPipeline:
        def to(self, device):
            calls["send_device"] = device

    def fake_load_pyannote(token):
        calls["hf_token"] = token
        return DummyPipeline()

    def fake_run_diarization(pipeline, wav_path, diar_kwargs):
        calls["diar_kwargs"] = diar_kwargs
        calls["wav_path"] = str(wav_path)
        return None

    monkeypatch.setattr(diarization, "load_pyannote", fake_load_pyannote)
    monkeypatch.setattr(diarization, "send_pipeline_to_device", lambda pipeline, device: None)
    monkeypatch.setattr(diarization, "run_diarization", fake_run_diarization)

    out_dir = tmp_path / "out"
    cli._run_diarization_only(
        cache_path,
        audio_path_override=None,
        diar_device="cpu",
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        hf_token=None,
        out_dir=out_dir,
        formats_raw="json",
        interactive_speakers=False,
    )

    assert calls["hf_token"] == "CACHEDTOKEN"
    assert calls["diar_kwargs"]["num_speakers"] == 2

    out_json = out_dir / "audio.json"
    with out_json.open() as jf:
        data = json.load(jf)
    assert data["metadata"]["num_speakers_param"] == 2
    assert "hf_token" not in data["metadata"]
