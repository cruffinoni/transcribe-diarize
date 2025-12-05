import json

from asr_cli import whisper_infer


class _DummyModel:
    def __init__(self, device="cpu"):
        self.calls = []
        self.device = device

    def transcribe(self, path, **opts):
        self.calls.append((path, opts))
        return {"segments": []}


def test_run_whisper_enables_progress_bar(tmp_path):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"dummy")

    model = _DummyModel()
    whisper_infer.run_whisper(model, wav_path, language="fr")

    assert model.calls, "transcribe was not invoked"
    called_path, opts = model.calls[0]
    assert called_path == str(wav_path)
    assert opts["verbose"] is False
    assert opts["beam_size"] == 1
    assert opts["language"] == "fr"
    assert opts["fp16"] is False


def test_export_whisper_cache_writes_files(tmp_path):
    wav_path = tmp_path / "audio.wav"
    wav_bytes = b"audio-bytes-here"
    wav_path.write_bytes(wav_bytes)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    wres = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
    json_path, audio_path, fingerprint = whisper_infer.export_whisper_cache(
        whisper_result=wres,
        wav_path=wav_path,
        out_dir=out_dir,
        base_name="clip",
        model="medium",
        device="cpu",
        language="en",
    )

    assert json_path.exists()
    assert audio_path.exists()
    assert audio_path.read_bytes() == wav_bytes

    import hashlib
    expected_fp = hashlib.sha256(wav_bytes).hexdigest()
    assert fingerprint == expected_fp

    data = json.loads(json_path.read_text())
    assert data["metadata"]["audio_fingerprint_sha256"] == expected_fp
    assert data["metadata"]["whisper_model"] == "medium"
    assert data["metadata"]["language"] == "en"
