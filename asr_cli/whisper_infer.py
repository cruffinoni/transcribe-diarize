import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import whisper

from .devices import resolve_device as _resolve_device
from .utils import fatal


def resolve_device(requested: str) -> str:
    return _resolve_device(requested, label="Whisper")


def load_whisper(model_name: str, device: str):
    """
    Load Whisper on the requested device, with auto-fallback to CPU if MPS/CUDA
    raises NotImplementedError (e.g., sparse ops on some Apple Silicon chips).
    """
    # Encourage MPS fallback when possible
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logging.info("Loading Whisper model '%s' on '%s'...", model_name, device)
    try:
        return whisper.load_model(model_name, device=device)
    except NotImplementedError as exc:
        logging.warning(
            "Whisper failed to load on device=%s (%s). Falling back to CPU.",
            device,
            exc,
        )
    except Exception as exc:
        fatal(f"Failed to load Whisper model '{model_name}' on {device}: {exc}")

    try:
        return whisper.load_model(model_name, device="cpu")
    except Exception as exc:  # pragma: no cover - fatal path
        fatal(f"Failed to load Whisper model on CPU: {exc}")


def run_whisper(model, wav_path: Path, language: Optional[str]) -> Dict:
    device = str(getattr(model, "device", "cpu"))
    opts = {
        "beam_size": 1,  # lower beam size for speed
        "verbose": False,  # enable tqdm progress bar instead of silent default
    }
    if device.startswith("cpu"):
        # Avoid Whisper warning: "FP16 is not supported on CPU; using FP32 instead"
        opts["fp16"] = False
    if language:
        opts["language"] = language
    logging.info("Running Whisper transcription...")
    return model.transcribe(str(wav_path), **opts)


def _fingerprint_file(path: Path) -> str:
    """
    Compute a SHA256 fingerprint of the audio file so we can cache and reuse
    Whisper results safely.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_tag(value: str) -> str:
    """Make a string safe to use in filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def export_whisper_cache(
    whisper_result: Dict,
    wav_path: Path,
    out_dir: Path,
    base_name: str,
    model: str,
    device: str,
    language: Optional[str],
    source_path: Optional[Path] = None,
) -> Tuple[Path, Path, str]:
    """
    Persist Whisper output and the normalized audio alongside a fingerprint.

    Returns: (json_path, audio_path, fingerprint)
    """
    fingerprint = _fingerprint_file(wav_path)
    model_tag = _safe_tag(model)
    fp_tag = fingerprint[:12]

    json_path = out_dir / f"{base_name}.{model_tag}.{fp_tag}.whisper.json"
    audio_path = out_dir / f"{base_name}.{model_tag}.{fp_tag}.audio.wav"

    shutil.copy2(wav_path, audio_path)

    payload = {
        "metadata": {
            "whisper_model": model,
            "whisper_device": str(device),
            "language": language,
            "audio_fingerprint_sha256": fingerprint,
            "cached_audio": str(audio_path),
        },
        "result": whisper_result,
    }
    if source_path:
        payload["metadata"]["input_file"] = str(source_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info(
        "Cached Whisper output and audio: %s , %s (fingerprint=%s)",
        json_path.name,
        audio_path.name,
        fp_tag,
    )

    return json_path, audio_path, fingerprint
