import logging
import os
from pathlib import Path
from typing import Dict, Optional

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
    opts = {"beam_size": 1}  # lower beam size for speed
    if language:
        opts["language"] = language
    logging.info("Running Whisper transcription...")
    return model.transcribe(str(wav_path), **opts)
