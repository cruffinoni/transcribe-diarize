import contextlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:  # torch >= 2.6 adds weights_only default; allowlist TorchVersion for pyannote checkpoints
    from torch.serialization import add_safe_globals as _add_safe_globals
    from torch.serialization import safe_globals as _safe_globals
except ImportError:  # pragma: no cover - older torch
    _add_safe_globals = None
    _safe_globals = None

from .devices import resolve_device as _resolve_device
from .utils import fatal

_PYANNOTE_IMPORT_ERROR: Optional[Exception] = None


def resolve_device(requested: str) -> str:
    return _resolve_device(requested, label="Diarization")


def _import_pyannote_pipeline():
    """
    Import pyannote Pipeline lazily so commands that don't need diarization
    don't crash if pyannote/torchaudio is unavailable. Cache the first error
    to avoid repeated import attempts.
    """
    global _PYANNOTE_IMPORT_ERROR
    if _PYANNOTE_IMPORT_ERROR is not None:
        raise _PYANNOTE_IMPORT_ERROR
    try:
        from pyannote.audio import Pipeline

        return Pipeline
    except Exception as exc:  # pragma: no cover - defensive: environment-specific import errors
        _PYANNOTE_IMPORT_ERROR = exc
        logging.warning(
            "pyannote.audio import failed; diarization unavailable (%s). "
            "Use diarization commands only on environments with a compatible torchaudio/pyannote install.",
            exc,
        )
        raise


def _import_pyannote_task_extras() -> Tuple[Any, Any, Any]:
    try:
        from pyannote.audio.core.task import (
            Specifications as _PyannoteSpecifications,
            Problem as _PyannoteProblem,
            Resolution as _PyannoteResolution,
        )

        return _PyannoteSpecifications, _PyannoteProblem, _PyannoteResolution
    except Exception:  # pragma: no cover - optional, not critical
        return None, None, None


def probe_pyannote_available() -> bool:
    """
    Best-effort probe to see if pyannote can be imported. Logs a warning instead
    of crashing so help/Whisper-only commands still work on hosts without a
    compatible torchaudio/pyannote build.
    """
    try:
        _import_pyannote_pipeline()
        return True
    except Exception:
        return False


def load_pyannote(hf_token: Optional[str]) -> Any:
    """
    Load the pyannote diarization pipeline. Requires that the user accepted the
    HF terms for pyannote/speaker-diarization-3.1.
    """
    logging.info("Loading pyannote speaker-diarization-3.1 pipeline...")
    try:
        Pipeline = _import_pyannote_pipeline()
    except Exception as exc:
        fatal(
            "pyannote is unavailable in this environment; diarization commands cannot run. "
            "Install a compatible torchaudio/pyannote combo or run Whisper-only commands. "
            f"Details: {exc}"
        )

    extras = [torch.torch_version.TorchVersion]
    _PyannoteSpecifications, _PyannoteProblem, _PyannoteResolution = _import_pyannote_task_extras()
    if _PyannoteSpecifications:
        extras.append(_PyannoteSpecifications)
    if _PyannoteProblem:
        extras.append(_PyannoteProblem)
    if _PyannoteResolution:
        extras.append(_PyannoteResolution)
    if _add_safe_globals:
        try:
            _add_safe_globals(extras)
        except Exception:
            # If it fails, continue; torch.load will raise its own error if needed.
            pass
    ctx = _safe_globals(extras) if _safe_globals else contextlib.nullcontext()
    try:
        with ctx:
            if hf_token:
                try:
                    # Newer huggingface_hub uses 'token'
                    return Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=hf_token,
                    )
                except TypeError:
                    # Older versions expect 'use_auth_token'
                    return Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token,
                    )
            return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as exc:
        fatal(
            "Failed to load pyannote pipeline. Ensure your Hugging Face token is valid "
            "and model terms are accepted. "
            f"Details: {exc}"
        )


def send_pipeline_to_device(pipeline: Any, device: str) -> None:
    logging.info("Moving pyannote pipeline to '%s'...", device)
    try:
        pipeline.to(torch.device(device))
    except Exception as exc:
        fatal(f"Failed to move pyannote pipeline to '{device}': {exc}")


def run_diarization(pipeline: Any, wav_path: Path, diar_kwargs: Dict):
    diar_kwargs = diar_kwargs or {}
    logging.info("Running diarization with kwargs: %s", diar_kwargs)
    try:
        return pipeline(str(wav_path), **diar_kwargs)
    except Exception as exc:
        fatal(f"Diarization failed: {exc}")
