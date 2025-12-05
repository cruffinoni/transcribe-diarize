import contextlib
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from pyannote.audio import Pipeline

try:  # torch >= 2.6 adds weights_only default; allowlist TorchVersion for pyannote checkpoints
    from torch.serialization import add_safe_globals as _add_safe_globals
    from torch.serialization import safe_globals as _safe_globals
except ImportError:  # pragma: no cover - older torch
    _add_safe_globals = None
    _safe_globals = None

try:
    from pyannote.audio.core.task import Specifications as _PyannoteSpecifications
    from pyannote.audio.core.task import Problem as _PyannoteProblem
    from pyannote.audio.core.task import Resolution as _PyannoteResolution
except Exception:  # pragma: no cover - defensive: if module path changes
    _PyannoteSpecifications = None
    _PyannoteProblem = None
    _PyannoteResolution = None

from .devices import resolve_device as _resolve_device
from .utils import fatal


def resolve_device(requested: str) -> str:
    return _resolve_device(requested, label="Diarization")


def load_pyannote(hf_token: Optional[str]) -> Pipeline:
    """
    Load the pyannote diarization pipeline. Requires that the user accepted the
    HF terms for pyannote/speaker-diarization-3.1.
    """
    logging.info("Loading pyannote speaker-diarization-3.1 pipeline...")
    extras = [torch.torch_version.TorchVersion]
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


def send_pipeline_to_device(pipeline: Pipeline, device: str) -> None:
    logging.info("Moving pyannote pipeline to '%s'...", device)
    try:
        pipeline.to(torch.device(device))
    except Exception as exc:
        fatal(f"Failed to move pyannote pipeline to '{device}': {exc}")


def run_diarization(pipeline: Pipeline, wav_path: Path, diar_kwargs: Dict):
    diar_kwargs = diar_kwargs or {}
    logging.info("Running diarization with kwargs: %s", diar_kwargs)
    try:
        return pipeline(str(wav_path), **diar_kwargs)
    except Exception as exc:
        fatal(f"Diarization failed: {exc}")
