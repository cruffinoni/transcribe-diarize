import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from pyannote.audio import Pipeline

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
    try:
        if hf_token:
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
