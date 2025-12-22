import logging
import subprocess
import wave
from pathlib import Path

from .utils import fatal, which


def extract_wav(in_path: Path, out_path: Path) -> None:
    """
    Use ffmpeg to extract mono 16kHz WAV audio from the input media file.
    """
    ffmpeg = which("ffmpeg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(in_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_path),
    ]
    logging.info("Running ffmpeg to extract mono 16k WAV...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        fatal(
            f"ffmpeg failed (exit {exc.returncode}). Check input file and ffmpeg installation."
        )


def wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except wave.Error as exc:
        logging.warning("Failed to read WAV duration from %s: %s", path, exc)
        return 0.0
