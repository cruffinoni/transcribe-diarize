# transcribe_diarize

CLI tool for end-to-end transcription and speaker diarization using OpenAI Whisper and `pyannote/speaker-diarization-3.1`. It extracts mono 16k WAV audio with `ffmpeg`, runs Whisper, diarizes speakers, and writes SRT/TXT/JSON/RTTM/dialog outputs.

## Prerequisites
- Python 3.9+
- `ffmpeg` available on PATH (e.g., `brew install ffmpeg` or `apt install ffmpeg`)
- Hugging Face token with accepted terms for `pyannote/speaker-diarization-3.1`
- `rich` is installed with the package and is required for progress display

## Installation
```bash
pip install -e .
```

If you already have a virtual environment with the dependencies, activate it instead (example using the existing `~/venvs/asr-mps` env used to freeze `requirements.txt`):
```bash
source ~/venvs/asr-mps/bin/activate
```

To recreate the same environment from scratch:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

While running, the CLI shows progress for each step (audio extraction, Whisper load/transcription, diarization, exports) with a spinner when `rich` is available, otherwise via INFO logs.

## Quickstart
macOS / Apple Silicon (auto uses MPS when available):
```bash
transcribe_diarize input.mp4 --model medium --hf-token "$HF_TOKEN" --format srt,json,dialog
```

Linux / WSL2 with CUDA:
```bash
transcribe_diarize input.mp4 --model large-v2 --whisper-device auto --diar-device auto \
  --num-speakers 2 --hf-token "$HF_TOKEN" --format srt,json,dialog
```

CPU-only:
```bash
transcribe_diarize input.wav --model tiny --whisper-device cpu --diar-device cpu
```

Interactive speaker naming:
```bash
transcribe_diarize input.mp4 --hf-token "$HF_TOKEN" --interactive-speakers
```

Outputs are written next to the input by default; override with `--out DIR`. Choose formats with `--format srt,txt,json,rttm,dialog`.

## Devices
- `--whisper-device/--diar-device`: `auto` (default) prefers CUDA, then MPS, else CPU.
- Requests for unavailable devices automatically fall back to CPU with a warning.
- Whisper retries on CPU if the target device raises `NotImplementedError` (e.g., some MPS sparse ops).

## Notes
- Default Whisper beam size is 1 for speed; adjust in code if needed.
- JSON metadata includes model names/devices, speaker params, language, input path, and approximate duration.
