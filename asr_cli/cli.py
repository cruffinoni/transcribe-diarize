import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
from dotenv import load_dotenv

from . import audio, diarization, export, speakers, whisper_infer
from .utils import fatal, setup_logging, step_status


VALID_FORMATS: Set[str] = {"srt", "txt", "json", "rttm", "dialog"}

def parse_formats(fmt: str) -> Set[str]:
    formats = {f.strip().lower() for f in fmt.split(",") if f.strip()}
    unknown = formats - VALID_FORMATS
    if unknown:
        fatal(
            f"Unknown format(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(sorted(VALID_FORMATS))}."
        )
    return formats


def _duration_from_segments(segments: List[Dict]) -> float:
    return float(segments[-1]["end"]) if segments else 0.0


def _build_diar_kwargs(
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Dict:
    if num_speakers is not None:
        if min_speakers is not None or max_speakers is not None:
            fatal("Use either --num-speakers or min/max, not both.")
        if num_speakers < 1:
            fatal("--num-speakers must be >= 1.")
        return {"num_speakers": int(num_speakers)}

    kwargs: Dict = {}
    if min_speakers is not None:
        if min_speakers < 1:
            fatal("--min-speakers must be >= 1.")
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        if max_speakers < 1:
            fatal("--max-speakers must be >= 1.")
        kwargs["max_speakers"] = int(max_speakers)
    if (
        "min_speakers" in kwargs
        and "max_speakers" in kwargs
        and kwargs["min_speakers"] > kwargs["max_speakers"]
    ):
        fatal("--min-speakers cannot be greater than --max-speakers.")
    return kwargs


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token.strip()
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if env_token:
        logging.info("Using HF token from environment.")
    return env_token or None


def _ensure_out_dir(out_dir: Optional[Path], fallback: Path) -> Path:
    resolved = (out_dir or fallback).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_cache_dir(cache_root: Optional[Path], model: str, fingerprint: str) -> Path:
    root = (cache_root or Path.home() / ".cache" / "asr-mps").expanduser().resolve()
    model_tag = whisper_infer.safe_tag(model)
    target = root / model_tag / fingerprint
    target.mkdir(parents=True, exist_ok=True)
    return target


def _attach_speakers_to_segments(whisper_segments, diarization_result) -> List[Dict]:
    enriched = []
    for seg in whisper_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        speaker_label = speakers.best_speaker(diarization_result, start, end)
        enriched.append(
            {
                "id": seg.get("id"),
                "start": start,
                "end": end,
                "text": (seg.get("text") or "").strip(),
                "speaker": speaker_label,
            }
        )
    return enriched


def _export_outputs(
    enriched_segments: List[Dict],
    diarization_result,
    formats: Set[str],
    out_dir: Path,
    base_name: str,
    metadata: Dict,
) -> None:
    speaker_labels = speakers.collect_speakers(enriched_segments) if enriched_segments else []

    if "rttm" in formats:
        if diarization_result is None:
            logging.warning("Skipping RTTM export; diarization result not provided.")
        else:
            export.export_rttm(diarization_result, out_dir / f"{base_name}.rttm")
    if "srt" in formats:
        export.export_srt(enriched_segments, out_dir / f"{base_name}.srt")
    if "txt" in formats:
        export.export_txt(enriched_segments, out_dir / f"{base_name}.txt")
    if "dialog" in formats:
        export.export_dialog(enriched_segments, out_dir / f"{base_name}.dialog.txt")
    if "json" in formats:
        export.export_json(enriched_segments, out_dir / f"{base_name}.json", metadata, speaker_labels)


def _load_whisper_cache(cache_path: Path) -> Tuple[Dict, Dict]:
    cache_path = cache_path.expanduser().resolve()
    if not cache_path.exists():
        fatal(f"Whisper cache not found: {cache_path}")
    try:
        data = json.loads(cache_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        fatal(f"Failed to read Whisper cache: {exc}")
    if "result" not in data:
        fatal("Invalid Whisper cache: missing 'result' key.")
    return data["result"], data.get("metadata", {})


def _resolve_cached_audio(
    audio_override: Optional[Path],
    metadata: Dict,
    cache_path: Path,
) -> Path:
    candidates = []
    if audio_override:
        candidates.append(audio_override.expanduser())
    meta_audio = metadata.get("cached_audio")
    if meta_audio:
        candidates.append(Path(meta_audio).expanduser())
    name = cache_path.name
    if name.endswith(".whisper.json"):
        candidates.append(cache_path.with_name(name.replace(".whisper.json", ".audio.wav")))

    for cand in candidates:
        if cand and cand.exists():
            return cand
    fatal(
        "Could not locate cached audio. Provide --audio or ensure the cache metadata has 'cached_audio'."
    )


def _base_name_from_metadata(
    cache_path: Path,
    metadata: Dict,
    audio_path: Optional[Path],
    input_path: Optional[Path] = None,
) -> str:
    if input_path:
        return input_path.stem
    if metadata.get("input_file"):
        return Path(metadata["input_file"]).stem

    def _trim_name(name: str, suffix: str, drop_parts: int) -> str:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        parts = name.split(".")
        if len(parts) > drop_parts:
            return ".".join(parts[:-drop_parts])
        return name

    if audio_path:
        trimmed = _trim_name(audio_path.name, ".audio.wav", drop_parts=2)
        if trimmed:
            return trimmed

    trimmed = _trim_name(cache_path.name, ".whisper.json", drop_parts=2)
    if trimmed and trimmed != cache_path.name:
        return trimmed
    return cache_path.stem


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """ASR CLI with subcommands for full pipeline, transcription only, diarization only, and exports."""


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model", default="medium", show_default=True, help="Whisper model name.")
@click.option(
    "--whisper-device",
    default="auto",
    type=click.Choice(["cpu", "cuda", "mps", "auto"], case_sensitive=False),
    show_default=True,
    help="Device for Whisper.",
)
@click.option(
    "--diar-device",
    default="auto",
    type=click.Choice(["cpu", "cuda", "mps", "auto"], case_sensitive=False),
    show_default=True,
    help="Device for diarization.",
)
@click.option("--num-speakers", type=int, default=None, help="Exact number of speakers.")
@click.option("--min-speakers", type=int, default=None, help="Minimum number of speakers.")
@click.option("--max-speakers", type=int, default=None, help="Maximum number of speakers.")
@click.option("--language", type=str, default=None, help="Force language for Whisper (optional).")
@click.option("--hf-token", type=str, default=None, help="HF token; assumes terms already accepted.")
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory (default: same as input file).",
)
@click.option(
    "--format",
    "format_",
    default="srt,txt,json,rttm,dialog",
    show_default=True,
    help="Comma-separated output formats: srt,txt,json,rttm,dialog",
)
@click.option(
    "--cache",
    "cache_enable",
    is_flag=True,
    help="Also store Whisper cache under a cache directory (default: ~/.cache/asr-mps/<model>/<fingerprint>).",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Root directory for Whisper cache; implies --cache (default root: ~/.cache/asr-mps/<model>/<fingerprint>).",
)
@click.option(
    "--interactive-speakers",
    is_flag=True,
    help="Prompt to rename speakers after diarization.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Show tracebacks on unexpected errors.",
)
def full(
    input: Path,
    model: str,
    whisper_device: str,
    diar_device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    language: Optional[str],
    hf_token: Optional[str],
    out: Optional[Path],
    format_: str,
    cache_enable: bool,
    cache_dir: Optional[Path],
    interactive_speakers: bool,
    log_level: str,
    debug: bool,
) -> None:
    """Run the full pipeline: extract audio -> Whisper -> diarization -> exports."""
    setup_logging(log_level)
    hf_token = _resolve_hf_token(hf_token)
    try:
        _run_full_pipeline(
            input,
            model,
            whisper_device,
            diar_device,
            num_speakers,
            min_speakers,
            max_speakers,
            language,
            hf_token,
            out,
            format_,
            cache_enable or cache_dir is not None,
            cache_dir,
            interactive_speakers,
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        fatal(str(exc))


def _run_full_pipeline(
    input_path: Path,
    model: str,
    whisper_device: str,
    diar_device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    language: Optional[str],
    hf_token: Optional[str],
    out_dir: Optional[Path],
    formats_raw: str,
    cache_enable: bool,
    cache_root: Optional[Path],
    interactive_speakers: bool,
) -> None:
    in_path = input_path.expanduser().resolve()
    if not in_path.exists():
        fatal(f"Input not found: {in_path}")

    out_dir = _ensure_out_dir(out_dir, in_path.parent)
    formats = parse_formats(formats_raw)
    diar_kwargs = _build_diar_kwargs(num_speakers, min_speakers, max_speakers)

    whisper_dev = whisper_infer.resolve_device(whisper_device)
    diar_dev = diarization.resolve_device(diar_device)
    base_name = in_path.stem

    with tempfile.TemporaryDirectory() as tmpd:
        wav_path = Path(tmpd) / "audio_16k.wav"
        with step_status("Extracting audio (ffmpeg)"):
            audio.extract_wav(in_path, wav_path)

        with step_status(f"Loading Whisper model '{model}' on {whisper_dev}", spinner=False):
            w_model = whisper_infer.load_whisper(model, whisper_dev)
        with step_status("Running Whisper transcription", spinner=False):
            wres = whisper_infer.run_whisper(w_model, wav_path, language)
        w_segments = wres.get("segments", [])
        duration = _duration_from_segments(w_segments)
        fingerprint = whisper_infer.fingerprint_file(wav_path)

        with step_status("Loading diarization pipeline", spinner=False):
            pipeline = diarization.load_pyannote(hf_token)
        with step_status(f"Moving diarization pipeline to {diar_dev}"):
            diarization.send_pipeline_to_device(pipeline, diar_dev)
        with step_status("Running diarization"):
            diar = diarization.run_diarization(pipeline, wav_path, diar_kwargs)

        whisper_infer.export_whisper_cache(
            wres,
            wav_path,
            out_dir,
            base_name,
            model,
            whisper_dev,
            language,
            source_path=in_path,
        )
        if cache_enable or cache_root is not None:
            cache_dir = _resolve_cache_dir(cache_root, model, fingerprint)
            whisper_infer.export_whisper_cache(
                wres,
                wav_path,
                cache_dir,
                base_name,
                model,
                whisper_dev,
                language,
                source_path=in_path,
            )

    enriched = _attach_speakers_to_segments(w_segments, diar)
    if interactive_speakers and enriched:
        mapping = speakers.interactive_speaker_naming(enriched)
        speakers.apply_speaker_mapping(enriched, mapping)

    metadata = {
        "whisper_model": model,
        "whisper_device": str(whisper_dev),
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "diar_device": str(diar_dev),
        "num_speakers_param": num_speakers,
        "min_speakers_param": min_speakers,
        "max_speakers_param": max_speakers,
        "language": language,
        "input_file": str(in_path),
        "duration": duration,
    }
    _export_outputs(enriched, diar, formats, out_dir, base_name, metadata)
    logging.info("Done. Outputs written to %s", out_dir)


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--model", default="medium", show_default=True, help="Whisper model name.")
@click.option(
    "--whisper-device",
    default="auto",
    type=click.Choice(["cpu", "cuda", "mps", "auto"], case_sensitive=False),
    show_default=True,
    help="Device for Whisper.",
)
@click.option("--language", type=str, default=None, help="Force language for Whisper (optional).")
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory (default: same as input file).",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Root directory for Whisper cache (default: ~/.cache/asr-mps/<model>/<fingerprint>).",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Show tracebacks on unexpected errors.",
)
def transcribe(
    input: Path,
    model: str,
    whisper_device: str,
    language: Optional[str],
    out: Optional[Path],
    cache_dir: Optional[Path],
    log_level: str,
    debug: bool,
) -> None:
    """Run Whisper only and write the cache (no diarization)."""
    setup_logging(log_level)
    try:
        cache_root = cache_dir or out
        _run_transcription_only(input, model, whisper_device, language, cache_root)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        fatal(str(exc))


def _run_transcription_only(
    input_path: Path,
    model: str,
    whisper_device: str,
    language: Optional[str],
    cache_root: Optional[Path],
) -> None:
    in_path = input_path.expanduser().resolve()
    if not in_path.exists():
        fatal(f"Input not found: {in_path}")
    fallback = Path.home() / ".cache" / "asr-mps"
    cache_root = (cache_root or fallback).expanduser().resolve()
    whisper_dev = whisper_infer.resolve_device(whisper_device)

    with tempfile.TemporaryDirectory() as tmpd:
        wav_path = Path(tmpd) / "audio_16k.wav"
        with step_status("Extracting audio (ffmpeg)"):
            audio.extract_wav(in_path, wav_path)

        with step_status(f"Loading Whisper model '{model}' on {whisper_dev}", spinner=False):
            w_model = whisper_infer.load_whisper(model, whisper_dev)
        with step_status("Running Whisper transcription", spinner=False):
            wres = whisper_infer.run_whisper(w_model, wav_path, language)
        fingerprint = whisper_infer.fingerprint_file(wav_path)
        cache_dir = _resolve_cache_dir(cache_root, model, fingerprint)
        whisper_infer.export_whisper_cache(
            wres,
            wav_path,
            cache_dir,
            in_path.stem,
            model,
            whisper_dev,
            language,
            source_path=in_path,
        )
    logging.info("Whisper cache written to %s", cache_dir)


@cli.command()
@click.argument("whisper_cache", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--audio",
    "audio_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override audio path (defaults to cached audio referenced by the whisper cache).",
)
@click.option(
    "--diar-device",
    default="auto",
    type=click.Choice(["cpu", "cuda", "mps", "auto"], case_sensitive=False),
    show_default=True,
    help="Device for diarization.",
)
@click.option("--num-speakers", type=int, default=None, help="Exact number of speakers.")
@click.option("--min-speakers", type=int, default=None, help="Minimum number of speakers.")
@click.option("--max-speakers", type=int, default=None, help="Maximum number of speakers.")
@click.option("--hf-token", type=str, default=None, help="HF token; assumes terms already accepted.")
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory (default: same as cache directory).",
)
@click.option(
    "--format",
    "format_",
    default="srt,txt,json,rttm,dialog",
    show_default=True,
    help="Comma-separated output formats: srt,txt,json,rttm,dialog",
)
@click.option(
    "--interactive-speakers",
    is_flag=True,
    help="Prompt to rename speakers after diarization.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Show tracebacks on unexpected errors.",
)
def diarize(
    whisper_cache: Path,
    audio_path: Optional[Path],
    diar_device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    hf_token: Optional[str],
    out: Optional[Path],
    format_: str,
    interactive_speakers: bool,
    log_level: str,
    debug: bool,
) -> None:
    """Run diarization using an existing Whisper cache + audio, then export outputs."""
    setup_logging(log_level)
    hf_token = _resolve_hf_token(hf_token)
    try:
        _run_diarization_only(
            whisper_cache,
            audio_path,
            diar_device,
            num_speakers,
            min_speakers,
            max_speakers,
            hf_token,
            out,
            format_,
            interactive_speakers,
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        fatal(str(exc))


def _run_diarization_only(
    whisper_cache: Path,
    audio_path_override: Optional[Path],
    diar_device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    hf_token: Optional[str],
    out_dir: Optional[Path],
    formats_raw: str,
    interactive_speakers: bool,
) -> None:
    cache_path = whisper_cache.expanduser().resolve()
    wres, cache_metadata = _load_whisper_cache(cache_path)
    formats = parse_formats(formats_raw)
    diar_kwargs = _build_diar_kwargs(num_speakers, min_speakers, max_speakers)

    audio_path = _resolve_cached_audio(audio_path_override, cache_metadata, cache_path)
    out_dir = _ensure_out_dir(out_dir, cache_path.parent)
    base_name = _base_name_from_metadata(cache_path, cache_metadata, audio_path)

    diar_dev = diarization.resolve_device(diar_device)

    with step_status("Loading diarization pipeline", spinner=False):
        pipeline = diarization.load_pyannote(hf_token)
    with step_status(f"Moving diarization pipeline to {diar_dev}"):
        diarization.send_pipeline_to_device(pipeline, diar_dev)
    with step_status("Running diarization"):
        diar = diarization.run_diarization(pipeline, audio_path, diar_kwargs)

    w_segments = wres.get("segments", [])
    duration = _duration_from_segments(w_segments)
    enriched = _attach_speakers_to_segments(w_segments, diar)
    if interactive_speakers and enriched:
        mapping = speakers.interactive_speaker_naming(enriched)
        speakers.apply_speaker_mapping(enriched, mapping)

    metadata = {
        **cache_metadata,
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "diar_device": str(diar_dev),
        "num_speakers_param": num_speakers,
        "min_speakers_param": min_speakers,
        "max_speakers_param": max_speakers,
        "duration": cache_metadata.get("duration") or duration,
    }
    _export_outputs(enriched, diar, formats, out_dir, base_name, metadata)
    logging.info("Done. Outputs written to %s", out_dir)


@cli.command("export")
@click.argument("aligned_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory (default: same as aligned JSON).",
)
@click.option(
    "--format",
    "format_",
    default="srt,txt,json,dialog",
    show_default=True,
    help="Comma-separated output formats: srt,txt,json,dialog (RTTM not supported here).",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Show tracebacks on unexpected errors.",
)
def export_cmd(
    aligned_json: Path,
    out: Optional[Path],
    format_: str,
    log_level: str,
    debug: bool,
) -> None:
    """
    Re-export human-readable formats from a final aligned JSON (speaker-tagged segments).
    """
    setup_logging(log_level)
    try:
        _run_export(aligned_json, out, format_)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        fatal(str(exc))


def _run_export(aligned_json: Path, out_dir: Optional[Path], formats_raw: str) -> None:
    aligned_json = aligned_json.expanduser().resolve()
    out_dir = _ensure_out_dir(out_dir, aligned_json.parent)

    try:
        data = json.loads(aligned_json.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        fatal(f"Failed to read aligned JSON: {exc}")

    segments = data.get("segments")
    if not isinstance(segments, list):
        fatal("Aligned JSON missing 'segments' list.")

    formats = parse_formats(formats_raw)
    if "rttm" in formats:
        logging.warning("RTTM export requires diarization; skipping RTTM for export command.")
        formats = {f for f in formats if f != "rttm"}

    metadata = data.get("metadata", {})
    base_name = _base_name_from_metadata(aligned_json, metadata, audio_path=None)

    _export_outputs(segments, None, formats, out_dir, base_name, metadata)
    logging.info("Exports written to %s", out_dir)


@cli.command("cache-info")
@click.argument("whisper_cache", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
def cache_info(whisper_cache: Path, log_level: str) -> None:
    """Print metadata summary for a Whisper cache."""
    setup_logging(log_level)
    wres, metadata = _load_whisper_cache(whisper_cache)
    segments = wres.get("segments", [])
    duration = _duration_from_segments(segments)

    info = {
        "cache_path": str(whisper_cache),
        "cached_audio": metadata.get("cached_audio", "unknown"),
        "input_file": metadata.get("input_file", "unknown"),
        "whisper_model": metadata.get("whisper_model", "unknown"),
        "whisper_device": metadata.get("whisper_device", "unknown"),
        "language": metadata.get("language", "auto"),
        "audio_fingerprint_sha256": metadata.get("audio_fingerprint_sha256", "n/a"),
        "segment_count": len(segments),
        "duration": duration,
    }
    for k, v in info.items():
        click.echo(f"{k}: {v}")


def main(argv: Optional[List[str]] = None) -> None:
    # Best-effort warning if pyannote cannot be imported; keep CLI usable for
    # Whisper-only commands/help even on hosts with incompatible torchaudio.
    diarization.probe_pyannote_available()
    load_dotenv(override=False)
    cli.main(args=argv, prog_name="asr-mps")


if __name__ == "__main__":
    main()
