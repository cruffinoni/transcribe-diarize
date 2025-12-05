import logging
import tempfile
from pathlib import Path
from typing import Optional, Set

import click

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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
def cli(
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
    interactive_speakers: bool,
    log_level: str,
    debug: bool,
) -> None:
    setup_logging(log_level)
    try:
        run_pipeline(
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
            interactive_speakers,
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        fatal(str(exc))


def run_pipeline(
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
    interactive_speakers: bool,
) -> None:
    in_path = input_path.expanduser().resolve()
    if not in_path.exists():
        fatal(f"Input not found: {in_path}")

    out_dir = (out_dir or in_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = parse_formats(formats_raw)
    base_name = in_path.stem

    whisper_dev = whisper_infer.resolve_device(whisper_device)
    diar_dev = diarization.resolve_device(diar_device)

    with tempfile.TemporaryDirectory() as tmpd:
        wav_path = Path(tmpd) / "audio_16k.wav"
        with step_status("Extracting audio (ffmpeg)"):
            audio.extract_wav(in_path, wav_path)

        with step_status(f"Loading Whisper model '{model}' on {whisper_dev}"):
            w_model = whisper_infer.load_whisper(model, whisper_dev)
        # Disable spinner to avoid clashing with Whisper's tqdm progress bar
        with step_status("Running Whisper transcription", spinner=False):
            wres = whisper_infer.run_whisper(w_model, wav_path, language)
        w_segments = wres.get("segments", [])
        duration = float(w_segments[-1]["end"]) if w_segments else 0.0

        with step_status("Loading diarization pipeline"):
            pipeline = diarization.load_pyannote(hf_token)
        with step_status(f"Moving diarization pipeline to {diar_dev}"):
            diarization.send_pipeline_to_device(pipeline, diar_dev)

        diar_kw = {}
        if num_speakers is not None:
            diar_kw["num_speakers"] = int(num_speakers)
        else:
            if min_speakers is not None:
                diar_kw["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                diar_kw["max_speakers"] = int(max_speakers)

        with step_status("Running diarization"):
            diar = diarization.run_diarization(pipeline, wav_path, diar_kw)

    enriched = []
    for seg in w_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        speaker_label = speakers.best_speaker(diar, start, end)
        enriched.append(
            {
                "id": seg.get("id"),
                "start": start,
                "end": end,
                "text": (seg.get("text") or "").strip(),
                "speaker": speaker_label,
            }
        )

    if interactive_speakers and enriched:
        mapping = speakers.interactive_speaker_naming(enriched)
        speakers.apply_speaker_mapping(enriched, mapping)

    speaker_labels = speakers.collect_speakers(enriched)

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

    if "rttm" in formats:
        export.export_rttm(diar, out_dir / f"{base_name}.rttm")
    if "srt" in formats:
        export.export_srt(enriched, out_dir / f"{base_name}.srt")
    if "txt" in formats:
        export.export_txt(enriched, out_dir / f"{base_name}.txt")
    if "dialog" in formats:
        export.export_dialog(enriched, out_dir / f"{base_name}.dialog.txt")
    if "json" in formats:
        export.export_json(enriched, out_dir / f"{base_name}.json", metadata, speaker_labels)

    logging.info("Done. Outputs written to %s", out_dir)


def main() -> None:
    cli(prog_name="transcribe_diarize")


if __name__ == "__main__":
    main()
