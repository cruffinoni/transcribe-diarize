import json
import logging
from pathlib import Path
from typing import Dict, List


def format_srt_ts(t: float) -> str:
    t_ms = max(0, int(round(t * 1000)))
    h = t_ms // 3_600_000
    m = (t_ms % 3_600_000) // 60_000
    s = (t_ms % 60_000) // 1000
    ms = t_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_hhmmss(t: float) -> str:
    t_ms = max(0, int(round(t * 1000)))
    h = t_ms // 3_600_000
    m = (t_ms % 3_600_000) // 60_000
    s = (t_ms % 60_000) // 1000
    return f"{h:02d}:{m:02d}:{s:02d}"


def export_srt(segments: List[Dict], out_path: Path) -> None:
    logging.info("Writing SRT to %s", out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_ts(seg['start'])} --> {format_srt_ts(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")


def export_txt(segments: List[Dict], out_path: Path) -> None:
    logging.info("Writing TXT to %s", out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['speaker']}: {seg['text']}\n")


def export_dialog(segments: List[Dict], out_path: Path) -> None:
    """
    Human-friendly dialog view: [HH:MM:SS] Speaker: text
    """
    logging.info("Writing dialog TXT to %s", out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            ts = format_hhmmss(seg["start"])
            f.write(f"[{ts}] {seg['speaker']}: {seg['text']}\n")


def export_json(
    segments: List[Dict],
    out_path: Path,
    metadata: Dict,
    speakers: List[str],
) -> None:
    logging.info("Writing JSON to %s", out_path)
    payload = {
        "metadata": metadata,
        "speakers": [{"id": s, "display_name": s} for s in speakers],
        "segments": segments,
    }
    with out_path.open("w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)


def export_rttm(diarization, out_path: Path) -> None:
    logging.info("Writing RTTM to %s", out_path)
    with out_path.open("w") as rttm:
        diarization.write_rttm(rttm)
