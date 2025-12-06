import logging
from typing import Dict, List, Set

from .export import format_hhmmss
from .utils import fatal


def best_speaker(diarization, start: float, end: float) -> str:
    try:
        from pyannote.core import Segment
    except Exception as exc:  # pragma: no cover - optional dependency might be missing
        fatal(f"pyannote.core is unavailable; cannot assign speakers. Details: {exc}")

    window = Segment(start, end)
    overlap_by_label: Dict[str, float] = {}
    for seg, _, label in diarization.itertracks(yield_label=True):
        inter = seg & window
        if inter is not None:
            overlap_by_label[label] = overlap_by_label.get(label, 0.0) + inter.duration
    if not overlap_by_label:
        return "SPEAKER_??"
    return max(overlap_by_label.items(), key=lambda kv: kv[1])[0]


def collect_speakers(segments: List[Dict]) -> List[str]:
    labels: Set[str] = {seg["speaker"] for seg in segments}
    return sorted(labels)


def interactive_speaker_naming(segments: List[Dict]) -> Dict[str, str]:
    """
    Prompt user to rename SPEAKER_xx to human names.
    Returns mapping from old label -> new label.
    """
    speaker_labels = collect_speakers(segments)
    if not speaker_labels:
        return {}

    print("\n=== Interactive speaker naming ===")
    print("You can rename speakers (e.g. SPEAKER_00 -> Alice).")
    print("Press Enter to keep the default label.\n")

    mapping: Dict[str, str] = {}
    for label in speaker_labels:
        examples = [s for s in segments if s["speaker"] == label][:3]
        print(f"\n{label}:")
        for ex in examples:
            ts = format_hhmmss(ex["start"])
            print(f"  [{ts}] {ex['text']}")
        new_name = input(f"Name for {label} (or Enter to keep '{label}'): ").strip()
        if new_name:
            mapping[label] = new_name

    if mapping:
        logging.info("Speaker mapping chosen: %s", mapping)
    else:
        logging.info("No speaker renaming provided; keeping default labels.")
    return mapping


def apply_speaker_mapping(segments: List[Dict], mapping: Dict[str, str]) -> None:
    if not mapping:
        return
    for seg in segments:
        old = seg["speaker"]
        if old in mapping:
            seg["speaker"] = mapping[old]
