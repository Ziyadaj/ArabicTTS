"""SADA 2022 → XTTS-friendly corpus.

SADA ships long audio files plus a CSV of utterance-level segments with
`SegmentStart`/`SegmentEnd` offsets. XTTS fine-tuning wants one wav per
utterance at 22050 Hz plus an LJSpeech-style `metadata.csv`
(`wav|text|speaker`). This module does that conversion.

The heavy lifting (audio I/O) is factored behind small injectable callables
so we can unit-test the filtering/manifest logic without pulling in librosa.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arabic_tts.config import XTTS_SAMPLE_RATE
from arabic_tts.data.text import normalize_arabic

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "FileName",
    "ProcessedText",
    "SpeakerDialect",
    "SegmentStart",
    "SegmentEnd",
    "Speaker",
}

MIN_SECONDS = 1.0
MAX_SECONDS = 11.0  # XTTS chunks around this length
MIN_CHARS = 3


@dataclass(frozen=True)
class Utterance:
    wav_relpath: str
    text: str
    speaker: str


LoadAudio = Callable[[Path, int], tuple[np.ndarray, int]]
SaveAudio = Callable[[Path, np.ndarray, int], None]


def _default_load(path: Path, sr: int) -> tuple[np.ndarray, int]:
    import librosa  # local import: not needed for unit tests

    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32), sr


def _default_save(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="PCM_16")


def load_sada_csv(csv_path: Path, dialect: str = "Najdi") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"SADA CSV {csv_path} missing columns: {sorted(missing)}")
    df = df[df["SpeakerDialect"] == dialect].reset_index(drop=True)
    logger.info("Loaded %d %s rows from %s", len(df), dialect, csv_path)
    return df


def filter_segments(
    df: pd.DataFrame,
    *,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    min_chars: int = MIN_CHARS,
) -> pd.DataFrame:
    """Drop segments that are too short, too long, or have empty text."""
    duration = df["SegmentEnd"].astype(float) - df["SegmentStart"].astype(float)
    text = df["ProcessedText"].fillna("").astype(str).map(normalize_arabic)
    keep = (
        (duration >= min_seconds)
        & (duration <= max_seconds)
        & (text.str.len() >= min_chars)
    )
    out = df.loc[keep].copy()
    out["ProcessedText"] = text[keep]
    logger.info("Kept %d / %d segments after filtering", len(out), len(df))
    return out.reset_index(drop=True)


def _segment_id(row: pd.Series, idx: int) -> str:
    stem = Path(row["FileName"]).stem
    start_ms = int(round(float(row["SegmentStart"]) * 1000))
    end_ms = int(round(float(row["SegmentEnd"]) * 1000))
    return f"{stem}_{start_ms:08d}_{end_ms:08d}_{idx:06d}"


def _slice_segment(wav: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(round(start_s * sr)))
    end = min(len(wav), int(round(end_s * sr)))
    if end <= start:
        return np.zeros(0, dtype=np.float32)
    return wav[start:end]


def build_manifest(
    df: pd.DataFrame,
    *,
    source_audio_dir: Path,
    out_dir: Path,
    sample_rate: int = XTTS_SAMPLE_RATE,
    load_audio: LoadAudio | None = None,
    save_audio: SaveAudio | None = None,
) -> list[Utterance]:
    """Segment SADA audio and emit per-utterance wavs + return manifest rows.

    `load_audio`/`save_audio` are injectable so tests can run without librosa.
    """
    load = load_audio or _default_load
    save = save_audio or _default_save

    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, tuple[np.ndarray, int]] = {}
    rows: list[Utterance] = []

    for idx, row in df.iterrows():
        src = source_audio_dir / row["FileName"]
        if src not in cache:
            try:
                cache[src] = load(src, sample_rate)
            except FileNotFoundError:
                logger.warning("Missing source audio, skipping: %s", src)
                continue
            # bound cache — SADA files are long, don't hoard
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
                cache[src] = load(src, sample_rate)

        wav, sr = cache[src]
        segment = _slice_segment(wav, sr, float(row["SegmentStart"]), float(row["SegmentEnd"]))
        if segment.size == 0:
            continue

        seg_id = _segment_id(row, idx)
        out_wav = wavs_dir / f"{seg_id}.wav"
        save(out_wav, segment, sr)

        rows.append(
            Utterance(
                wav_relpath=f"wavs/{seg_id}.wav",
                text=str(row["ProcessedText"]),
                speaker=str(row["Speaker"]),
            )
        )

    logger.info("Wrote %d segment wavs to %s", len(rows), wavs_dir)
    return rows


def write_metadata(rows: Iterable[Utterance], out_dir: Path, *, filename: str = "metadata.csv") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow([row.wav_relpath, row.text, row.speaker])
    return path


def prepare_split(
    csv_path: Path,
    *,
    source_audio_dir: Path,
    out_dir: Path,
    dialect: str = "Najdi",
    sample_rate: int = XTTS_SAMPLE_RATE,
    load_audio: LoadAudio | None = None,
    save_audio: SaveAudio | None = None,
) -> Path:
    """End-to-end: CSV → filtered manifest + segmented wavs. Returns metadata path."""
    df = load_sada_csv(csv_path, dialect=dialect)
    df = filter_segments(df)
    rows = build_manifest(
        df,
        source_audio_dir=source_audio_dir,
        out_dir=out_dir,
        sample_rate=sample_rate,
        load_audio=load_audio,
        save_audio=save_audio,
    )
    return write_metadata(rows, out_dir)
