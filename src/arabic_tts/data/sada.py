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
    keep = (duration >= min_seconds) & (duration <= max_seconds) & (text.str.len() >= min_chars)
    out = df.loc[keep].copy()
    out["ProcessedText"] = text[keep]
    logger.info("Kept %d / %d segments after filtering", len(out), len(df))
    return out.reset_index(drop=True)


def filter_quality(
    df: pd.DataFrame,
    *,
    environments: tuple[str, ...] | None = ("Clean",),
    genders: tuple[str, ...] | None = ("male", "female"),
    exclude_multispeaker: bool = True,
) -> pd.DataFrame:
    """Optional quality filters using SADA metadata columns.

    `environments=None` keeps all; `genders=None` keeps all. Defaults match a
    reasonable starting recipe: Clean recordings, named gender, single speaker.
    """
    out = df
    if environments and "Environment" in out.columns:
        out = out[out["Environment"].isin(environments)]
    if genders and "SpeakerGender" in out.columns:
        out = out[out["SpeakerGender"].isin(genders)]
    if exclude_multispeaker and "SpeakerDialect" in out.columns:
        out = out[out["SpeakerDialect"] != "More than 1 speaker"]
    out = out.reset_index(drop=True)
    logger.info("Quality filter kept %d / %d rows", len(out), len(df))
    return out


def balance_speakers(
    df: pd.DataFrame,
    *,
    min_utterances: int = 5,
    max_utterances: int | None = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop tiny speakers and cap dominant ones so no one voice swamps training."""
    if "Speaker" not in df.columns or df.empty:
        return df.reset_index(drop=True)
    counts = df["Speaker"].value_counts()
    keep_speakers = counts[counts >= min_utterances].index
    out = df[df["Speaker"].isin(keep_speakers)]
    if max_utterances is not None:
        rng = np.random.default_rng(seed)
        capped: list[pd.DataFrame] = []
        for _, group in out.groupby("Speaker", sort=False):
            if len(group) > max_utterances:
                idx = rng.choice(len(group), size=max_utterances, replace=False)
                capped.append(group.iloc[sorted(idx)])
            else:
                capped.append(group)
        out = pd.concat(capped, ignore_index=True) if capped else out.iloc[0:0]
    out = out.reset_index(drop=True)
    logger.info(
        "Speaker balance: %d → %d rows across %d speakers",
        len(df),
        len(out),
        out["Speaker"].nunique() if "Speaker" in out.columns else 0,
    )
    return out


def cap_total_hours(df: pd.DataFrame, *, max_hours: float, seed: int = 0) -> pd.DataFrame:
    """Take a random subset that sums to at most `max_hours` of audio."""
    if df.empty:
        return df.reset_index(drop=True)
    durations = df["SegmentEnd"].astype(float) - df["SegmentStart"].astype(float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cumulative = durations.iloc[order].cumsum() / 3600.0
    keep_idx = order[cumulative.values <= max_hours]
    out = df.iloc[sorted(keep_idx)].reset_index(drop=True)
    logger.info(
        "Hour cap %.2f h → %d rows (%.2f h)",
        max_hours,
        len(out),
        float(durations.iloc[sorted(keep_idx)].sum() / 3600.0),
    )
    return out


def pick_reference_clip(
    rows: list[Utterance], *, prefer_speaker: str | None = None
) -> Utterance | None:
    """Pick a stable reference clip for `test_sentences` and demo speaker_wav.

    Strategy: longest text, optionally restricted to a chosen speaker.
    """
    candidates = [r for r in rows if (prefer_speaker is None or r.speaker == prefer_speaker)]
    if not candidates:
        candidates = rows
    if not candidates:
        return None
    return max(candidates, key=lambda r: len(r.text))


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


def write_metadata(
    rows: Iterable[Utterance], out_dir: Path, *, filename: str = "metadata.csv"
) -> Path:
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
    environments: tuple[str, ...] | None = ("Clean",),
    genders: tuple[str, ...] | None = ("male", "female"),
    exclude_multispeaker: bool = True,
    min_utterances_per_speaker: int = 5,
    max_utterances_per_speaker: int | None = 200,
    max_hours: float | None = None,
    load_audio: LoadAudio | None = None,
    save_audio: SaveAudio | None = None,
) -> Path:
    """End-to-end: CSV → filtered manifest + segmented wavs. Returns metadata path.

    Defaults reflect a sensible starting recipe: Najdi only, Clean recordings,
    named gender, single speaker, length-filtered, speaker-balanced.
    """
    df = load_sada_csv(csv_path, dialect=dialect)
    df = filter_quality(
        df,
        environments=environments,
        genders=genders,
        exclude_multispeaker=exclude_multispeaker,
    )
    df = filter_segments(df)
    df = balance_speakers(
        df,
        min_utterances=min_utterances_per_speaker,
        max_utterances=max_utterances_per_speaker,
    )
    if max_hours is not None:
        df = cap_total_hours(df, max_hours=max_hours)
    rows = build_manifest(
        df,
        source_audio_dir=source_audio_dir,
        out_dir=out_dir,
        sample_rate=sample_rate,
        load_audio=load_audio,
        save_audio=save_audio,
    )
    meta = write_metadata(rows, out_dir)
    ref = pick_reference_clip(rows)
    if ref is not None:
        (out_dir / "reference.txt").write_text(ref.wav_relpath, encoding="utf-8")
    return meta
