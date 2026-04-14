"""Coqui TTS expects a dataset loader callable with a specific signature."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def sada_najdi_formatter(root_path: str, meta_file_train: str, **_: Any) -> list[dict[str, Any]]:
    """Coqui `formatter` callable for the prepared SADA Najdi corpus.

    Expects `root_path/meta_file_train` to be our pipe-delimited `metadata.csv`
    with columns `wav|text|speaker`, and wavs living under `root_path/wavs/`.
    """
    root = Path(root_path)
    meta = root / meta_file_train
    items: list[dict[str, Any]] = []
    with meta.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            wav_rel, text, speaker = row[0], row[1], row[2]
            items.append(
                {
                    "text": text,
                    "audio_file": str(root / wav_rel),
                    "speaker_name": speaker,
                    "root_path": str(root),
                    "language": "ar",
                }
            )
    return items
