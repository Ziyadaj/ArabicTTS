from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

XTTS_SAMPLE_RATE: int = 22050
XTTS_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"


def default_xtts_dir() -> Path:
    return Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"


@dataclass(frozen=True)
class Paths:
    data_root: Path = Path("data/sada2022")
    prepared_root: Path = Path("data/sada_najdi_prepared")
    output_root: Path = Path("runs")
    xtts_dir: Path = field(default_factory=default_xtts_dir)
