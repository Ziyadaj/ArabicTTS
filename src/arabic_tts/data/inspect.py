"""Compute hours-by-bucket summaries for a SADA split.

Realistic training planning starts with: how much *Najdi* / *Clean* / *female*
data do I actually have? This module answers that without loading any audio —
it's all from the CSV's `SegmentStart`/`SegmentEnd` columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SECONDS_PER_HOUR = 3600.0


@dataclass
class SplitReport:
    csv: Path
    total_segments: int
    total_hours: float
    by_dialect: dict[str, float] = field(default_factory=dict)
    by_environment: dict[str, float] = field(default_factory=dict)
    by_gender: dict[str, float] = field(default_factory=dict)
    speakers: int = 0
    speakers_top10: list[tuple[str, int]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Split: {self.csv}",
            f"  segments: {self.total_segments:,}",
            f"  hours:    {self.total_hours:,.2f}",
            f"  speakers: {self.speakers:,}",
            "  by dialect (hours):",
        ]
        for k, v in sorted(self.by_dialect.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:30s} {v:8.2f}")
        lines.append("  by environment (hours):")
        for k, v in sorted(self.by_environment.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:30s} {v:8.2f}")
        lines.append("  by gender (hours):")
        for k, v in sorted(self.by_gender.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:30s} {v:8.2f}")
        if self.speakers_top10:
            lines.append("  top 10 speakers (utterance count):")
            for spk, n in self.speakers_top10:
                lines.append(f"    {spk:30s} {n:8d}")
        return "\n".join(lines)


def _hours(df: pd.DataFrame) -> pd.Series:
    return (df["SegmentEnd"].astype(float) - df["SegmentStart"].astype(float)) / SECONDS_PER_HOUR


def summarize(csv_path: Path, *, dialect: str | None = None) -> SplitReport:
    df = pd.read_csv(csv_path)
    if dialect is not None:
        df = df[df["SpeakerDialect"] == dialect].reset_index(drop=True)

    hours = _hours(df)
    df = df.assign(_hours=hours)

    by_dialect: dict[str, float] = {}
    if "SpeakerDialect" in df.columns:
        by_dialect = df.groupby("SpeakerDialect")["_hours"].sum().to_dict()

    by_environment: dict[str, float] = {}
    if "Environment" in df.columns:
        by_environment = df.groupby("Environment")["_hours"].sum().to_dict()

    by_gender: dict[str, float] = {}
    if "SpeakerGender" in df.columns:
        by_gender = df.groupby("SpeakerGender")["_hours"].sum().to_dict()

    speakers = int(df["Speaker"].nunique()) if "Speaker" in df.columns else 0
    top10: list[tuple[str, int]] = []
    if "Speaker" in df.columns:
        counts = df["Speaker"].value_counts().head(10)
        top10 = [(str(k), int(v)) for k, v in counts.items()]

    return SplitReport(
        csv=csv_path,
        total_segments=len(df),
        total_hours=float(hours.sum()),
        by_dialect={str(k): float(v) for k, v in by_dialect.items()},
        by_environment={str(k): float(v) for k, v in by_environment.items()},
        by_gender={str(k): float(v) for k, v in by_gender.items()},
        speakers=speakers,
        speakers_top10=top10,
    )
