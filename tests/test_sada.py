from pathlib import Path

import numpy as np
import pandas as pd

from arabic_tts.data.sada import (
    REQUIRED_COLUMNS,
    Utterance,
    build_manifest,
    filter_segments,
    load_sada_csv,
    prepare_split,
    write_metadata,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _base_row(**overrides) -> dict:
    row = {
        "FileName": "f1.wav",
        "ProcessedText": "مرحبا بك في هذا الاختبار",
        "SpeakerDialect": "Najdi",
        "SegmentStart": 0.0,
        "SegmentEnd": 3.0,
        "Speaker": "spk1",
    }
    row.update(overrides)
    return row


def test_required_columns_constant_matches_base_row():
    assert set(_base_row().keys()) >= REQUIRED_COLUMNS


def test_load_sada_csv_filters_dialect(tmp_path):
    csv = tmp_path / "train.csv"
    _write_csv(
        csv,
        [
            _base_row(Speaker="a"),
            _base_row(Speaker="b", SpeakerDialect="Hijazi"),
            _base_row(Speaker="c"),
        ],
    )
    df = load_sada_csv(csv)
    assert len(df) == 2
    assert set(df["Speaker"]) == {"a", "c"}


def test_load_sada_csv_raises_on_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame([{"FileName": "x.wav"}]).to_csv(csv, index=False)
    try:
        load_sada_csv(csv)
    except ValueError as e:
        assert "missing columns" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_filter_segments_drops_too_short_and_too_long():
    df = pd.DataFrame(
        [
            _base_row(SegmentEnd=0.5),  # 0.5s — too short
            _base_row(SegmentEnd=4.0),  # 4s — keep
            _base_row(SegmentEnd=20.0),  # 20s — too long
            _base_row(ProcessedText="", SegmentEnd=3.0),  # empty text — drop
            _base_row(ProcessedText="ab", SegmentEnd=3.0),  # too few chars — drop
        ]
    )
    out = filter_segments(df)
    assert len(out) == 1
    assert out.iloc[0]["SegmentEnd"] == 4.0


def test_filter_segments_normalizes_text():
    df = pd.DataFrame([_base_row(ProcessedText="كِتَابٌ جَمِيلٌ")])
    out = filter_segments(df)
    assert out.iloc[0]["ProcessedText"] == "كتاب جميل"


def test_build_manifest_segments_audio(tmp_path):
    df = pd.DataFrame(
        [
            _base_row(FileName="a.wav", SegmentStart=0.0, SegmentEnd=1.0),
            _base_row(FileName="a.wav", SegmentStart=1.0, SegmentEnd=2.0, Speaker="spk2"),
        ]
    )
    df = filter_segments(df)

    sr = 22050
    # Fake 5 seconds of audio. First second = 1.0, second second = 2.0, etc.
    fake = np.concatenate([np.full(sr, float(i)) for i in range(1, 6)]).astype(np.float32)

    loads: list[Path] = []
    saves: list[tuple[Path, int, float]] = []

    def fake_load(path: Path, rate: int):
        loads.append(path)
        return fake, rate

    def fake_save(path: Path, wav: np.ndarray, rate: int):
        saves.append((path, rate, float(wav[0]) if wav.size else 0.0))

    rows = build_manifest(
        df,
        source_audio_dir=tmp_path,
        out_dir=tmp_path / "prepared",
        load_audio=fake_load,
        save_audio=fake_save,
    )

    assert len(rows) == 2
    assert loads == [tmp_path / "a.wav"]  # cached, read once
    # First segment is the first second → value 1.0
    assert saves[0][2] == 1.0
    # Second segment is the second → value 2.0
    assert saves[1][2] == 2.0
    assert all(r.wav_relpath.startswith("wavs/") for r in rows)


def test_build_manifest_skips_missing_source(tmp_path, caplog):
    df = pd.DataFrame([_base_row(FileName="ghost.wav")])
    df = filter_segments(df)

    def fake_load(path: Path, rate: int):
        raise FileNotFoundError(path)

    def fake_save(path: Path, wav: np.ndarray, rate: int):
        raise AssertionError("should not be called")

    rows = build_manifest(
        df,
        source_audio_dir=tmp_path,
        out_dir=tmp_path / "prepared",
        load_audio=fake_load,
        save_audio=fake_save,
    )
    assert rows == []


def test_write_metadata_format(tmp_path):
    rows = [
        Utterance("wavs/a.wav", "مرحبا", "spk1"),
        Utterance("wavs/b.wav", "كيف حالك", "spk2"),
    ]
    path = write_metadata(rows, tmp_path)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "wavs/a.wav|مرحبا|spk1"
    assert text[1] == "wavs/b.wav|كيف حالك|spk2"


def test_prepare_split_end_to_end(tmp_path):
    csv = tmp_path / "train.csv"
    _write_csv(
        csv,
        [
            _base_row(FileName="a.wav", SegmentStart=0.0, SegmentEnd=2.0),
            _base_row(FileName="a.wav", SegmentStart=2.0, SegmentEnd=4.0),
            _base_row(FileName="a.wav", SegmentStart=0.0, SegmentEnd=0.5),  # too short
            _base_row(SpeakerDialect="Hijazi", SegmentEnd=3.0),  # wrong dialect
        ],
    )

    sr = 22050
    fake = np.zeros(sr * 5, dtype=np.float32)

    def fake_load(path: Path, rate: int):
        return fake, rate

    writes: list[Path] = []

    def fake_save(path: Path, wav: np.ndarray, rate: int):
        writes.append(path)

    out_dir = tmp_path / "out"
    meta_path = prepare_split(
        csv,
        source_audio_dir=tmp_path,
        out_dir=out_dir,
        load_audio=fake_load,
        save_audio=fake_save,
        # Disable optional filters that would otherwise drop the test fixture.
        environments=None,
        genders=None,
        min_utterances_per_speaker=1,
        max_utterances_per_speaker=None,
    )
    assert meta_path.is_file()
    # 2 kept rows
    assert len(writes) == 2
    assert len(meta_path.read_text().splitlines()) == 2
    # reference.txt is written so training can auto-pick a speaker_wav.
    assert (out_dir / "reference.txt").is_file()
