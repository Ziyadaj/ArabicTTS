import pandas as pd

from arabic_tts.data.inspect import summarize


def _csv(tmp_path, rows):
    path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(**kw):
    base = {
        "FileName": "f.wav",
        "ProcessedText": "x",
        "SpeakerDialect": "Najdi",
        "SegmentStart": 0.0,
        "SegmentEnd": 3600.0,  # exactly 1 hour
        "Speaker": "spk1",
        "Environment": "Clean",
        "SpeakerGender": "male",
    }
    base.update(kw)
    return base


def test_summarize_aggregates_hours_correctly(tmp_path):
    csv = _csv(
        tmp_path,
        [
            _row(SpeakerDialect="Najdi", Environment="Clean", SegmentEnd=3600.0),
            _row(SpeakerDialect="Najdi", Environment="Music", SegmentEnd=1800.0),
            _row(SpeakerDialect="Hijazi", Environment="Clean", SegmentEnd=900.0),
        ],
    )
    r = summarize(csv)
    assert r.total_segments == 3
    assert r.total_hours == 1 + 0.5 + 0.25
    assert r.by_dialect["Najdi"] == 1.5
    assert r.by_dialect["Hijazi"] == 0.25
    assert r.by_environment["Clean"] == 1.25
    assert r.by_environment["Music"] == 0.5


def test_summarize_dialect_filter(tmp_path):
    csv = _csv(
        tmp_path,
        [
            _row(SpeakerDialect="Najdi"),
            _row(SpeakerDialect="Hijazi"),
        ],
    )
    r = summarize(csv, dialect="Najdi")
    assert r.total_segments == 1
    assert "Najdi" in r.by_dialect
    assert "Hijazi" not in r.by_dialect


def test_render_smoke(tmp_path):
    csv = _csv(tmp_path, [_row()])
    out = summarize(csv).render()
    assert "Najdi" in out
    assert "Clean" in out
    assert "spk1" in out
