import pandas as pd

from arabic_tts.data.sada import (
    Utterance,
    balance_speakers,
    cap_total_hours,
    filter_quality,
    pick_reference_clip,
)


def _row(**kw):
    base = {
        "FileName": "f.wav",
        "ProcessedText": "نص قصير",
        "SpeakerDialect": "Najdi",
        "SegmentStart": 0.0,
        "SegmentEnd": 3.0,
        "Speaker": "spk1",
        "Environment": "Clean",
        "SpeakerGender": "male",
    }
    base.update(kw)
    return base


def test_filter_quality_keeps_clean_named_gender_single_speaker():
    df = pd.DataFrame(
        [
            _row(),
            _row(Environment="Music"),
            _row(SpeakerGender="unknown"),
            _row(SpeakerDialect="More than 1 speaker"),
        ]
    )
    out = filter_quality(df)
    assert len(out) == 1
    assert out.iloc[0]["Environment"] == "Clean"


def test_filter_quality_environments_none_keeps_all_envs():
    df = pd.DataFrame([_row(), _row(Environment="Music"), _row(Environment="Noisy")])
    out = filter_quality(df, environments=None)
    assert len(out) == 3


def test_balance_speakers_drops_tiny_and_caps_dominant():
    rows = []
    # 3 utterances from spk_small (below min 5), 50 from spk_big (above max 10)
    rows += [_row(Speaker="spk_small") for _ in range(3)]
    rows += [_row(Speaker="spk_big") for _ in range(50)]
    rows += [_row(Speaker="spk_mid") for _ in range(8)]
    df = pd.DataFrame(rows)
    out = balance_speakers(df, min_utterances=5, max_utterances=10)
    counts = out["Speaker"].value_counts().to_dict()
    assert "spk_small" not in counts
    assert counts["spk_big"] == 10
    assert counts["spk_mid"] == 8


def test_balance_speakers_max_none_keeps_all_above_min():
    rows = [_row(Speaker="a") for _ in range(20)] + [_row(Speaker="b") for _ in range(2)]
    df = pd.DataFrame(rows)
    out = balance_speakers(df, min_utterances=5, max_utterances=None)
    assert (out["Speaker"] == "a").sum() == 20
    assert (out["Speaker"] == "b").sum() == 0


def test_cap_total_hours_respects_budget():
    # 100 segments × 60s = 100 minutes total. Cap at 0.5h (30 min).
    rows = [_row(SegmentStart=0.0, SegmentEnd=60.0, Speaker=f"s{i}") for i in range(100)]
    df = pd.DataFrame(rows)
    out = cap_total_hours(df, max_hours=0.5)
    duration = (out["SegmentEnd"] - out["SegmentStart"]).sum() / 3600.0
    assert duration <= 0.5
    # Should keep close to 30 segments (0.5h / 60s/segment).
    assert 25 <= len(out) <= 31


def test_pick_reference_clip_picks_longest_text():
    rows = [
        Utterance("wavs/a.wav", "short", "spk1"),
        Utterance("wavs/b.wav", "this is a much longer reference clip", "spk2"),
        Utterance("wavs/c.wav", "mid length text here", "spk1"),
    ]
    pick = pick_reference_clip(rows)
    assert pick.wav_relpath == "wavs/b.wav"

    pick2 = pick_reference_clip(rows, prefer_speaker="spk1")
    assert pick2.wav_relpath == "wavs/c.wav"


def test_pick_reference_clip_empty_returns_none():
    assert pick_reference_clip([]) is None
