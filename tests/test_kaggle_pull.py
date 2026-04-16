import pytest

from arabic_tts.data.kaggle_pull import verify_layout


def test_verify_layout_finds_csvs_and_audio(tmp_path):
    root = tmp_path / "extracted"
    inner = root / "sada2022"
    (inner / "audios").mkdir(parents=True)
    for name in ("train.csv", "valid.csv", "test.csv"):
        (inner / name).write_text("FileName,SegmentStart\n", encoding="utf-8")
    (inner / "audios" / "x.wav").write_bytes(b"\x00")

    layout = verify_layout(root)
    assert layout["__root__"] == inner
    assert layout["train.csv"].name == "train.csv"


def test_verify_layout_missing_csv_raises(tmp_path):
    root = tmp_path / "extracted"
    root.mkdir()
    (root / "train.csv").write_text("a,b\n", encoding="utf-8")
    (root / "x.wav").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match="Missing"):
        verify_layout(root)


def test_verify_layout_no_audio_raises(tmp_path):
    root = tmp_path / "extracted"
    root.mkdir()
    for name in ("train.csv", "valid.csv", "test.csv"):
        (root / name).write_text("a\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No .wav"):
        verify_layout(root)
