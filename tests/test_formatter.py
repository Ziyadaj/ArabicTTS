from pathlib import Path

from arabic_tts.data.formatter import sada_najdi_formatter


def test_formatter_parses_metadata(tmp_path: Path):
    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "wavs/a.wav|مرحبا|spk1\n"
        "wavs/b.wav|كيف حالك|spk2\n"
        "\n"  # blank line should be ignored
        "malformed\n",  # too few fields, ignored
        encoding="utf-8",
    )
    items = sada_najdi_formatter(str(tmp_path), "metadata.csv")
    assert len(items) == 2
    assert items[0]["text"] == "مرحبا"
    assert items[0]["speaker_name"] == "spk1"
    assert items[0]["language"] == "ar"
    assert items[0]["audio_file"].endswith("wavs/a.wav")
    assert items[1]["speaker_name"] == "spk2"


def test_formatter_ignores_unknown_kwargs(tmp_path: Path):
    meta = tmp_path / "metadata.csv"
    meta.write_text("wavs/a.wav|hi|spk\n", encoding="utf-8")
    # Coqui passes extra kwargs like `eval_split`, `ignored_speakers`.
    items = sada_najdi_formatter(
        str(tmp_path), "metadata.csv", eval_split=True, ignored_speakers=None
    )
    assert len(items) == 1
