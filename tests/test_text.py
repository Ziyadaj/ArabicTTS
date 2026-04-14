from arabic_tts.data.text import normalize_arabic


def test_strips_tashkeel():
    # "kitāb" with fatha/kasra/damma marks
    assert normalize_arabic("كِتَابٌ") == "كتاب"


def test_collapses_whitespace():
    assert normalize_arabic("  مرحبا   بك  ") == "مرحبا بك"


def test_arabic_punctuation_normalized():
    assert normalize_arabic("ما اسمك؟") == "ما اسمك?"
    assert normalize_arabic("نعم، شكرا") == "نعم, شكرا"


def test_strip_tatweel():
    assert normalize_arabic("سلـــام") == "سلام"


def test_empty_input():
    assert normalize_arabic("") == ""
    assert normalize_arabic(None) == ""  # type: ignore[arg-type]
