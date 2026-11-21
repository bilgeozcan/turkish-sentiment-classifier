from src.sentiment.preprocess import normalize, turkish_lower


def test_turkish_lower_dotted_dotless_i():
    assert turkish_lower("ISPARTA İZMİR") == "ısparta izmir"


def test_normalize_removes_noise_and_squeezes_repeats():
    assert normalize("ÇOOOOK güzel!!! http://x.co @ali #etiket 123") == "çook güzel"


def test_normalize_empty():
    assert normalize("   ...  ") == ""


def test_normalize_none_and_non_string():
    assert normalize(None) == ""
    assert normalize(12345) == ""


def test_normalize_emoji_and_mixed_i():
    assert normalize("İstanbul Isparta 👍") == "istanbul ısparta"
