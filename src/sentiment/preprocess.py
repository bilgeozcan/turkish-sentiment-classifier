"""Turkish-aware text normalization."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"[@#]\w+")
_NON_LETTER = re.compile(r"[^a-zçğıöşü\s]")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACES = re.compile(r"\s+")


def turkish_lower(text: str) -> str:
    """Lowercase handling the Turkish dotted/dotless i correctly (I -> ı, İ -> i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str) -> str:
    """Lowercase, strip URLs/mentions/digits/punctuation and squeeze letter repetitions.

    'ÇOOOK güzel!!! http://x.co @ali' -> 'çook güzel'
    """
    text = turkish_lower(text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)  # 'çoooook' -> 'çoo'
    return _SPACES.sub(" ", text).strip()
