"""Whisper-style English text normalisation for WER.

WER on raw text mostly measures formatting: "Liberty," vs "liberty" or "7" vs "seven"
would count as errors although the words were recognised. Both reference and hypothesis
go through ``normalize`` so only recognition differences remain. Deliberately lighter
than Whisper's ``EnglishTextNormalizer`` (no spelling standardisation, no contraction
expansion, numbers only 0-100), because the references are known texts and every extra
rule is one more way to hide a real error.
"""

from __future__ import annotations

import re
import unicodedata

_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>")
"""Annotations such as ``[music]`` or ``(laughs)``: not words anyone said."""
_FILLERS = re.compile(r"\b(?:hmm|mm|mhm|mmm|uh|um)\b")
_APOSTROPHES = re.compile("['\u2018\u2019\u02bc]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")
_NUMBER = re.compile(r"\b(?:100|[1-9]?[0-9])\b")

_ONES = (
    *("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    *("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen"),
    *("eighteen", "nineteen"),
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def normalize(text: str) -> str:
    """Lowercase, words-only, single-spaced form of ``text`` for WER/CER.

    Apostrophes are dropped rather than split on (``nation's`` -> ``nations``), so a
    possessive and a plural Whisper can't hear apart don't count as an error; other
    punctuation and symbols become spaces (``self-evident`` -> ``self evident``) except
    thousands separators (``1,000`` -> ``1000``).
    Standalone integers 0-100 are spelled out because Whisper writes small numbers
    either way. Idempotent: ``normalize(normalize(x)) == normalize(x)``.
    """
    text = _BRACKETED.sub(" ", text.lower())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _APOSTROPHES.sub("", text)
    text = _THOUSANDS.sub("", text)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    text = _NUMBER.sub(lambda m: _spell(int(m.group())), text)
    text = _FILLERS.sub(" ", text)
    return " ".join(text.split())


def _spell(n: int) -> str:
    if n == 100:
        return "one hundred"
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"
