from __future__ import annotations

import pytest

from transcription.asr.languages import normalize_language


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "en"),
        ("eng", "en"),
        ("EN", "en"),
        (" en ", "en"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("pt-BR", "pt"),
        ("zh-Hant-TW", "zh"),
        ("cmn", "zh"),
        ("zho", "zh"),
        ("chi", "zh"),
        ("spa", "es"),
        ("deu", "de"),
        ("ger", "de"),
        ("fra", "fr"),
        ("jpn", "ja"),
        ("fil", "tl"),
        ("nob", "no"),
        ("nb", "no"),
        ("jv", "jw"),
        ("jav", "jw"),
        ("iw", "he"),
        ("heb", "he"),
        ("sqi", "sq"),
        ("alb", "sq"),
    ],
)
def test_maps_to_whisper_codes(code: str, expected: str) -> None:
    assert normalize_language(code) == expected


@pytest.mark.parametrize("code", ["yue", "haw", "yue-HK"])
def test_whisper_only_639_3_codes_are_kept(code: str) -> None:
    assert normalize_language(code) == code[:3]


def test_unknown_codes_pass_through_lower_cased() -> None:
    assert normalize_language("XYZ") == "xyz"
    assert normalize_language("Klingon") == "klingon"


@pytest.mark.parametrize("code", [None, "", "   ", "-US"])
def test_missing_language_means_detect(code: str | None) -> None:
    assert normalize_language(code) is None


def test_every_mapping_targets_a_whisper_language() -> None:
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    from transcription.asr.languages import _TO_WHISPER

    assert set(_TO_WHISPER.values()) <= set(_LANGUAGE_CODES)


def test_every_two_letter_whisper_language_has_a_639_3_spelling() -> None:
    """A hosted engine detects in 639-3; that code is pinned and may reach Whisper."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    from transcription.asr.languages import _TO_WHISPER

    three_letter_targets = {v for k, v in _TO_WHISPER.items() if len(k) == 3}
    assert {c for c in _LANGUAGE_CODES if len(c) == 2} <= three_letter_targets
