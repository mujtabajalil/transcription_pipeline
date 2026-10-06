"""Language-code normalisation.

Clients, hosted engines and Whisper disagree on spelling: ElevenLabs answers ``eng``,
browsers send ``en-US``, Whisper only accepts ``en``. Everything that crosses an engine
boundary goes through :func:`normalize_language` so a pinned language means the same
thing to every engine and to the persisted job.
"""

from __future__ import annotations

_TO_WHISPER: dict[str, str] = {
    # ISO 639-3 / 639-2 (terminology and bibliographic forms) -> Whisper's ISO 639-1.
    # Covers every language Whisper knows by a two-letter code, because hosted engines
    # answer in 639-3 and the detected language is pinned for later chunks, which
    # may fall back to Whisper.
    "afr": "af",
    "amh": "am",
    "ara": "ar",
    "asm": "as",
    "aze": "az",
    "bak": "ba",
    "bel": "be",
    "ben": "bn",
    "bod": "bo",
    "tib": "bo",
    "bos": "bs",
    "bre": "br",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cze": "cs",
    "chi": "zh",
    "cmn": "zh",
    "zho": "zh",
    "cym": "cy",
    "wel": "cy",
    "dan": "da",
    "deu": "de",
    "ger": "de",
    "ell": "el",
    "gre": "el",
    "eng": "en",
    "est": "et",
    "eus": "eu",
    "baq": "eu",
    "fao": "fo",
    "fas": "fa",
    "per": "fa",
    "fin": "fi",
    "fra": "fr",
    "fre": "fr",
    "glg": "gl",
    "guj": "gu",
    "hat": "ht",
    "hau": "ha",
    "heb": "he",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "hye": "hy",
    "arm": "hy",
    "ind": "id",
    "isl": "is",
    "ice": "is",
    "ita": "it",
    "jav": "jw",
    "jpn": "ja",
    "kan": "kn",
    "kat": "ka",
    "geo": "ka",
    "kaz": "kk",
    "khm": "km",
    "kor": "ko",
    "lao": "lo",
    "lat": "la",
    "lav": "lv",
    "lin": "ln",
    "lit": "lt",
    "ltz": "lb",
    "mal": "ml",
    "mar": "mr",
    "mkd": "mk",
    "mac": "mk",
    "mlg": "mg",
    "mlt": "mt",
    "mon": "mn",
    "mri": "mi",
    "mao": "mi",
    "msa": "ms",
    "may": "ms",
    "mya": "my",
    "bur": "my",
    "nep": "ne",
    "nld": "nl",
    "dut": "nl",
    "nno": "nn",
    "nob": "no",
    "nor": "no",
    "oci": "oc",
    "pan": "pa",
    "pol": "pl",
    "por": "pt",
    "pus": "ps",
    "ron": "ro",
    "rum": "ro",
    "rus": "ru",
    "san": "sa",
    "sin": "si",
    "slk": "sk",
    "slo": "sk",
    "slv": "sl",
    "sna": "sn",
    "snd": "sd",
    "som": "so",
    "spa": "es",
    "sqi": "sq",
    "alb": "sq",
    "srp": "sr",
    "sun": "su",
    "swa": "sw",
    "swh": "sw",
    "swe": "sv",
    "tam": "ta",
    "tat": "tt",
    "tel": "te",
    "tgk": "tg",
    "tgl": "tl",
    "fil": "tl",
    "tha": "th",
    "tuk": "tk",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "uzb": "uz",
    "vie": "vi",
    "yid": "yi",
    "yor": "yo",
    # ISO 639-1 codes Whisper spells differently (legacy or macrolanguage forms).
    "jv": "jw",
    "iw": "he",
    "nb": "no",
}


def normalize_language(code: str | None) -> str | None:
    """Map ``code`` to the spelling Whisper uses.

    Locale forms (``en-US``, ``en_GB``) lose their region, ISO 639-3 codes with an ISO
    639-1 equivalent are shortened (``eng`` -> ``en``, ``cmn`` -> ``zh``), and codes
    Whisper only knows in 639-3 (``yue``, ``haw``) are kept. Unknown codes pass through
    lower-cased so the engine, not this table, decides whether they are valid. ``None``
    and blank strings mean "detect" and return ``None``.
    """
    if code is None:
        return None
    primary = code.strip().lower().replace("_", "-").partition("-")[0]
    if not primary:
        return None
    return _TO_WHISPER.get(primary, primary)
