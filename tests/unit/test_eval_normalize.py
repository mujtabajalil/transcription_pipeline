"""WER normalisation and error localisation of the eval harness (eval/)."""

from __future__ import annotations

import jiwer
import pytest
from eval.normalize import normalize
from eval.run import Output, Token, boundaries, covered_seconds, error_spans, score, tokens

from transcription.domain import Segment, Word

CASES = [
    ("Four score and SEVEN years ago,", "four score and seven years ago"),
    ("  many\t spaces\n\nhere  ", "many spaces here"),
    ("the nation's wounds", "the nations wounds"),
    ("nation\u2019s", "nations"),
    ("self-evident", "self evident"),
    ("twenty-one", "twenty one"),
    ("He said: \u201cI\u2019m sure!\u201d", "he said im sure"),
    ("[Music] hello (laughs) <unk>", "hello"),
    ("uh, so um, hmm yes", "so yes"),
    ("caf\u00e9 na\u00efve", "cafe naive"),
    ("0 7 13 20 21 99 100", "zero seven thirteen twenty twenty one ninety nine one hundred"),
    ("101 007 1st", "101 007 1st"),
    ("1,000 men", "1000 men"),
    ("40.", "forty"),
    ("", ""),
    ("?!...", ""),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize("text", [text for text, _ in CASES])
def test_normalize_is_idempotent(text: str) -> None:
    assert normalize(normalize(text)) == normalize(text)


def test_digits_and_words_normalise_alike() -> None:
    assert normalize("Turning round with a 1, 2, 3") == normalize(
        "turning round with a one, two, three"
    )


# --- error localisation ----------------------------------------------------------------


def test_tokens_normalise_each_word_and_keep_its_absolute_time() -> None:
    segment = Segment(
        start=1.0,
        end=2.0,
        text=" Seven, years-ago.",
        words=[Word(start=1.0, end=1.4, text=" 7,"), Word(start=1.5, end=2.0, text=" years-ago.")],
    )

    out = tokens([segment], offset=30.0)

    assert out == [
        Token("seven", 31.0, 31.4),
        Token("years", 31.5, 32.0),
        Token("ago", 31.5, 32.0),
    ]


def test_tokens_without_word_timings_use_the_segment_span() -> None:
    out = tokens([Segment(start=2.0, end=3.0, text="Hello there.")])

    assert out == [Token("hello", 2.0, 3.0), Token("there", 2.0, 3.0)]


def test_tokens_drop_words_that_normalise_to_nothing() -> None:
    segment = Segment(start=0, end=1, text="uh", words=[Word(start=0, end=1, text=" uh,")])

    assert tokens([segment]) == []


@pytest.mark.parametrize(
    ("spans", "expected"),
    [
        ([], []),
        ([(0.0, 30.0)], []),
        ([(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)], [30.0, 60.0]),
        ([(0.0, 21.0), (22.3, 48.8)], [21.0, 22.3]),
    ],
)
def test_boundaries_are_internal_chunk_edges(
    spans: list[tuple[float, float]], expected: list[float]
) -> None:
    assert boundaries(spans) == expected


def test_covered_seconds_merges_overlapping_windows_and_clips_to_the_audio() -> None:
    # [0, 1.5] clipped at 0, [20, 22.5] from two overlapping windows, [28.5, 30] clipped.
    assert covered_seconds([0.5, 21.0, 21.5, 29.5], 1.0, 30.0) == pytest.approx(5.5)
    assert covered_seconds([], 1.0, 30.0) == 0.0


def test_error_spans_time_each_error_kind() -> None:
    hyp = [Token("a", 0.0, 0.5), Token("x", 1.0, 1.5), Token("c", 2.0, 2.5), Token("z", 9, 9.5)]
    alignment = jiwer.process_words("q a b c d e", "a x c z").alignments[0]

    spans = error_spans(alignment, hyp, duration=10.0)

    # q deleted before the first word, b -> x substituted, d e -> z: one substitution
    # and one deletion; jiwer pairs them in order.
    assert sorted(spans) == sorted([(0.0, 0.0), (1.0, 1.5), (9.0, 9.5), (9.5, 10.0)])


def test_score_counts_a_deletion_at_a_seam_as_a_seam_error() -> None:
    # "brought" was lost at the 30 s cut; "forth" is heard well after it.
    hyp = [Token("fathers", 28.0, 29.5), Token("forth", 31.5, 32.0), Token("on", 40.0, 40.2)]

    result = score(
        "fathers brought forth on",
        Output(tokens=hyp, seams=[30.0]),
        audio_s=60.0,
        seconds=6.0,
    )

    assert (result.deletions, result.substitutions, result.insertions) == (1, 0, 0)
    assert result.seam_errors == 1
    assert result.near_seam_s == pytest.approx(2.0)


def test_score_ignores_errors_far_from_seams() -> None:
    hyp = [Token("fathers", 1.0, 1.5), Token("broke", 2.0, 2.5), Token("forth", 3.0, 3.5)]

    result = score("fathers brought forth", Output(hyp, seams=[30.0]), audio_s=60, seconds=1)

    assert (result.substitutions, result.seam_errors) == (1, 0)


def test_score_with_empty_reference_counts_every_word_as_hallucinated() -> None:
    hyp = [Token("thank", 3.0, 3.2), Token("you", 3.2, 3.4)]

    result = score("", Output(hyp, seams=[]), audio_s=30.0, seconds=3.0)

    assert (result.ref_words, result.hyp_words, result.insertions) == (0, 2, 2)
    assert result.errors == 2
    assert result.char_errors == 0


def test_score_with_empty_hypothesis_is_all_deletions() -> None:
    result = score("four score", Output([], seams=[]), audio_s=5.0, seconds=1.0)

    assert (result.deletions, result.errors, result.ref_words) == (2, 2, 2)
    assert result.char_errors == len("four score")
