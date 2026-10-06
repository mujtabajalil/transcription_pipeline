"""Measure the claims DESIGN.md makes about the audio front-end.

    uv run python -m eval.run --model small --configs pipeline,naive30,denoise [--limit N]

Configs, all sharing one Whisper model loaded once:

* ``pipeline``: ``transcription.pipeline.transcribe_file``, i.e. VAD, pause-aligned
  chunks of at most 30 s, quality guards. What the service runs.
* ``naive30``: decode, cut into fixed 30 s windows with no VAD, transcribe each window
  independently, concatenate. The usual first attempt, and the baseline the pipeline's
  complexity has to beat.
* ``denoise``: the pipeline on audio pre-filtered by ffmpeg ``afftdn``, i.e. a classic
  spectral-subtraction enhancement front-end, which DESIGN.md deliberately leaves out.

Per clip and per config: WER and CER (jiwer, after ``eval.normalize``), hallucinated
words on clips whose reference is empty, word errors within 1 s of a chunk seam, and
speed. Results go to ``eval/results/<UTC timestamp>.{md,json}``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer
import numpy as np

from eval.normalize import normalize
from transcription.asr.base import ASREngine
from transcription.asr.faster_whisper import FasterWhisperEngine
from transcription.audio.decode import decode_to_pcm, open_pcm, to_float32
from transcription.domain import (
    SAMPLE_RATE,
    PipelineConfig,
    Segment,
    TranscriptionOptions,
    Word,
)
from transcription.logging import configure_logging
from transcription.pipeline import InMemoryCheckpoint, transcribe_file

EVAL_DIR = Path(__file__).resolve().parent
MANIFEST = EVAL_DIR / "manifest.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

NAIVE_WINDOW_S = 30
SEAM_TOLERANCE_S = 1.0
DENOISE_FILTER = "volume=-40dB,afftdn=nr=25:nf=-70:tn=1,volume=40dB"
"""afftdn models a noise floor between -80 and -20 dB; at its defaults it removed under
0.5 dB of the 0-10 dB SNR noise in this set. Attenuating first puts that noise inside
its range (10 dB SNR white noise drops ~20 dB in pauses), and the gain is restored after."""

CONFIG = PipelineConfig()
OPTIONS = TranscriptionOptions(word_timestamps=True)
"""Word timings locate errors relative to seams; the decoded text is the same either way."""


@dataclass(frozen=True)
class Token:
    """One normalised hypothesis word and where it was heard."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Clip:
    id: str
    path: Path
    reference: str
    tags: list[str]


@dataclass(frozen=True)
class Output:
    tokens: list[Token]
    seams: list[float]
    """Internal chunk boundaries, in seconds."""


@dataclass(frozen=True)
class Score:
    """Raw counts, so clips aggregate by summing rather than averaging rates."""

    audio_s: float
    seconds: float
    ref_words: int
    hyp_words: int
    substitutions: int
    deletions: int
    insertions: int
    ref_chars: int
    char_errors: int
    seams: int
    seam_errors: int
    near_seam_s: float
    hypothesis: str

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


Runner = Callable[[Path, ASREngine], Output]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    configs = [name.strip() for name in args.configs.split(",") if name.strip()]
    if unknown := sorted(set(configs) - RUNNERS.keys()):
        sys.exit(f"unknown config(s) {', '.join(unknown)}; choose from {', '.join(RUNNERS)}")
    clips = _load_manifest(MANIFEST)[: args.limit]
    configure_logging("WARNING", json_output=False)
    engine = FasterWhisperEngine(args.model, device=args.device, compute_type=args.compute_type)
    # The first decode pays one-off allocation costs; keep them out of the first clip's speed.
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en", prompt=None)
    scores: dict[str, dict[str, Score]] = {clip.id: {} for clip in clips}
    for n, clip in enumerate(clips, start=1):
        for config in configs:
            result = _evaluate(clip, RUNNERS[config], engine)
            scores[clip.id][config] = result
            print(f"[{n}/{len(clips)}] {clip.id:<16} {config:<9} {_cell(result)}", file=sys.stderr)
    meta = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "configs": configs,
        "clips": len(clips),
        "machine": f"{platform.platform()} {platform.machine()}",
        "denoise_filter": DENOISE_FILTER,
    }
    stem = RESULTS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(_to_json(meta, clips, scores, configs), indent=2) + "\n", encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(_to_markdown(meta, clips, scores, configs))
    print(f"wrote {stem}.md and .json", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run", description="WER, hallucination and seam errors per front-end config."
    )
    parser.add_argument("--model", default="small", help="faster-whisper model name")
    parser.add_argument("--configs", default="pipeline,naive30,denoise")
    parser.add_argument("--limit", type=int, help="only the first N clips of the manifest")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="int8")
    return parser


def _load_manifest(path: Path) -> list[Clip]:
    clips = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            clips.append(
                Clip(
                    record["id"], path.parent / record["audio"], record["reference"], record["tags"]
                )
            )
    return clips


# --- configs -----------------------------------------------------------------------------


def run_pipeline(path: Path, engine: ASREngine) -> Output:
    checkpoint = InMemoryCheckpoint()
    transcript = transcribe_file(path, engine, OPTIONS, CONFIG, checkpoint=checkpoint)
    chunks = checkpoint.plan.chunks if checkpoint.plan else []
    return Output(
        tokens=tokens(transcript.segments),
        seams=boundaries([(chunk.start_s, chunk.end_s) for chunk in chunks]),
    )


def run_naive30(path: Path, engine: ASREngine) -> Output:
    window = NAIVE_WINDOW_S * SAMPLE_RATE
    found: list[Token] = []
    with tempfile.TemporaryDirectory() as tmp:
        pcm_path = Path(tmp) / "audio.pcm"
        decode_to_pcm(
            path,
            pcm_path,
            max_seconds=CONFIG.max_audio_seconds,
            timeout_s=CONFIG.ffmpeg_timeout_s,
        )
        pcm = open_pcm(pcm_path)
        for start in range(0, len(pcm), window):
            result = engine.transcribe(
                to_float32(pcm[start : start + window]),
                language=None,
                prompt=None,
                word_timestamps=True,
            )
            found += tokens(result.segments, offset=start / SAMPLE_RATE)
        seams = [start / SAMPLE_RATE for start in range(window, len(pcm), window)]
    return Output(tokens=found, seams=seams)


def run_denoise(path: Path, engine: ASREngine) -> Output:
    with tempfile.TemporaryDirectory() as tmp:
        denoised = Path(tmp) / "denoised.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-af",
                DENOISE_FILTER,
                str(denoised),
            ],
            check=True,
        )
        return run_pipeline(denoised, engine)


RUNNERS: dict[str, Runner] = {
    "pipeline": run_pipeline,
    "naive30": run_naive30,
    "denoise": run_denoise,
}


# --- scoring -----------------------------------------------------------------------------


def tokens(segments: Iterable[Segment], *, offset: float = 0.0) -> list[Token]:
    """Normalised hypothesis words with absolute times.

    Normalising word by word keeps each token's timing; a segment without word timings
    lends its own span to all of its words.
    """
    out = []
    for segment in segments:
        words = segment.words or [Word(start=segment.start, end=segment.end, text=segment.text)]
        for word in words:
            out += [
                Token(text, offset + word.start, offset + word.end)
                for text in normalize(word.text).split()
            ]
    return out


def boundaries(spans: Sequence[tuple[float, float]]) -> list[float]:
    """Where one chunk ends and the next begins; two points when a gap separates them."""
    return sorted({t for (_, end), (start, _) in itertools.pairwise(spans) for t in (end, start)})


def error_spans(
    alignment: Iterable[Any], hyp: Sequence[Token], duration: float
) -> list[tuple[float, float]]:
    """A time span for every word error in a jiwer alignment.

    Substituted and inserted words have their own timing. A deleted word was never
    heard, so it gets the gap between the hypothesis words around it.
    """
    spans: list[tuple[float, float]] = []
    for op in alignment:
        if op.type == "equal":
            continue
        if op.type == "delete":
            at = op.hyp_start_idx
            start = hyp[at - 1].end if at > 0 else 0.0
            end = hyp[at].start if at < len(hyp) else duration
            spans += [(start, end)] * (op.ref_end_idx - op.ref_start_idx)
        else:
            spans += [(t.start, t.end) for t in hyp[op.hyp_start_idx : op.hyp_end_idx]]
    return spans


def near_any(span: tuple[float, float], seams: Iterable[float], tolerance: float) -> bool:
    start, end = span
    return any(seam - tolerance <= end and start <= seam + tolerance for seam in seams)


def covered_seconds(seams: Sequence[float], tolerance: float, duration: float) -> float:
    """Audio time within ``tolerance`` of a seam (overlapping windows counted once)."""
    covered = 0.0
    reach = 0.0
    for seam in sorted(seams):
        start, end = max(seam - tolerance, reach, 0.0), min(seam + tolerance, duration)
        covered += max(0.0, end - start)
        reach = max(reach, end)
    return covered


def score(reference: str, output: Output, *, audio_s: float, seconds: float) -> Score:
    ref = normalize(reference)
    hyp = " ".join(token.text for token in output.tokens)
    words = jiwer.process_words(ref, hyp) if ref else None
    chars = jiwer.process_characters(ref, hyp) if ref else None
    spans = error_spans(words.alignments[0], output.tokens, audio_s) if words else []
    return Score(
        audio_s=audio_s,
        seconds=seconds,
        ref_words=len(ref.split()),
        hyp_words=len(output.tokens),
        substitutions=words.substitutions if words else 0,
        deletions=words.deletions if words else 0,
        insertions=words.insertions if words else len(output.tokens),
        ref_chars=len(ref),
        char_errors=chars.substitutions + chars.deletions + chars.insertions if chars else 0,
        seams=len(output.seams),
        seam_errors=sum(near_any(span, output.seams, SEAM_TOLERANCE_S) for span in spans),
        near_seam_s=covered_seconds(output.seams, SEAM_TOLERANCE_S, audio_s),
        hypothesis=hyp,
    )


def _evaluate(clip: Clip, runner: Runner, engine: ASREngine) -> Score:
    started = time.perf_counter()
    output = runner(clip.path, engine)
    seconds = time.perf_counter() - started
    return score(clip.reference, output, audio_s=_duration(clip.path), seconds=seconds)


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


# --- reporting ---------------------------------------------------------------------------


def aggregate(scores: Iterable[Score]) -> dict[str, Any]:
    """Corpus-level rates: errors summed over clips, divided by summed reference size."""
    items = list(scores)
    speech = [s for s in items if s.ref_words]
    silent = [s for s in items if not s.ref_words]
    ref_words = sum(s.ref_words for s in speech)
    errors = sum(s.errors for s in speech)
    seam_errors = sum(s.seam_errors for s in speech)
    audio_s = sum(s.audio_s for s in items)
    return {
        "clips": len(items),
        "wer": _ratio(errors, ref_words),
        "cer": _ratio(sum(s.char_errors for s in speech), sum(s.ref_chars for s in speech)),
        "substitutions": sum(s.substitutions for s in speech),
        "deletions": sum(s.deletions for s in speech),
        "insertions": sum(s.insertions for s in speech),
        "ref_words": ref_words,
        "hallucinated_words": sum(s.hyp_words for s in silent) if silent else None,
        "seams": sum(s.seams for s in speech),
        "seam_errors": seam_errors,
        "seam_error_share": _ratio(seam_errors, errors),
        "near_seam_audio_share": _ratio(
            sum(s.near_seam_s for s in speech), sum(s.audio_s for s in speech)
        ),
        "x_realtime": _ratio(audio_s, sum(s.seconds for s in items)),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _to_json(
    meta: dict[str, Any],
    clips: Sequence[Clip],
    scores: dict[str, dict[str, Score]],
    configs: Sequence[str],
) -> dict[str, Any]:
    return {
        "meta": meta,
        "aggregate": {c: aggregate(scores[clip.id][c] for clip in clips) for c in configs},
        "by_tag": {
            tag: {c: aggregate(scores[clip.id][c] for clip in tagged) for c in configs}
            for tag, tagged in _by_tag(clips).items()
        },
        "clips": [
            {
                "id": clip.id,
                "tags": clip.tags,
                "reference": normalize(clip.reference),
                "results": {
                    c: {
                        **asdict(scores[clip.id][c]),
                        "wer": _ratio(scores[clip.id][c].errors, scores[clip.id][c].ref_words),
                    }
                    for c in configs
                },
            }
            for clip in clips
        ],
    }


def _by_tag(clips: Sequence[Clip]) -> dict[str, list[Clip]]:
    tags: dict[str, list[Clip]] = {}
    for clip in clips:
        for tag in clip.tags:
            tags.setdefault(tag, []).append(clip)
    return dict(sorted(tags.items()))


def _to_markdown(
    meta: dict[str, Any],
    clips: Sequence[Clip],
    scores: dict[str, dict[str, Score]],
    configs: Sequence[str],
) -> str:
    totals = {c: aggregate(scores[clip.id][c] for clip in clips) for c in configs}
    lines = [
        f"# Eval {meta['created']}",
        "",
        f"Model `{meta['model']}` ({meta['device']}, {meta['compute_type']}) on "
        f"{meta['machine']}; {meta['clips']} clips. Denoise filter: `{meta['denoise_filter']}`.",
        "",
        "## Aggregate",
        "",
        "| config | WER | CER | sub / del / ins | hallucinated words (no-speech clips) | seams "
        "| errors ≤1 s from a seam | share of errors | share of audio | speed (x realtime) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, t in totals.items():
        lines.append(
            f"| {c} | {_pct(t['wer'])} | {_pct(t['cer'])} "
            f"| {t['substitutions']} / {t['deletions']} / {t['insertions']} "
            f"| {_num(t['hallucinated_words'])} | {t['seams']} | {t['seam_errors']} "
            f"| {_pct(t['seam_error_share'])} | {_pct(t['near_seam_audio_share'])} "
            f"| {_num(t['x_realtime'], '{:.1f}')} |"
        )
    lines += ["", "## By tag (WER; hallucinated words where the reference is empty)", ""]
    lines += [f"| tag | clips | {' | '.join(configs)} |", "|---|---|" + "---|" * len(configs)]
    for tag, tagged in _by_tag(clips).items():
        cells = [_cell_aggregate(aggregate(scores[clip.id][c] for clip in tagged)) for c in configs]
        lines.append(f"| {tag} | {len(tagged)} | {' | '.join(cells)} |")
    lines += ["", "## Per clip", ""]
    lines += [f"| clip | audio s | {' | '.join(configs)} |", "|---|---|" + "---|" * len(configs)]
    for clip in clips:
        row = scores[clip.id]
        audio_s = next(iter(row.values())).audio_s
        cells = [_cell(row[c]) for c in configs]
        lines.append(f"| {clip.id} | {audio_s:.1f} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def _cell(s: Score) -> str:
    if not s.ref_words:
        return f"{s.hyp_words} words"
    return f"{_pct(s.errors / s.ref_words)} ({s.seam_errors} errors near {s.seams} seams)"


def _cell_aggregate(t: dict[str, Any]) -> str:
    parts = [] if t["wer"] is None else [_pct(t["wer"])]
    if t["hallucinated_words"] is not None:
        parts.append(f"{t['hallucinated_words']} words on no-speech")
    return "; ".join(parts)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | None, fmt: str = "{}") -> str:
    return "n/a" if value is None else fmt.format(value)


if __name__ == "__main__":
    main()
