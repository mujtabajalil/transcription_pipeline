"""``transcribe``: run the full pipeline on a local file, no service infrastructure.

Same probe/decode/VAD/chunking/quality guards as the worker, so it doubles as the way
to reproduce a job's result locally. stdout carries only the transcript; logs and
progress go to stderr so the output can be piped.

Exit codes: 0 ok, 1 unexpected failure, 2 usage or configuration error, 3 bad input
(``InputError``), 4 speech recognition failed (``EngineError``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from transcription.asr.factory import build_engine
from transcription.config import Settings
from transcription.domain import Transcript, TranscriptionOptions
from transcription.errors import EngineError, InputError, TranscriptionError
from transcription.formats import to_srt, to_text, to_vtt
from transcription.logging import configure_logging
from transcription.pipeline import transcribe_file

log = logging.getLogger(__name__)

_RENDERERS: dict[str, Callable[[Transcript], str]] = {
    "json": lambda t: t.model_dump_json(indent=2) + "\n",
    "text": to_text,
    "srt": to_srt,
    "vtt": to_vtt,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``transcribe`` console script; returns the exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help and on usage errors
        return exc.code if isinstance(exc.code, int) else 2
    if not Path(args.audio).is_file():
        return _usage_error(parser, f"no such file: {args.audio}")
    # Checked up front: finding out only after an hour of transcription loses the work.
    if args.output is not None and (args.output.is_dir() or not args.output.parent.is_dir()):
        return _usage_error(parser, f"cannot write output file: {args.output}")
    overrides: dict[str, Any] = {"asr_engine": args.engine, "whisper_model": args.model}
    try:
        options = TranscriptionOptions(
            language=args.language,
            prompt=args.prompt,
            split_channels=args.split_channels,
            word_timestamps=args.word_timestamps,
        )
        # Also reads TX_* env vars and .env, so a malformed value is a configuration error.
        settings = Settings(**{key: value for key, value in overrides.items() if value is not None})
    except ValidationError as exc:
        return _usage_error(parser, _describe(exc))
    # stderr: stdout carries the transcript.
    configure_logging(
        "WARNING" if args.quiet else settings.log_level, json_output=False, stream=sys.stderr
    )
    try:
        engine = build_engine(settings)
    except ValueError as exc:  # missing engine configuration, e.g. an API key
        return _usage_error(parser, str(exc))
    try:
        transcript = transcribe_file(
            args.audio,
            engine,
            options,
            settings.pipeline_config(),
            on_progress=None if args.quiet else _print_progress,
        )
        _write(_RENDERERS[args.format](transcript), args.output)
    except InputError as exc:
        return _failure(exc, 3)
    except EngineError as exc:
        return _failure(exc, 4)
    except Exception:
        log.exception("transcription failed")
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe", description="Transcribe an audio or video file to text."
    )
    parser.add_argument("audio", metavar="AUDIO", help="audio/video file in any supported format")
    parser.add_argument("--language", help="ISO 639 code (e.g. en); detected when omitted")
    parser.add_argument("--prompt", help="glossary of names/jargon to spell consistently")
    parser.add_argument(
        "--split-channels",
        action="store_true",
        help="transcribe each channel separately (one speaker per channel)",
    )
    parser.add_argument("--word-timestamps", action="store_true", help="include word timings")
    parser.add_argument("--format", choices=sorted(_RENDERERS), default="json")
    parser.add_argument(
        "-o", "--output", type=Path, metavar="FILE", help="write here instead of stdout"
    )
    parser.add_argument(
        "--engine",
        choices=get_args(Settings.model_fields["asr_engine"].annotation),
        help="overrides TX_ASR_ENGINE",
    )
    parser.add_argument("--model", help="Whisper model, overrides TX_WHISPER_MODEL")
    parser.add_argument("--quiet", action="store_true", help="only warnings and errors on stderr")
    return parser


def _print_progress(done: int, total: int) -> None:
    print(f"chunk {done}/{total}", file=sys.stderr)


def _write(rendered: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")


def _describe(exc: ValidationError) -> str:
    # pydantic's msg alone ("String should have at most 800 characters") omits the field.
    return "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
    )


def _usage_error(parser: argparse.ArgumentParser, message: str) -> int:
    print(f"{parser.prog}: error: {message}", file=sys.stderr)
    return 2


def _failure(exc: TranscriptionError, exit_code: int) -> int:
    print(f"error {exc.code}: {exc.message}", file=sys.stderr)
    return exit_code
