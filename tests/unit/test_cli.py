from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fakes import FakeEngine
from transcription import cli
from transcription.config import Settings
from transcription.domain import Transcript


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """main() reconfigures the root logger and reads TX_* settings from the env."""
    monkeypatch.setenv("TX_CHUNK_RETRY_BASE_DELAY_S", "0")
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def use_engine(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine) -> list[Settings]:
    built: list[Settings] = []

    def build(settings: Settings) -> FakeEngine:
        built.append(settings)
        return engine

    monkeypatch.setattr(cli, "build_engine", build)
    return built


@pytest.fixture
def gaps(samples_dir: Path) -> str:
    return str(samples_dir / "gaps.mp3")


def test_json_to_stdout_with_progress_on_stderr(
    gaps: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_engine(monkeypatch, FakeEngine())

    assert cli.main([gaps]) == 0

    out, err = capsys.readouterr()
    transcript = Transcript.model_validate(json.loads(out))
    assert transcript.text == "chunk 0 chunk 1 chunk 2"
    assert out.startswith('{\n  "language": "en"')
    assert [line for line in err.splitlines() if line.startswith("chunk ")] == [
        "chunk 0/3",
        "chunk 1/3",
        "chunk 2/3",
        "chunk 3/3",
    ]
    assert "INFO transcription.pipeline transcription finished" in err


def test_srt_to_file_in_utf8(
    gaps: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_engine(monkeypatch, FakeEngine(prefix="café"))
    output = tmp_path / "out.srt"

    assert cli.main([gaps, "--format", "srt", "-o", str(output), "--quiet"]) == 0

    srt = output.read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,")
    assert srt.count(" --> ") == 3
    assert "café 2\n" in srt
    assert capsys.readouterr() == ("", "")


def test_options_and_engine_overrides_reach_the_pipeline(
    samples_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = FakeEngine()
    built = use_engine(monkeypatch, engine)
    argv = [
        str(samples_dir / "call_stereo.wav"),
        "--split-channels",
        "--language=de",
        "--prompt=Acme",
        "--format=text",
        "--engine=elevenlabs",
        "--model=tiny",
    ]

    assert cli.main(argv) == 0

    assert capsys.readouterr().out == "[Speaker 1] chunk 0\n[Speaker 2] chunk 1\n"
    assert [(c["language"], c["prompt"]) for c in engine.calls] == [("de", "Acme")] * 2
    (settings,) = built
    assert (settings.asr_engine, settings.whisper_model) == ("elevenlabs", "tiny")


def test_bad_input_exits_3(
    samples_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_engine(monkeypatch, FakeEngine())

    assert cli.main([str(samples_dir / "text.wav")]) == 3

    out, err = capsys.readouterr()
    assert out == ""
    assert "error unsupported_media_type: unsupported or corrupt file" in err


def test_engine_failure_exits_4(
    gaps: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_engine(monkeypatch, FakeEngine(fail_times=1_000))

    assert cli.main([gaps, "--quiet"]) == 4

    out, err = capsys.readouterr()
    assert out == ""
    assert err.endswith("error asr_engine_error: boom\n")


def test_unexpected_failure_exits_1(
    gaps: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_engine(monkeypatch, FakeEngine(fail_times=1, error=RuntimeError("bug")))

    assert cli.main([gaps, "--quiet"]) == 1

    err = capsys.readouterr().err
    assert "transcription failed" in err
    assert "RuntimeError: bug" in err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--format=docx", "x.mp3"],
        ["no-such-file.mp3"],
        ["{gaps}", "--language=english!"],
        ["{gaps}", "--prompt=" + "x" * 801],
        ["{gaps}", "-o", "{gaps}/out.srt"],  # output's parent is not a directory
    ],
)
def test_usage_errors_exit_2(
    argv: list[str],
    gaps: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = FakeEngine()
    use_engine(monkeypatch, engine)

    assert cli.main([arg.format(gaps=gaps) for arg in argv]) == 2

    assert "error" in capsys.readouterr().err
    assert engine.calls == []


def test_malformed_environment_setting_exits_2_naming_the_field(
    gaps: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = FakeEngine()
    use_engine(monkeypatch, engine)
    monkeypatch.setenv("TX_CHUNK_MAX_RETRIES", "three")

    assert cli.main([gaps]) == 2

    assert "transcribe: error: chunk_max_retries: Input should be a valid integer" in (
        capsys.readouterr().err
    )
    assert engine.calls == []


def test_missing_engine_configuration_exits_2(
    gaps: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TX_ELEVENLABS_API_KEY", raising=False)

    assert cli.main([gaps, "--engine=elevenlabs"]) == 2

    assert "TX_ELEVENLABS_API_KEY is required" in capsys.readouterr().err


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "usage: transcribe" in capsys.readouterr().out
