"""Build the evaluation set: synthetic English speech in the conditions DESIGN.md makes
claims about (paused long-form, nonstop speech, long silences, music, noise, telephony,
and no speech at all).

    uv run python -m eval.build_dataset

Speech comes from macOS ``say`` (several voices and accents) reading public-domain
texts, so references are exact and the set rebuilds without shipping anyone's
recordings. Synthetic speech is cleaner and more regular than real speech: absolute WERs
here are optimistic, so the set is for comparing front-ends, not for quoting accuracy.
Noise and music are deterministic (fixed seeds, closed-form signals) and mixed at an SNR
measured on active speech only, so pauses don't inflate it.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from transcription.domain import SAMPLE_RATE

Audio = npt.NDArray[np.float32]

EVAL_DIR = Path(__file__).resolve().parent
AUDIO_DIR = EVAL_DIR / "audio"
MANIFEST = EVAL_DIR / "manifest.jsonl"

PREFERRED_VOICES = ("Samantha", "Daniel", "Karen", "Moira")
"""US, British, Australian and Irish English; all ship with macOS."""

PARAGRAPH_PAUSE_MS = 1500
"""Long enough for VAD (500 ms min silence) to see a pause, short enough to be packed
into one chunk (2 s max gap)."""
PASSAGE_PAUSE_S = 3.0
"""Longer than the 2 s packing gap: a topic change the chunker must never pack across."""

MUSIC = (
    "0.3*exp(-5*mod(t,0.25))*sin(2*PI*220*t*pow(2,"
    "if(eq(mod(floor(4*t),4),0),0,if(eq(mod(floor(4*t),4),1),3,"
    "if(eq(mod(floor(4*t),4),2),7,12)))/12))+0.1*sin(2*PI*110*t)"
)
"""A plucked A-minor arpeggio over a bass drone: tonal, rhythmic and nothing like noise."""

# Modern spellings (``cannot``, ``battlefield``) where the period text differs, so WER
# measures recognition rather than orthography.
GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth on this continent, a new "
    "nation, conceived in Liberty, and dedicated to the proposition that all men are "
    "created equal.",
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so "
    "conceived and so dedicated, can long endure. We are met on a great battlefield of that "
    "war. We have come to dedicate a portion of that field, as a final resting place for "
    "those who here gave their lives that that nation might live. It is altogether fitting "
    "and proper that we should do this.",
    "But, in a larger sense, we cannot dedicate, we cannot consecrate, we cannot hallow, "
    "this ground. The brave men, living and dead, who struggled here, have consecrated it, "
    "far above our poor power to add or detract. The world will little note, nor long "
    "remember what we say here, but it can never forget what they did here. It is for us "
    "the living, rather, to be dedicated here to the unfinished work which they who fought "
    "here have thus far so nobly advanced. It is rather for us to be here dedicated to the "
    "great task remaining before us, that from these honored dead we take increased "
    "devotion to that cause for which they gave the last full measure of devotion, that we "
    "here highly resolve that these dead shall not have died in vain, that this nation, "
    "under God, shall have a new birth of freedom, and that government of the people, by "
    "the people, for the people, shall not perish from the earth.",
)
DECLARATION = (
    "When in the Course of human events, it becomes necessary for one people to dissolve "
    "the political bands which have connected them with another, and to assume among the "
    "powers of the earth, the separate and equal station to which the Laws of Nature and of "
    "Nature's God entitle them, a decent respect to the opinions of mankind requires that "
    "they should declare the causes which impel them to the separation.",
    "We hold these truths to be self-evident, that all men are created equal, that they are "
    "endowed by their Creator with certain unalienable Rights, that among these are Life, "
    "Liberty and the pursuit of Happiness. That to secure these rights, Governments are "
    "instituted among Men, deriving their just powers from the consent of the governed. "
    "That whenever any Form of Government becomes destructive of these ends, it is the "
    "Right of the People to alter or to abolish it, and to institute new Government, laying "
    "its foundation on such principles and organizing its powers in such form, as to them "
    "shall seem most likely to effect their Safety and Happiness.",
)
SECOND_INAUGURAL = (
    "Both parties deprecated war, but one of them would make war rather than let the nation "
    "survive, and the other would accept war rather than let it perish, and the war came.",
    "Fondly do we hope, fervently do we pray, that this mighty scourge of war may speedily "
    "pass away.",
    "With malice toward none, with charity for all, with firmness in the right as God gives "
    "us to see the right, let us strive on to finish the work we are in, to bind up the "
    "nation's wounds, to care for him who shall have borne the battle and for his widow and "
    "his orphan, to do all which may achieve and cherish a just and lasting peace among "
    "ourselves and with all nations.",
)
NORTH_WIND = (
    "The North Wind and the Sun were disputing which was the stronger, when a traveler came "
    "along wrapped in a warm cloak. They agreed that the one who first succeeded in making "
    "the traveler take his cloak off should be considered stronger than the other. Then the "
    "North Wind blew as hard as he could, but the more he blew the more closely did the "
    "traveler fold his cloak around him, and at last the North Wind gave up the attempt. "
    "Then the Sun shined out warmly, and immediately the traveler took off his cloak. And so "
    "the North Wind was obliged to confess that the Sun was the stronger of the two.",
)
TORTOISE_AND_HARE = (
    "A hare one day ridiculed the short feet and slow pace of the tortoise, who replied, "
    "laughing, though you be swift as the wind, I will beat you in a race. The hare, "
    "believing her assertion to be simply impossible, assented to the proposal, and they "
    "agreed that the fox should choose the course and fix the goal. On the day appointed "
    "for the race the two started together. The tortoise never for a moment stopped, but "
    "went on with a slow but steady pace straight to the end of the course. The hare, lying "
    "down by the wayside, fell fast asleep. At last waking up, and moving as fast as he "
    "could, he saw the tortoise had reached the goal, and was comfortably dozing after her "
    "fatigue. Slow but steady wins the race.",
)
FOX_AND_GRAPES = (
    "One hot summer day a fox was strolling through an orchard till he came to a bunch of "
    "grapes just ripening on a vine which had been trained over a lofty branch. Just the "
    "thing to quench my thirst, said he. Drawing back a few paces, he took a run and a "
    "jump, and just missed the bunch. Turning round again with a one, two, three, he jumped "
    "up, but with no greater success. Again and again he tried after the tempting morsel, "
    "but at last had to give it up, and walked away with his nose in the air, saying, I am "
    "sure they are sour. It is easy to despise what you cannot get.",
)
BOY_WHO_CRIED_WOLF = (
    "There was once a young shepherd boy who tended his sheep at the foot of a mountain near "
    "a dark forest. It was rather lonely for him all day, so he thought upon a plan by which "
    "he could get a little company and some excitement. He rushed down towards the village "
    "calling out wolf, wolf, and the villagers came out to meet him. This pleased the boy so "
    "much that a few days afterwards he tried the same trick, and again the villagers came "
    "to his help. But shortly after this a wolf actually did come out from the forest, and "
    "the boy cried out wolf, wolf, still louder than before. But this time the villagers, "
    "who had been fooled twice before, thought the boy was again deceiving them, and nobody "
    "stirred to come to his help. A liar will not be believed, even when he speaks the "
    "truth.",
)


@dataclass(frozen=True)
class Clip:
    id: str
    audio: Audio
    reference: str
    tags: list[str]


def main() -> None:
    if sys.platform != "darwin" or shutil.which("say") is None:
        sys.exit(
            "build_dataset needs macOS `say` for speech synthesis. The committed "
            "eval/audio/ and eval/manifest.jsonl are its output; use those instead."
        )
    if shutil.which("ffmpeg") is None:
        sys.exit("build_dataset needs ffmpeg on PATH")
    clips = _build(_voices())
    AUDIO_DIR.mkdir(exist_ok=True)
    for stale in AUDIO_DIR.glob("*.wav"):
        stale.unlink()
    lines = []
    for clip in clips:
        _write_wav(AUDIO_DIR / f"{clip.id}.wav", clip.audio)
        record = {
            "id": clip.id,
            "audio": f"audio/{clip.id}.wav",
            "reference": clip.reference,
            "tags": clip.tags,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        print(f"{clip.id:<16} {len(clip.audio) / SAMPLE_RATE:7.1f} s  {' '.join(clip.tags)}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(len(clip.audio) for clip in clips) / SAMPLE_RATE
    print(f"{len(clips)} clips, {total / 60:.1f} min of audio -> {MANIFEST}")


def _voices() -> list[str]:
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    english = set(re.findall(r"^(\S.*?)\s+en_[A-Z]{2}\s", listing, flags=re.MULTILINE))
    voices = [voice for voice in PREFERRED_VOICES if voice in english]
    if len(voices) < 3:
        sys.exit(
            f"need at least 3 of the voices {', '.join(PREFERRED_VOICES)} (found "
            f"{voices or 'none'}); install them under System Settings > Accessibility > "
            "Spoken Content > System voice > Manage Voices"
        )
    return voices


def _build(voices: Sequence[str]) -> list[Clip]:
    def voice(i: int) -> str:
        return voices[i % len(voices)]

    def tags(*names: str, speaker: int) -> list[str]:
        return [*names, f"voice:{voice(speaker).lower()}"]

    declaration = _speak(DECLARATION, voice(2))
    long_form = [
        _speak(GETTYSBURG, voice(0)),
        _speak(DECLARATION, voice(1)),
        _speak(SECOND_INAUGURAL, voice(3)),
    ]
    fox = _speak(FOX_AND_GRAPES, voice(3))
    wolf = _speak(BOY_WHO_CRIED_WOLF, voice(0))
    music_bed = _with_lead(_music(len(wolf) / SAMPLE_RATE + 10), wolf, lead_s=5)
    return [
        Clip("clean", declaration, _text(DECLARATION), tags("clean", speaker=2)),
        Clip(
            "long_form",
            _join(long_form, pause_s=PASSAGE_PAUSE_S),
            _text(GETTYSBURG + DECLARATION + SECOND_INAUGURAL),
            ["long_form", "clean", "voice:mixed"],
        ),
        Clip(
            "nonstop_fables",
            _nonstop(NORTH_WIND + TORTOISE_AND_HARE, voice(1)),
            _text(NORTH_WIND + TORTOISE_AND_HARE),
            tags("nonstop", "clean", speaker=1),
        ),
        Clip(
            "nonstop_speeches",
            _nonstop(GETTYSBURG + SECOND_INAUGURAL, voice(2)),
            _text(GETTYSBURG + SECOND_INAUGURAL),
            tags("nonstop", "clean", speaker=2),
        ),
        Clip(
            "silence_padded",
            _join([_silence(30), fox, _silence(60)], pause_s=0),
            _text(FOX_AND_GRAPES),
            tags("silence_padded", speaker=3),
        ),
        Clip(
            "music_bed",
            music_bed,
            _text(BOY_WHO_CRIED_WOLF),
            tags("music", speaker=0),
        ),
        *(
            Clip(
                f"{color}_snr{snr}",
                _mix(declaration, _noise(color, len(declaration)), snr),
                _text(DECLARATION),
                tags("noise", color, f"snr{snr}", speaker=2),
            )
            for color in ("white", "pink")
            for snr in (10, 0)
        ),
        Clip(
            "telephony",
            _telephone(declaration),
            _text(DECLARATION),
            tags("telephony", speaker=2),
        ),
        Clip("silence", _silence(30), "", ["no_speech", "silence"]),
        Clip(
            "noise_only",
            _at_dbfs(_noise("pink", 30 * SAMPLE_RATE), -35),
            "",
            ["no_speech", "noise"],
        ),
        Clip("music_only", _music(20), "", ["no_speech", "music"]),
    ]


def _text(paragraphs: Sequence[str]) -> str:
    return " ".join(paragraphs)


def _speak(paragraphs: Sequence[str], voice: str, *, pause_ms: int = PARAGRAPH_PAUSE_MS) -> Audio:
    """``paragraphs`` read by ``voice``, with ``pause_ms`` of silence between paragraphs."""
    text = f" [[slnc {pause_ms}]] ".join(paragraphs) if pause_ms else " ".join(paragraphs)
    with tempfile.TemporaryDirectory() as tmp:
        source, out = Path(tmp) / "text.txt", Path(tmp) / "speech.aiff"
        source.write_text(text, encoding="utf-8")
        subprocess.run(["say", "-v", voice, "-o", str(out), "-f", str(source)], check=True)
        return _ffmpeg_audio("-i", str(out))


def _nonstop(paragraphs: Sequence[str], voice: str) -> Audio:
    """Read without paragraph pauses, every pause cut to 0.1 s: VAD finds no gap to cut
    at, so the chunker has to force cuts inside speech."""
    return _filter(
        _speak(paragraphs, voice, pause_ms=0),
        "silenceremove=stop_periods=-1:stop_duration=0.1:stop_threshold=-45dB",
    )


_RAW = ("-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1")


def _ffmpeg(*args: str, stdin: bytes | None = None) -> bytes:
    """Run ffmpeg (inputs, filters and an output on ``pipe:1`` in ``args``); its stdout."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", *args], input=stdin, capture_output=True
    )
    if proc.returncode != 0:
        sys.exit(f"ffmpeg {' '.join(args)} failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def _ffmpeg_audio(*args: str, stdin: bytes | None = None) -> Audio:
    """ffmpeg output as 16 kHz mono float32."""
    return np.frombuffer(_ffmpeg(*args, *_RAW, "pipe:1", stdin=stdin), dtype=np.float32)


def _filter(audio: Audio, af: str) -> Audio:
    return _ffmpeg_audio(*_RAW, "-i", "pipe:0", "-af", af, stdin=audio.tobytes())


def _telephone(audio: Audio) -> Audio:
    """Round trip through a narrowband phone line: 300-3400 Hz, 8 kHz, G.711 mu-law."""
    ulaw = _ffmpeg(
        *_RAW,
        "-i",
        "pipe:0",
        "-af",
        "highpass=f=300,lowpass=f=3400",
        "-ar",
        "8000",
        "-f",
        "mulaw",
        "pipe:1",
        stdin=audio.tobytes(),
    )
    return _ffmpeg_audio("-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "pipe:0", stdin=ulaw)


def _noise(color: str, samples: int) -> Audio:
    seconds = samples / SAMPLE_RATE + 1  # lavfi durations round; trim to the exact length
    return _ffmpeg_audio(
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=color={color}:amplitude=0.5:seed=7:d={seconds:.3f}:r={SAMPLE_RATE}",
    )[:samples]


def _music(seconds: float) -> Audio:
    return _ffmpeg_audio("-f", "lavfi", "-i", f"aevalsrc='{MUSIC}':s={SAMPLE_RATE}:d={seconds:.3f}")


def _silence(seconds: float) -> Audio:
    return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)


def _join(parts: Sequence[Audio], *, pause_s: float) -> Audio:
    gap = _silence(pause_s)
    pieces = [piece for part in parts for piece in (gap, part)][1:]
    return np.concatenate(pieces).astype(np.float32)


def _with_lead(bed: Audio, speech: Audio, *, lead_s: float) -> Audio:
    """``speech`` over ``bed`` starting ``lead_s`` in, the bed 10 dB below the speech."""
    start = round(lead_s * SAMPLE_RATE)
    padded = np.zeros_like(bed)
    padded[start : start + len(speech)] = speech
    return _mix(padded, bed, 10)


def _active_power(audio: Audio) -> float:
    """Mean power of 20 ms frames within 40 dB of the loudest: speech, not the pauses."""
    frame = SAMPLE_RATE // 50
    frames = audio[: len(audio) // frame * frame].astype(np.float64).reshape(-1, frame)
    power = np.square(frames).mean(axis=1)
    return float(power[power > power.max() * 1e-4].mean())


def _mix(speech: Audio, noise: Audio, snr_db: float) -> Audio:
    noise_power = float(np.square(noise.astype(np.float64)).mean())
    gain = math.sqrt(_active_power(speech) / (noise_power * 10 ** (snr_db / 10)))
    return (speech + noise * gain).astype(np.float32)


def _at_dbfs(audio: Audio, dbfs: float) -> Audio:
    rms = float(np.sqrt(np.square(audio.astype(np.float64)).mean()))
    return (audio * (10 ** (dbfs / 20) / rms)).astype(np.float32)


def _write_wav(path: Path, audio: Audio) -> None:
    """16-bit PCM WAV; peak-limited to -0.4 dBFS by uniform scaling (keeps the SNR)."""
    peak = float(np.abs(audio).max(initial=0.0))
    scaled = audio / peak * 0.95 if peak > 0.95 else audio
    samples = np.round(scaled * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())


if __name__ == "__main__":
    main()
