# Front-end evaluation

DESIGN.md makes three empirical claims about the audio front-end:

1. VAD plus pause-aligned chunking beats naive fixed 30 s windows at chunk seams.
2. A denoising front-end doesn't help Whisper and can hurt it.
3. VAD stops Whisper from hallucinating over silence and music.

This directory measures them on a small, reproducible dataset.

```bash
uv run python -m eval.build_dataset   # macOS only: regenerates eval/audio + manifest.jsonl
uv run python -m eval.run --model small --configs pipeline,naive30,denoise [--limit N]
```

`eval.run` loads the model once and writes `eval/results/<UTC timestamp>.md` and `.json`.
`eval/results/` is gitignored. The JSON has every normalised hypothesis, so you can read
the errors behind each number.

## Dataset

The speech is macOS `say` reading public-domain texts in four voices: Samantha (US),
Daniel (GB), Karen (AU) and Moira (IE). The texts are the Gettysburg Address, the
Declaration of Independence preamble, an excerpt from Lincoln's second inaugural, and
Aesop fables. That makes the references exact and the set reproducible. Noise and music
come from ffmpeg (fixed seed and a closed-form signal), mixed in numpy at an SNR measured
over active speech only. Total: 14 clips, 16.1 min, 30 MB of 16 kHz mono WAV.

| clip | s | condition |
|---|---|---|
| `clean` | 57.5 | Declaration, Karen. The base clip for the noise and telephony variants |
| `long_form` | 187.8 | Three passages, three voices, 1.5 s pauses between paragraphs and 3 s between passages |
| `nonstop_fables` | 68.5 | Two fables with every pause cut to 0.1 s, so VAD finds nowhere to cut |
| `nonstop_speeches` | 109.4 | Gettysburg plus the inaugural excerpt, also nonstop |
| `silence_padded` | 125.2 | 30 s of digital silence, a fable, then 60 s of silence |
| `music_bed` | 51.8 | A fable over a plucked arpeggio 10 dB below the speech, with 5 s of music before and after |
| `white_snr10`, `white_snr0`, `pink_snr10`, `pink_snr0` | 57.5 | `clean` plus white or pink noise at 10 dB and 0 dB SNR |
| `telephony` | 57.5 | `clean` band-passed to 300-3400 Hz, run through 8 kHz G.711 mu-law and back |
| `silence`, `noise_only`, `music_only` | 30 / 30 / 20 | No speech. The reference is empty |

## Configs

* **pipeline**: `transcription.pipeline.transcribe_file`, exactly what the worker runs
  (default `PipelineConfig`, `word_timestamps=True`).
* **naive30**: decode, cut into fixed 30 s windows with no VAD, transcribe each window
  independently with the same engine and decoding settings, concatenate. No quality
  guards. Language is detected per window, like a plain `model.transcribe(window)` loop.
* **denoise**: the pipeline on audio pre-filtered with
  `volume=-40dB,afftdn=nr=25:nf=-70:tn=1,volume=40dB`. At its default settings,
  `afftdn` removed less than 0.5 dB of this set's noise, because it models a noise floor
  between -80 and -20 dB. Attenuating the audio first brings the noise into that range,
  and the gain is restored afterwards. With this gain staging it takes about 20 dB off
  10 dB SNR white noise in the pauses and about 6 dB off 0 dB SNR noise. Without it,
  the denoise config would be a no-op and the comparison would mean nothing.

## Metrics

* **WER / CER**: jiwer, after `eval/normalize.py` on both sides. That normaliser
  lowercases, drops punctuation and apostrophes, removes bracketed annotations and the
  fillers `uh`/`um`/`hmm`, and spells out numbers 0-100. Rates are corpus-level (errors
  summed over clips ÷ reference words summed), over clips whose reference is non-empty.
* **Hallucinated words**: every hypothesis word on a clip whose reference is empty.
  Insertions over the silences in `silence_padded` show up in that clip's WER instead.
* **Seam errors**: word errors within 1 s of a chunk boundary. The boundaries are the
  pipeline's chunk edges (both edges when a gap separates two chunks) or every 30 s for
  naive30. Errors are located through the jiwer alignment. Substitutions and insertions
  take the hypothesis word's timing. A deleted word takes the gap between its neighbours,
  so a long deletion that spans a seam counts as a seam error. *Share of errors* only
  means something next to *share of audio*, the fraction of speech-clip time within 1 s
  of a seam. If seams were harmless, the two would be about equal.
* **Speed**: seconds of audio per wall-clock second (higher is faster). It includes
  decoding and, for denoise, the filter.

