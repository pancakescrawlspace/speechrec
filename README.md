# speechrec

A proof of concept for generating subtitles for our company's video files automatically.
It runs speech recognition locally on Apple Silicon, so no audio leaves the machine.

`transcribe.py` writes an `.srt` subtitle file for each input video, using one of six
engines:

| Engine | Model | Notes |
|---|---|---|
| `whisper` (default) | OpenAI Whisper large-v3-turbo, via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | The baseline. Capitals and punctuation. |
| `parakeet` | NVIDIA Parakeet TDT 0.6B v3, via [parakeet-mlx](https://github.com/senstella/parakeet-mlx) | Multilingual (25 European languages), detects the language itself. Capitals and punctuation. |
| `wav2vec2` | [jonatasgrosman/wav2vec2-large-xlsr-53-dutch](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch), via transformers | Dutch only. Lowercase, no punctuation. |
| `canary` | NVIDIA Canary 1B v2 ([MLX conversion](https://huggingface.co/CogniSoftOrg/canary-1b-v2-mlx-bf16)), via [mlx-audio](https://github.com/Blaizzy/mlx-audio) | Multilingual (25 European languages). Capitals and punctuation. No timestamps of its own, so cue timing is approximate. |
| `voxtral` | Mistral Voxtral Mini 3B ([MLX conversion](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16)), via mlx-audio | Multilingual. Capitals and punctuation. No timestamps of its own. Can invent text during music or silence. About 5× slower than the others. |
| `vosk` | [Vosk](https://alphacephei.com/vosk/) with `vosk-model-nl-spraakherkenning-0.6` (the [Kaldi_NL](https://github.com/opensource-spraakherkenning-nl/Kaldi_NL) model) | Dutch only, runs on the CPU. Lowercase, no punctuation. |

`compare.py` compares the output of these engines with each other and, once they exist,
with hand-corrected reference subtitles. `align.py` compares them word by word and makes a
majority-vote transcript. See [JOURNAL.md](JOURNAL.md) for the plan and
results so far.

## Requirements

- A Mac with Apple Silicon (M1 or later). MLX does not run on Intel Macs or other platforms.
- Python 3.13 (for example from MacPorts or Homebrew). `requirements.txt` is pinned for
  3.13; it includes replacements for standard modules that 3.13 removed.
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (for example `brew install ffmpeg`)

## Setup

```sh
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The first run of the Whisper, Parakeet, wav2vec2, Canary and Voxtral engines downloads their
models from Hugging Face (1.6–9.4 GB each) and caches them in `~/.cache/huggingface`.

The Vosk model has to be downloaded by hand (1.6 GB unpacked):

```sh
mkdir -p ~/.cache/vosk && cd ~/.cache/vosk
curl -fLO https://alphacephei.com/vosk/models/vosk-model-nl-spraakherkenning-0.6.zip
unzip -q vosk-model-nl-spraakherkenning-0.6.zip && rm vosk-model-nl-spraakherkenning-0.6.zip
cd -
```

[JOURNAL.md](JOURNAL.md) has a "Reproducing" section with every command used so far.

## Transcribing

```sh
./venv/bin/python transcribe.py videos/vos/LeesWijs-bladerboek-33.mp4 [more videos...]
```

Options:

| Option | Default | Purpose |
|---|---|---|
| `--engine NAME` | `whisper` | `whisper`, `parakeet`, `wav2vec2`, `vosk`, `canary` or `voxtral`. |
| `--output-dir DIR` | next to each video | Write the `.srt` files to this folder instead (created if missing). |
| `--model NAME` | per engine, see `DEFAULT_MODELS` | Another model for the chosen engine: a Hugging Face repo, or for Vosk a model name in `~/.cache/vosk`. |
| `--language CODE` | `nl` | Whisper, Canary and Voxtral only. ISO 639-1 language code. |
| `--prompt TEXT` | `"Een voorleesverhaal voor kinderen."` | Whisper only. Short description of the material; steers spelling and style. Pass `''` for no prompt. |

Whisper writes `video.srt`. The other engines add their name (`video.parakeet.srt`), so
they never overwrite the Whisper subtitles. The script skips files it cannot find and
continues with the rest.

To run all engines on all videos, one output folder per engine:

```sh
for e in whisper parakeet wav2vec2 vosk canary voxtral; do
  ./venv/bin/python transcribe.py --engine $e --output-dir work/$e videos/*/*.mp4
done
```

## Comparing engines

```sh
./venv/bin/python compare.py work/whisper work/parakeet work/wav2vec2 work/vosk work/canary \
    work/voxtral --review work/review --min-disagree 2
```

This prints how much each pair of engines differs, as a word error rate (WER) of one
against the other. That shows agreement, not accuracy: two engines can make the same
mistake.

With `--review DIR` it also writes one Markdown sheet per video listing the cues of the
first engine (here Whisper) where the others disagree, with every engine's version side
by side. Use these sheets as a guide for hand-correcting. `--min-disagree N` lists only
cues where at least N other engines disagree. wav2vec2 often runs words together, so
requiring more than one engine to disagree gives a much shorter list.

### Word-by-word comparison and majority vote

`transcribe.py` also writes a `.words.json` next to every `.srt`, with each word's timing.
`align.py` uses these to compare all engines word by word:

```sh
./venv/bin/python align.py work/whisper work/canary work/voxtral work/parakeet \
    work/wav2vec2 work/vosk --out work/align --vote work/vote
```

Capitals and punctuation in the vote are decided by the engines that write them; ties go
to the engine listed first, so list the most reliable engines first.

It cuts each video into windows of at most 15 seconds at pauses, puts every engine's
words into the window they were spoken in, and aligns the engines within each window,
all pairs and all together. It prints, per engine, how many of its words other engines
confirm. `--out` writes one Markdown file per video with, per window, the majority-vote
text and the word positions where the engines disagree. `--vote` writes a majority-vote
`.srt` per video. The majority can be wrong, so use it as a draft to correct, not as a
reference.

### Accuracy against hand-corrected references

1. Copy the majority-vote `.srt` (or the Whisper `.srt`) of a video into `references/` and
   correct it by hand, using its review sheet or `align.py` output. Keep the file name
   (`LeesWijs-bladerboek-33.srt`).
2. Run `compare.py` with `--reference references`. It prints each engine's WER against the
   corrected files.

Before comparing, all text is lowercased and stripped of punctuation, so only the words
count. Cue timing is not scored yet.

Pick reference videos from several series, and never use them for training or fine-tuning
a model later, or its scores will look better than they are.

## How the Whisper engine works

1. ffmpeg extracts the audio track as 16 kHz mono WAV, the format all engines expect.
2. Whisper transcribes the audio. Two settings help with quality:
   - A short Dutch initial prompt ("Een voorleesverhaal voor kinderen.") steers the
     model toward Dutch spelling from the first segment. This default prompt is tuned for
     read-aloud children's stories, so pass `--prompt` if your material is different.
   - `condition_on_previous_text=False` stops the model from repeating the same line
     over and over.
3. The script drops cues that contain no letters. Whisper sometimes produces cues like
   `***` or lone punctuation during music or silence.
4. The remaining segments are written out as an `.srt` file.

Parakeet, wav2vec2 and Vosk return timings per word instead of per segment. The script groups
those words into cues itself: a new cue starts after a pause of 0.8 seconds, after 14
words, or when a cue would last longer than 7 seconds.

Canary and Voxtral return no timings at all. The script cuts the audio into pieces of at most 15
seconds at the quietest moment, transcribes each piece, and shares the piece's time out
over its sentences by text length.

## Limitations

This is a proof of concept, so:

- It only runs on Apple Silicon.
- The cue filtering, cue grouping and decoding settings are hard-coded in `transcribe.py`.
- wav2vec2 runs without its optional language model (that needs the `kenlm` and
  `pyctcdecode` packages), which makes it less accurate than it could be.
- There are no hand-corrected references yet, so no engine has an accuracy score.
- Whisper and Vosk don't always give the same output for the same video. See JOURNAL.md.
- There are no tests.

## Repository notes

The videos live in `videos/<series>/`. Video files (`*.mp4`, `*.wmv`), subtitles
(`*.srt`), and the `work/` and `references/` folders are excluded from git through
`.gitignore`. The videos are large, and the videos, their subtitles and transcripts are all
copyrighted content, so keep them out of the repository.
