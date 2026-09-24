# speechrec

A proof of concept for generating subtitles for our company's video files automatically.
It runs speech recognition locally on Apple Silicon, so no audio leaves the machine.

`transcribe.py` writes an `.srt` subtitle file for each input video, using one of eight
engines:

| Engine | Model | Notes |
|---|---|---|
| `whisper` (default) | OpenAI Whisper large-v3-turbo, via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | The baseline. Capitals and punctuation. |
| `parakeet` | NVIDIA Parakeet TDT 0.6B v3, via [parakeet-mlx](https://github.com/senstella/parakeet-mlx) | Multilingual (25 European languages), detects the language itself. Capitals and punctuation. |
| `wav2vec2` | [jonatasgrosman/wav2vec2-large-xlsr-53-dutch](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch), via transformers | Dutch only. Lowercase, no punctuation. |
| `wav2vec2-lm` | The same model, decoded with its own Dutch 5-gram language model, via pyctcdecode and kenlm | Fewer garbled words, but its mistakes are more often real (wrong) words. Needs an extra setup step, see below. Slower. |
| `canary` | NVIDIA Canary 1B v2 ([MLX conversion](https://huggingface.co/CogniSoftOrg/canary-1b-v2-mlx-bf16)), via [mlx-audio](https://github.com/Blaizzy/mlx-audio) | Multilingual (25 European languages). Capitals and punctuation. No timestamps of its own: its words are timed by aligning them to the model's separate CTC model. |
| `voxtral` | Mistral Voxtral Mini 3B ([MLX conversion](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16)), via mlx-audio | Multilingual. Capitals and punctuation. No timestamps of its own. Can invent text during music or silence. About 5× slower than the others. |
| `voxtral-rt` | Mistral Voxtral Mini 4B Realtime ([MLX conversion](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16)), via mlx-audio | Multilingual. Capitals and punctuation. No timestamps of its own. Doesn't invent text on music. Among the most accurate, but slower than real time on the Mac used here. The 4-bit version (`--model mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`) is about 4× faster with nearly the same accuracy. |
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

The `wav2vec2-lm` engine needs two packages that `requirements.txt` can't install:
`pyctcdecode` pins numpy below 2.0 (it predates numpy 2, but works with it), and the
`kenlm` release doesn't compile on Python 3.13, so it is built from source after
regenerating its Cython code (needs `cmake`, e.g. from MacPorts or Homebrew):

```sh
./venv/bin/pip install --no-deps pyctcdecode==0.5.0
mkdir -p work/build && cd work/build
curl -fLO https://github.com/kpu/kenlm/archive/master.zip && unzip -q master.zip
python3.13 -m venv cython && cython/bin/pip install cython
cython/bin/cython --cplus -3 kenlm-master/python/kenlm.pyx -o kenlm-master/python/kenlm.cpp
cd - && ./venv/bin/pip install work/build/kenlm-master
```

Its language model (1.4 GB) downloads on first use into `~/.cache/pyctcdecode`.

[JOURNAL.md](JOURNAL.md) has a "Reproducing" section with every command used so far.

## Transcribing

```sh
./venv/bin/python transcribe.py videos/vos/LeesWijs-bladerboek-33.mp4 [more videos...]
```

Options:

| Option | Default | Purpose |
|---|---|---|
| `--engine NAME` | `whisper` | `whisper`, `parakeet`, `wav2vec2`, `wav2vec2-lm`, `vosk`, `canary`, `voxtral` or `voxtral-rt`. |
| `--output-dir DIR` | next to each video | Write the `.srt` files to this folder instead (created if missing). |
| `--model NAME` | per engine, see `DEFAULT_MODELS` | Another model for the chosen engine: a Hugging Face repo, or for Vosk a model name in `~/.cache/vosk`. |
| `--language CODE` | `nl` | Whisper, Canary and Voxtral (not Voxtral Realtime) only. ISO 639-1 language code. |
| `--prompt TEXT` | `"Een voorleesverhaal voor kinderen."` | Whisper only. Short description of the material; steers spelling and style. Pass `''` for no prompt. |

Whisper writes `video.srt`. The other engines add their name (`video.parakeet.srt`), so
they never overwrite the Whisper subtitles. The script skips files it cannot find and
continues with the rest.

The output is reproducible: the same video gives the same subtitles every time, alone or
in a batch. Whisper uses a fixed random seed, and Whisper and Vosk process each video in a
separate process, because otherwise earlier videos in the same run affect the result.

To run all engines on all videos, one output folder per engine:

```sh
for e in whisper parakeet wav2vec2-lm vosk canary voxtral-rt; do
  ./venv/bin/python transcribe.py --engine $e --output-dir work/$e videos/*/*.mp4
done
```

### Keeping the Mac awake during long runs

Some runs take hours (Voxtral Realtime at full precision: about 7.5 hours for all 48
videos). macOS doesn't count a running job as activity: when nobody touches the Mac, it
goes to sleep and the job stops until someone wakes it. The only sign is that the fan
stops. On the Mac used for this project it sleeps after 1 minute, even on the charger.

To check whether a slow run was asleep:

```sh
pmset -g log | grep -E "Entering Sleep|Wake from"
```

Three ways to prevent it:

1. **Only during a job** (no settings change): start it with `caffeinate -i`, which keeps
   the Mac awake until the command ends.

   ```sh
   caffeinate -i ./venv/bin/python transcribe.py --engine voxtral-rt ...
   caffeinate -i -w PID      # for a job that is already running, by its process ID
   ```

2. **Never sleep on the charger** (battery behaviour stays as it is): System Settings →
   Battery → Options… → "Prevent automatic sleeping on power adapter when the display is
   off", or `sudo pmset -c sleep 0`. Undo with `sudo pmset -c sleep 1` (or another number
   of minutes). `pmset -g custom` shows the current settings for charger and battery.

3. **Lid closed:** a closed lid always puts the Mac to sleep, even with the two options
   above, unless it runs in clamshell mode (external display and power connected).
   `sudo pmset -a disablesleep 1` overrides even that, but a Mac that stays on in a bag
   can overheat; undo with `sudo pmset -a disablesleep 0`.

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
./venv/bin/python align.py work/whisper work/canary work/voxtral-rt work/parakeet \
    work/wav2vec2-lm work/vosk --out work/align --vote work/vote
```

Use either `wav2vec2` or `wav2vec2-lm`, not both: they share the same acoustic model and
would count double in the vote. `--lexicon FILE` (a word list, one per line, for example
the language model's `unigrams.txt`) also splits each engine's mistakes into real words,
which are easy to miss when correcting, and non-words, which stand out.

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

Parakeet, wav2vec2(-lm) and Vosk return timings per word instead of per segment. The script groups
those words into cues itself: a new cue starts after a pause of 0.8 seconds, after 14
words, or when a cue would last longer than 7 seconds.

Canary and both Voxtrals return no timings at all. The script cuts the audio into pieces of at most 15
seconds at the quietest moment and transcribes each piece. Then:

- Canary's model repository also holds a separate CTC model. Its frames (80 ms each) are
  force-aligned to Canary's text to time each word.
- For both Voxtrals, the piece's time is shared out over its sentences by text length,
  so their cue timing is approximate.

Canary's words are then grouped into cues like Parakeet's. Its timings were calibrated
against Whisper's and Parakeet's word timings (see JOURNAL.md). `timings.py` measures how
an engine's word timings compare with other engines':

```sh
./venv/bin/python timings.py work/canary work/whisper work/parakeet
```

## Limitations

This is a proof of concept, so:

- It only runs on Apple Silicon.
- The cue filtering, cue grouping and decoding settings are hard-coded in `transcribe.py`.
- There are no hand-corrected references yet, so no engine has an accuracy score.
- Whisper sometimes skips a passage (for example a sung opening); a fixed random seed
  makes this repeatable, not rarer. See JOURNAL.md.
- There are no tests.

## Repository notes

The videos live in `videos/<series>/`. Video files (`*.mp4`, `*.wmv`), subtitles
(`*.srt`), and the `work/` and `references/` folders are excluded from git through
`.gitignore`. The videos are large, and the videos, their subtitles and transcripts are all
copyrighted content, so keep them out of the repository.
