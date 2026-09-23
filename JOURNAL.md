# Project journal

A running record of decisions and results for this project, newest entries at the bottom.

## Next step: systematic comparison of the engines

Status: **started 2026-09-23**, see the dated entries at the bottom for progress. This
section describes the whole plan, so that a new session can pick it up.

### Where things stand
- Six engines (`transcribe.py --engine whisper|parakeet|wav2vec2|vosk|canary|voxtral`) have
  transcribed all 48 videos (`videos/<series>/*.mp4`, 5.7 hours) into `work/<engine>/*.srt`
  (git-ignored, copyrighted). How: see "Reproducing".
- `compare.py` currently joins all words of a video into one list per engine (lowercased,
  punctuation removed), aligns engines pairwise over the whole video with `jiwer`'s
  word-level edit distance, and reports WER. Review sheets align every engine to Whisper
  (the "pivot") and flag Whisper cues that another engine disagrees with.
- No hand-corrected references exist yet; René will make them later. Until then all numbers
  are agreement between engines, not accuracy.

### Why a new approach
Whole-file alignment is fine for a *score*: the edit-distance count is optimal even when
several alignments are equally good. It falls short for *showing which words differ*:
- Repeated text (refrains in the rhyming books): if an engine skips one of two identical
  lines, the alignment can pin the gap on the wrong one.
- Invented text (Voxtral's repeated "voormalige mijnbouwplaats" sentence, see the Voxtral
  entries) gets aligned against real words nearby.
- Pivot bias: aligning everything to Whisper gives the others nothing to line up with where
  Whisper itself fails (e.g. the skipped `beestje` openings).
Normalization is a separate weak point: "daarnet"/"daar net" counts as two errors,
spelling variants ("Mick"/"Mik"), digits versus words.

### The plan (in this order)
1. **Keep word timestamps.** `transcribe.py` writes, next to each `.srt`, a
   `<video>[.<engine>].words.json`: a list of `{"start", "end", "word"}` in seconds. Whisper
   needs `word_timestamps=True` in `mlx_whisper.transcribe` (its segment timings are coarse).
   Parakeet (`AlignedToken`s, merged into words), wav2vec2 (pipeline word chunks) and Vosk
   already produce word timings internally. Canary and Voxtral have none: give their words
   the start/end of the 15-second piece they came from. Then rerun Whisper, Parakeet,
   wav2vec2 and Vosk on all videos (about 25 minutes; run Vosk one process per video, see
   "Reproducibility checks"), and Canary/Voxtral if their pieces aren't recoverable (Voxtral
   takes 41 minutes).
2. **Common time windows.** Cut each video into windows of at most about 15 s with
   `split_at_pauses` from `transcribe.py` (the same windows Canary and Voxtral use, since
   the function is deterministic). Put every engine's words into windows by the midpoint of
   their timestamp. Words in different windows are never aligned with each other.
3. **All-pairs support counts, no pivot.** Within each window, align every pair of engines
   (15 pairs for 6 engines). For each word of each engine, count how many other engines
   match it exactly in their pairwise alignment (0–5). This gives every engine a
   per-word confidence map; low-support words are the ones to check.
4. **Multiple alignment for a side-by-side table.** Pairwise alignments don't have to be
   consistent with each other, so for one table with a column per word position use a
   ROVER-style progressive alignment (ROVER: NIST's "Recognizer Output Voting Error
   Reduction"): per window, start with the engine with the smallest total distance to all
   others ("centre star"), then add the other engines one by one, closest first, aligning
   each to the combined result. Output per video: a Markdown table per window (rows =
   engines, columns = aligned word positions) and a **majority-vote transcript** as an
   `.srt` with the window timings, as a starting point for René's hand corrections.
5. **Checks.** Compare the whole-file WER with the sum over windows; a big difference means
   words land in the wrong window. Add a character error rate (CER) next to WER, which is
   less sensitive to compound splitting. Report per series (the `beestje` and `tim` series
   behave very differently from `vos`).

Where to put it: a new script (e.g. `align.py`) or new options in `compare.py`, reusing
`read_srt`, `normalize` and `error_counts`. Document every command in "Reproducing".

Known pitfalls: Voxtral's invented sentence (filter it or treat it as a known
hallucination), Whisper is not deterministic between runs, Vosk output depends on earlier
videos in the same process.

## TODO

- **Upgrade Python** from the python.org 3.11 install to a newer version (MacPorts has 3.13
  and 3.14 in `/opt/local/bin`). René wants this anyway, to do at a suitable moment after
  the current engine comparison. Steps: recreate `venv` with the new Python, change
  `numpy==2.3.5` back to `2.4.6` in `requirements.txt` (the numpy cap from
  `mistral-common` only applies to Python ≤ 3.12), reinstall, then rerun the engines on the
  test video to check the output is unchanged. First check that every pinned package has a
  wheel for the new version. Probably also fixes the root certificate problem that broke
  Vosk's own model download.
- Make Whisper reproducible (fixed seed, or no temperature fallback).
- Find out why Vosk's output depends on the preceding videos in a run.
- Voxtral: stop the invented sentence on music/silence properly, e.g. by skipping pieces
  without speech (voice activity detection) instead of filtering its text afterwards.
- Try Voxtral Mini 4B Realtime.
- Proper timestamps for Canary via its CTC model (forced alignment).
- wav2vec2 with its language model (`kenlm`, `pyctcdecode`).
- Hand-correct a reference set (René), then score all engines against it.

## 2026-09-23

### README
Added `README.md` describing the Whisper proof of concept (`transcribe.py`): what it does, setup,
usage, how it works, limitations.

### Hard-coded settings
Reviewed which settings in `transcribe.py` were hard-coded. Decision: expose the ones that vary
per run as command-line flags rather than a config file; there were too few to justify one.

- Added `--prompt`, `--model` and `--output-dir`. Defaults unchanged.
- Kept hard-coded on purpose: ffmpeg audio format (16 kHz mono, required by Whisper), the
  "no letters" cue filter and `condition_on_previous_text=False` (both fix known Whisper problems).
- Revisit with a config file of *profiles* (for example `--profile childrens-stories`) if the
  company's videos turn out to need several different combinations of prompt/language/model.

### Plan: alternatives to Whisper, and training our own models
Goal: compare free speech recognition systems with Whisper, and train or fine-tune our own model
to learn the process. Whisper serves as the baseline system.

Important: Whisper's output must **not** be used as the reference. Scoring against it only
measures agreement with Whisper. The reference must be hand-corrected transcripts.

Agreed order:
1. Evaluation script and a reference set (5–10 videos from different series, Whisper output
   corrected by hand). Metrics: word error rate (WER), plus a look at cue timing.
2. Try alternatives, scored the same way: NVIDIA Parakeet TDT 0.6B v3 (via `parakeet-mlx`) and a
   Dutch wav2vec2 model. Other candidates: Canary 1B v2, Mistral Voxtral, Vosk/Kaldi_NL.
3. Fine-tune wav2vec2/XLS-R on Common Voice NL, to learn the process.
4. Fine-tune Whisper on our own corrected videos (never on the reference videos) and compare
   with plain Whisper.

Training from scratch was judged a learning exercise only; it won't come close to Whisper,
which was trained on hundreds of thousands of hours.

Status: no subtitles have been corrected yet; René will do that later. Until then, systems can
only be compared with each other (agreement), not scored for accuracy.

### Steps 1 and 2: engines and comparison script
- `transcribe.py` got `--engine whisper|parakeet|wav2vec2`. All engines share the audio
  extraction, the "no letters" cue filter and the SRT writer. Parakeet and wav2vec2 return
  word timings, which are grouped into cues (new cue after a 0.8 s pause, 14 words or 7 s).
- Models: `mlx-community/parakeet-tdt-0.6b-v3` and
  `jonatasgrosman/wav2vec2-large-xlsr-53-dutch` (Apache-2.0, the most downloaded Dutch
  wav2vec2 model). wav2vec2 runs without its n-gram language model for now, since that needs
  `kenlm` and `pyctcdecode`. Worth trying later.
- New `compare.py`: agreement table (WER of each engine against each other), accuracy table
  against `references/` once hand-corrected files exist, and per-video review sheets.
- The videos moved to `videos/<series>/`. The four `.wmv` files were removed; they were the
  sources of the `.mp4` files next to them. Engine output goes to `work/<engine>/`.
  `work/` and `references/` are in `.gitignore` (copyrighted text).

René's idea: compare the engines first to get a better starting point for hand-correcting.
Where engines disagree, at least one is wrong, so those cues deserve the closest look. The
review sheets list exactly those cues. Caveat: when all engines agree they can still share
a mistake (names, rare words), so the rest of the text still needs a read-through.

First test on `vos/LeesWijs-bladerboek-33` (10.5 minutes):

| | whisper | parakeet | wav2vec2 |
|---|---|---|---|
| whisper | — | 5.2% | 18.1% |
| parakeet | 5.2% | — | 18.5% |

(WER of the column engine measured against the row engine.)

- wav2vec2 often runs words together ("gedachtemet", "visvos") and drops short words.
  Listing every cue where any engine differs flagged 75% of the cues. With
  `--min-disagree 2` (both other engines disagree with Whisper) it was 24%.
- That filtered list did contain a real Whisper error: "Vos werd er dromerig *verhaald*"
  where both others heard "dromerig *van*".
- Timing: Whisper put the first cue at 0:00–0:10, while Parakeet and wav2vec2 both start
  the first words at about 0:08.6–0:08.9. Whisper's cue timing looks looser.

### First full run: 3 engines × 48 videos
Run times for all 48 videos (5.7 hours of video in total): Whisper 319 s,
Parakeet 215 s, wav2vec2 202 s, all on the Mac.

| | whisper | parakeet | wav2vec2 |
|---|---|---|---|
| whisper | — | 14.9% | 27.4% |
| parakeet | 14.8% | — | 27.3% |

Review sheets with `--min-disagree 2`: 1490 of 3625 Whisper cues flagged (41%).

The disagreement is very uneven per series:
- `vos` (4.5–10%) and `lammetje` are easy; the `vos` test video was not representative.
- `beestje` is the worst (37–55%). Whisper skipped the entire rhyming opening of the
  videos (first ~30 s, "Beestje kom je op mijn feestje…"), probably sung or over music,
  while both Parakeet and wav2vec2 transcribed it. Whisper also spells the character
  "Mick" where the others write "mik"; check against the books.
- `tim` (19–31%): here Parakeet has clearly fewer words than Whisper and wav2vec2.
- Where all three have the text, Whisper is usually the most accurate (e.g. Parakeet
  "dweeolen… blues een tommel" for "Twee violen… plus een trommel").

Lesson: Whisper can silently drop whole sections, and the review sheets catch this (the
missing words show up under the next cue). Correcting only flagged cues is not enough
for `beestje`-style videos: the missing sections need new cues.

Next (René's request): also try Vosk, NVIDIA Canary 1B v2 and Mistral Voxtral, one at a
time. René asked not to run too many things at once.

### Vosk (Kaldi_NL)
Vosk's large Dutch model, `vosk-model-nl-spraakherkenning-0.6`, is the Kaldi_NL model from
Radboud University (trained on the Spoken Dutch Corpus, CGN), so this also covers the Kaldi_NL
option. It runs on the CPU, about 20 s per 10-minute video. It writes `<unk>` for sounds it
can't match to a word; `transcribe.py` drops those.

Vosk downloads models from alphacephei.com itself, but that failed here: the python.org
Python install had no root certificates (`CERTIFICATE_VERIFY_FAILED`). Instead of changing
the system Python, the model was downloaded with `curl` (see "Reproducing" below).

Full run with Vosk added (WER of the column engine against the row engine):

| | whisper | parakeet | wav2vec2 | vosk |
|---|---|---|---|---|
| whisper | — | 14.9% | 27.4% | 24.6% |
| parakeet | 14.8% | — | 27.3% | 26.9% |
| wav2vec2 | 27.1% | 27.2% | — | 33.9% |
| vosk | 25.6% | 28.1% | 35.6% | — |

Vosk agrees with Whisper slightly more than wav2vec2 does, and less than Parakeet does.
With four engines, review sheets flag 65% of Whisper's cues at `--min-disagree 2` and 34%
at `--min-disagree 3` (all three others disagree).

**Whisper is not deterministic.** The earlier test run of `vos/LeesWijs-bladerboek-33`
produced "Vos werd er dromerig verhaald", the full run with the same settings "Vos werd er
dromerig van". Probable cause: Whisper's temperature fallback, which retries a segment with
random sampling when the first attempt looks poor. To look into later: fix the random seed
or turn off the fallback, so that reruns are reproducible.

### NVIDIA Canary 1B v2
Runs through `mlx-audio` (the `[stt]` extra), with the unquantized MLX conversion
`CogniSoftOrg/canary-1b-v2-mlx-bf16` (CC-BY-4.0, format conversion of `nvidia/canary-1b-v2`).
The model card warns about some rough edges with an older mlx-audio (dtype errors,
`language=` being ignored); none of those showed up with mlx-audio 0.5.5, where
`source_lang`/`target_lang` are passed directly.

Canary returns text only, without timestamps. `transcribe.py` therefore cuts the audio into
pieces of at most 15 s at the quietest moment (`split_at_pauses`), transcribes each piece,
and shares the piece's time out over its sentences by text length (`text_to_cues`). Cue
timing is approximate; the words are what the comparison uses. The model repo also contains
a CTC model for proper word timestamps by forced alignment; an option for later.

All 48 videos: 377 s. The download took about 8 minutes (3.3 GB including the CTC model).

| | whisper | parakeet | wav2vec2 | vosk | canary |
|---|---|---|---|---|---|
| whisper | — | 14.9% | 27.4% | 24.6% | 12.4% |
| parakeet | 14.8% | — | 27.3% | 26.9% | 14.1% |
| wav2vec2 | 27.1% | 27.2% | — | 33.9% | 25.8% |
| vosk | 25.6% | 28.1% | 35.6% | — | 26.1% |
| canary | 12.0% | 13.8% | 25.3% | 24.4% | — |

Canary is closest to Whisper and to every other engine. Unlike Whisper, it does transcribe
the rhyming opening of the `beestje` videos ("Beestje, kom je op mijn feestje? Mik is
jarig, een zonnige dag."), and spells the name "Mik" like Parakeet and wav2vec2.

### Reproducibility checks
Installing `mistral-common` (needed for Voxtral) downgraded numpy from 2.4.6 to 2.3.5. To
check nothing broke, all five engines were rerun on `vos/LeesWijs-bladerboek-33`:
Parakeet, wav2vec2 and Canary gave byte-identical output. Whisper differed, as expected
(see "Whisper is not deterministic" above). Vosk also differed, which led to a second
finding:

**Vosk's output for a video depends on which videos were processed before it in the same
process.** Two separate single-video runs are identical to each other, and match the
single-video run from before the numpy change. But when another video is transcribed first
in the same `transcribe.py` call, the result differs: 24 to 132 changed lines in a `diff`
of the `.srt` files (about 110 cues, so 440 lines), in both text and timing. Loading a fresh Vosk model per video did not help, so the state is kept somewhere
inside the Vosk library. Not investigated further.

Workaround for reproducible Vosk output: run one process per video:

```sh
for f in videos/*/*.mp4; do
  ./venv/bin/python transcribe.py --engine vosk --output-dir work/vosk "$f"
done
```

The Vosk results in `work/vosk` and in the tables above come from a single batch call, so
they are slightly different from what per-video runs would give. The differences are
small single words ("ik ken hem ben" / "ik hem ben", "baat" / "waad").

### Mistral Voxtral Mini 3B
Runs through `mlx-audio` with `mlx-community/Voxtral-Mini-3B-2507-bf16` (Apache-2.0, 9.4 GB,
unquantized; the download took 22 minutes). Needs the extra package `mistral-common[audio]`;
see the reproducibility checks below for the numpy downgrade it caused.

- mlx-audio's Voxtral code only accepts a file path, not an audio array (the `transformers`
  processor then complains that `format` is missing), so each piece is written to a
  temporary WAV file first.
- Like Canary it returns no timestamps, so it uses the same 15-second pieces.
- Test video: good text, but in the 7.5-second music intro before the story it **invented a
  whole sentence** ("Deze nieuwe stad wordt gebouwd op een voormalige mijnbouwplaats…").
  Engines built on a language model can do this on music or silence. Left in on purpose, so
  the comparison shows it.
- About 74 s per video, roughly 5× slower than the other engines.
- On loading, `transformers` warns that the tokenizer has "an incorrect regex pattern" and
  suggests `fix_mistral_regex=True`. mlx-audio doesn't expose that option. Not investigated;
  the Dutch output looked correct.

Not tried yet: Voxtral Mini 4B Realtime (`mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16`,
8.9 GB), Mistral's newer transcription-only model.

### All six engines on all 48 videos
Voxtral took 2451 s (41 minutes) for all videos, 6–12× slower than the others.

| | whisper | parakeet | wav2vec2 | vosk | canary | voxtral |
|---|---|---|---|---|---|---|
| whisper | — | 14.9% | 27.4% | 24.6% | 12.4% | 31.4% |
| parakeet | 14.8% | — | 27.3% | 26.9% | 14.1% | 34.8% |
| wav2vec2 | 27.1% | 27.2% | — | 33.9% | 25.8% | 46.7% |
| vosk | 25.6% | 28.1% | 35.6% | — | 26.1% | 46.8% |
| canary | 12.0% | 13.8% | 25.3% | 24.4% | — | 30.4% |
| voxtral | 25.1% | 28.0% | 37.7% | 36.0% | 25.1% | — |

Voxtral looked worst, but that is almost entirely one effect: in pieces with only music or
silence it outputs **the same invented sentence** every time, "Deze nieuwe stad wordt gebouwd
op een voormalige mijnbouwplaats, waar de mijnwerkers in de jaren 1920 en 1930 hun woningen
hadden." It appears 278 times in 47 of the 48 videos. At 21 words each, that is about 5,800
words, nearly all of the difference between Voxtral's total (34,432 words) and Canary's
(28,364). Whisper has 27,513.

With the cues holding that sentence removed (both halves; it spans two cues), Voxtral is
the engine closest to the others:

| | whisper | parakeet | canary | voxtral (cleaned) |
|---|---|---|---|---|
| whisper | — | 14.9% | 12.4% | 10.2% |
| parakeet | 14.8% | — | 14.1% | 13.7% |
| canary | 12.0% | 13.8% | — | 9.9% |
| voxtral (cleaned) | 9.9% | 13.3% | 9.8% | — |

So Voxtral's actual transcription is very good. It also gets the `beestje` openings right.
The raw output in `work/voxtral` still contains the sentence; the cleaned copy is in
`work/test/voxtral-clean`, made with:

```sh
mkdir -p work/test/voxtral-clean
for f in work/voxtral/*.srt; do
  awk 'BEGIN{RS="";ORS="\n\n"} !/voormalige mijnbouwplaats/ && !/jaren 1920 en 1930 hun woningen hadden/' \
      "$f" > work/test/voxtral-clean/$(basename $f)
done
./venv/bin/python compare.py work/whisper work/parakeet work/canary work/test/voxtral-clean
```

(A first attempt removed only the first half of the sentence and gave misleading numbers.)

Summary so far, without references: Canary and cleaned Voxtral agree best with the rest,
Whisper is close behind but skips sung/rhymed openings, Parakeet is a bit further off,
and Vosk and wav2vec2 are clearly weaker. Agreement is not accuracy; the hand-corrected
references will decide.

## Reproducing

Every command used so far, grouped by purpose. Run from the repository root on an Apple
Silicon Mac with Python 3 and ffmpeg (`brew install ffmpeg`). The videos go in
`videos/<series>/*.mp4`; they are not in git. Commands that originally wrote to a temporary
folder are shown with `work/test/` instead.

### Setup

To recreate the environment:

```sh
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins exact versions and is regenerated after every install with
`./venv/bin/pip freeze > requirements.txt`. The packages that matter are `mlx-whisper`,
`parakeet-mlx`, `transformers` and `torch` (wav2vec2), `vosk`, `mlx-audio` (Canary and
Voxtral), `mistral-common` (Voxtral) and `jiwer`.

How they were added, in order. Each install was first tried with `--dry-run` to check it
wouldn't change any package already installed:

```sh
# Parakeet, wav2vec2 and the comparison script
./venv/bin/pip install --dry-run parakeet-mlx transformers jiwer   # only additions
./venv/bin/pip install parakeet-mlx transformers jiwer

# Vosk
./venv/bin/pip install vosk

# Canary and Voxtral
./venv/bin/pip install --dry-run 'mlx-audio[stt]' vosk              # only additions
./venv/bin/pip install 'mlx-audio[stt]'

# Voxtral also needs mistral-common. The dry run showed it would downgrade numpy
# 2.4.6 -> 2.3.5; its metadata showed why (numpy<2.4 on Python <= 3.12):
./venv/bin/pip install --dry-run 'mistral-common[audio]'
./venv/bin/pip download --no-deps 'mistral_common==1.12.0' -d work/test/wheels
unzip -p work/test/wheels/mistral_common-1.12.0*.whl '*/METADATA' | grep '^Requires-Dist' | grep -i numpy
./venv/bin/pip install 'mistral-common[audio]'
./venv/bin/pip check                                                  # "No broken requirements found."
```

### Models

The Hugging Face models download automatically on first use, into `~/.cache/huggingface`:

| Engine | Model | Size |
|---|---|---|
| whisper | `mlx-community/whisper-large-v3-turbo` | 1.6 GB |
| parakeet | `mlx-community/parakeet-tdt-0.6b-v3` | 2.5 GB |
| wav2vec2 | `jonatasgrosman/wav2vec2-large-xlsr-53-dutch` | 2.4 GB (weights in two formats) |
| canary | `CogniSoftOrg/canary-1b-v2-mlx-bf16` | 3.3 GB (including a CTC model for timestamps, not used yet) |
| voxtral | `mlx-community/Voxtral-Mini-3B-2507-bf16` | 9.4 GB |

The sizes were measured in the cache. Measure `blobs/`, since the `snapshots/` folder only
holds links to the same files:

```sh
du -sh ~/.cache/huggingface/hub/models--*/blobs
```

The Vosk model has to be downloaded by hand (1.6 GB unpacked). `transcribe.py` finds it in
`~/.cache/vosk` by name. Vosk's own download failed with `CERTIFICATE_VERIFY_FAILED`
(the python.org Python has no root certificates set up), hence `curl`:

```sh
mkdir -p ~/.cache/vosk && cd ~/.cache/vosk
curl -fLO https://alphacephei.com/vosk/models/vosk-model-nl-spraakherkenning-0.6.zip
unzip -q vosk-model-nl-spraakherkenning-0.6.zip && rm vosk-model-nl-spraakherkenning-0.6.zip
cd -
```

Other Dutch Vosk models are listed at https://alphacephei.com/vosk/models (for example
`vosk-model-small-nl-0.22`, 39 MB); use one with `--engine vosk --model <name>` after
downloading it the same way.

### How the models were chosen

Hugging Face searches, sorted by downloads, plus the license of the chosen wav2vec2 model:

```sh
./venv/bin/python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for q in ["wav2vec2 dutch", "xls-r nl", "parakeet-tdt-0.6b-v3", "canary-1b-v2", "canary mlx", "voxtral"]:
    print("##", q)
    for m in api.list_models(search=q, sort="downloads", limit=8):
        print(" ", m.id, m.downloads)
info = api.model_info("jonatasgrosman/wav2vec2-large-xlsr-53-dutch")
print("license:", info.card_data.get("license"))
PY
```

Vosk's Dutch models:

```sh
curl -s https://alphacephei.com/vosk/models/model-list.json | ./venv/bin/python -c "
import json, sys
for m in json.load(sys.stdin):
    if m['lang'] == 'nl': print(m['name'], m['size_text'], m['type'], m.get('obsolete'))"
```

Which speech-to-text models mlx-audio supports (it has Canary and Voxtral, so NVIDIA's NeMo
toolkit isn't needed), and what it depends on, without installing it:

```sh
./venv/bin/pip download --no-deps mlx-audio -d work/test/wheels
unzip -l work/test/wheels/mlx_audio-*.whl | grep 'stt/models/[^/]*/__init__'
unzip -p work/test/wheels/mlx_audio-*.whl '*/METADATA' | grep '^Requires-Dist'
```

Which MLX conversions mlx-audio can load (it needs `model_type` in `config.json`), and their
size. Canary: `CogniSoftOrg/canary-1b-v2-mlx-bf16` was chosen as the only unquantized one
with a `model_type`. Voxtral: the `mlx-community` bf16/fp16 versions.

```sh
./venv/bin/python - <<'PY'
import json
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
repos = ["CogniSoftOrg/canary-1b-v2-mlx-bf16", "Mediform/canary-1b-v2-mlx-q8",
         "eelcor/canary-1b-v2-mlx", "TechHara/canary-1b-v2-mlx-q8",
         "mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Mini-4B-Realtime-2602"]
repos += [m.id for m in api.list_models(search="voxtral", author="mlx-community", limit=20)]
for repo in repos:
    info = api.model_info(repo, files_metadata=True)
    size = sum(f.size or 0 for f in info.siblings
               if f.rfilename.endswith((".safetensors", ".npz", ".bin")))
    try:
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        mt, q = cfg.get("model_type"), cfg.get("quantization")
    except Exception:
        mt, q = "no config.json", None
    lic = info.card_data.get("license") if info.card_data else None
    print(f"{repo}: {size/1e9:.1f} GB, model_type={mt}, quant={q}, license={lic}")
PY
```

The Canary conversion's model card (usage, known issues, the CTC timestamp model):

```sh
./venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(open(hf_hub_download('CogniSoftOrg/canary-1b-v2-mlx-bf16', 'README.md')).read())"
```

Free disk space before downloading the large Voxtral models: `df -h ~`.

### Test runs on one video

Every engine was first tried on `vos/LeesWijs-bladerboek-33` (10.5 minutes):

```sh
for e in whisper parakeet wav2vec2 vosk canary voxtral; do
  /usr/bin/time -p ./venv/bin/python transcribe.py --engine $e --output-dir work/test \
      videos/vos/LeesWijs-bladerboek-33.mp4
done
./venv/bin/python compare.py work/test/whisper work/test/parakeet work/test/wav2vec2 \
    --review work/test/review
```

(For the three-engine test, the `.srt` files were copied into one folder per engine first.)

Before adding Canary and Voxtral to `transcribe.py`, both were tried directly on two
14-second pieces of the audio:

```sh
ffmpeg -v error -y -i videos/vos/LeesWijs-bladerboek-33.mp4 -vn -ac 1 -ar 16000 work/test/vos33.wav
./venv/bin/python - <<'PY'
import time, soundfile
from mlx_audio.stt.utils import load
audio, rate = soundfile.read("work/test/vos33.wav", dtype="float32")

model = load("CogniSoftOrg/canary-1b-v2-mlx-bf16")
for a, b in [(8, 22), (22, 36)]:
    r = model.generate(audio[a*rate:b*rate], source_lang="nl", target_lang="nl", use_pnc=True)
    print(f"canary {a}-{b}s:", r.text)

# Voxtral needs a file path, not an array
model = load("mlx-community/Voxtral-Mini-3B-2507-bf16")
for a, b in [(8, 22), (22, 36)]:
    soundfile.write("work/test/piece.wav", audio[a*rate:b*rate], rate)
    print(f"voxtral {a}-{b}s:", model.generate("work/test/piece.wav", language="nl").text)
PY
```

### Transcribing all videos

One output folder per engine under `work/`. Times are for all 48 videos (5.7 hours) on
René's Mac, after the model download:

```sh
for e in whisper parakeet wav2vec2 vosk canary voxtral; do
  start=$SECONDS
  ./venv/bin/python transcribe.py --engine $e --output-dir work/$e videos/*/*.mp4
  echo "$e done in $((SECONDS-start))s"
done
```

| Engine | Time |
|---|---|
| whisper | 319 s |
| parakeet | 215 s |
| wav2vec2 | 202 s |
| vosk | 674 s (CPU only) |
| canary | 377 s |
| voxtral | 2451 s |

To total the video length (ffprobe comes with ffmpeg):

```sh
for f in videos/*/*.mp4; do ffprobe -v error -show_entries format=duration -of csv=p=0 "$f"; done |
  awk '{s+=$1} END {printf "%.2f hours\n", s/3600}'
```

### Comparing

```sh
./venv/bin/python compare.py work/whisper work/parakeet work/wav2vec2 work/vosk work/canary \
    work/voxtral --review work/review --min-disagree 2
```

Prints the agreement table and writes one review sheet per video to `work/review/`. The
first folder is the base engine that the review sheets are organised around. Once
hand-corrected files exist in `references/`, add `--reference references` for accuracy
scores. The flagged percentages in this journal came from runs with `--min-disagree 1`, `2`
and `3`.

Disagreement per video, Whisper against Parakeet and wav2vec2, worst first (this showed
the `beestje` and `tim` problems):

```sh
./venv/bin/python - <<'PY'
from pathlib import Path
import compare as c
w, p, v = (c.load_engine(Path(f"work/{e}")) for e in ("whisper", "parakeet", "wav2vec2"))
series = {x.stem: x.parent.name for x in Path("videos").glob("*/*.mp4")}
rows = []
for k in w:
    ww, pw, vw = c.words_of(w[k]), c.words_of(p[k]), c.words_of(v[k])
    e, n = c.error_counts(ww, pw)
    e2, _ = c.error_counts(ww, vw)
    rows.append((e / n, series[k], k, len(ww), len(pw), len(vw), e2 / n))
for r in sorted(rows, reverse=True):
    print(f"{r[0]:6.1%}  {r[1]:9} {r[2]:26} words w/p/v {r[3]:5} {r[4]:5} {r[5]:5}   w2v {r[6]:6.1%}")
PY
```

### Reproducibility checks

After the numpy downgrade, every engine was rerun on the test video and compared with the
full run:

```sh
for e in whisper parakeet wav2vec2 vosk canary; do
  ./venv/bin/python transcribe.py --engine $e --output-dir work/test/numpy-check/$e \
      videos/vos/LeesWijs-bladerboek-33.mp4
  f=$(ls work/test/numpy-check/$e/*.srt)
  cmp -s "$f" "work/$e/$(basename $f)" && echo "$e: identical" || diff "work/$e/$(basename $f)" "$f"
done
```

The Vosk experiments. Two separate single-video runs (identical to each other):

```sh
for i in 1 2; do
  ./venv/bin/python transcribe.py --engine vosk --output-dir work/test/vosk-repeat/$i \
      videos/vos/LeesWijs-bladerboek-33.mp4
done
diff work/test/vosk-repeat/1/*.srt work/test/vosk-repeat/2/*.srt
```

The same video after another one in the same process (differs from both):

```sh
./venv/bin/python transcribe.py --engine vosk --output-dir work/test/vosk-order \
    videos/vos/LeesWijs-bladerboek-34.mp4 videos/vos/LeesWijs-bladerboek-33.mp4
diff work/test/vosk-order/LeesWijs-bladerboek-33.vosk.srt work/test/vosk-repeat/1/*.srt
```

The same order test was repeated with `load_vosk` changed to load a fresh model for every
video (its `@functools.cache` removed). The output still differed, so that change was
reverted.
