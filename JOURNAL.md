# Project journal

A running record of decisions and results for this project, newest entries at the bottom.

## Next step: systematic comparison of the engines

Status: **done 2026-09-23** (all five steps, run on all 48 videos; results in
"Systematic comparison: results on all 48 videos" below). This section describes the
plan as it was carried out.

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

- Delete `venv-py311/` (the old Python 3.11 environment, kept as a fallback after the
  upgrade to 3.13) once the new environment has proven itself.
- Whisper robustness: it sometimes skips a whole passage (the `beestje` openings, a 20 s
  passage in `vos/LeesWijs-bladerboek-35` depending on the random draw). Idea: let Whisper
  transcribe the same short pieces as Canary, so one bad 30 s window can't swallow a
  passage, and compare against the vote.
- Optional: find out *why* Whisper and Vosk depend on earlier videos in the same process
  (worked around by running each video in its own process).
- Voxtral: stop the invented sentence on music/silence properly, e.g. by skipping pieces
  without speech (voice activity detection) instead of filtering its text afterwards.
- Voxtral Realtime: derive real word timings from its token positions.
- Proper timestamps for Canary via its CTC model (forced alignment).
- Hand-correct a reference set (René), starting from the vote `.srt` files in `work/vote/`,
  then score all engines against it with `compare.py --reference references`.

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

Workaround for reproducible Vosk output: run one process per video (later built into
`transcribe.py`, see "Reproducible Whisper (and Vosk)"):

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

### Systematic comparison, step 1: word timestamps
`transcribe.py` now writes a `.words.json` next to every `.srt`: a list of
`{"start", "end", "word"}` (seconds, original spelling and punctuation).
- Whisper: `word_timestamps=True`. Its word timings are far more precise than its segment
  timings (first word "Vos" at 8.32 s, while the first segment started at 0:00).
- Parakeet: its tokens are word pieces; a piece starting with a space starts a new word.
- wav2vec2 and Vosk: their word timings were already used for the cues.
- Canary and Voxtral: every word gets the start and end of its 15-second piece.
Test on `vos/LeesWijs-bladerboek-33`: the four engines with real word timings agree within
a few tenths of a second (first word 8.3–8.9 s). Canary's and Voxtral's `.srt` output was
byte-identical to the full run, so unlike Whisper and Vosk they are deterministic.
With `word_timestamps=True` Whisper made 134 cues instead of 127 for this video.

The first run's outputs, which all tables above are based on, were copied to `work/run1/`
before rerunning all engines with word timings (`work/run_words.sh`, see "Reproducing").

### Systematic comparison, steps 2 and 3: windows and all-pairs support
New script `align.py`: cuts each video into the same ≤ 15 s windows as `split_at_pauses`,
puts every engine's words into windows by the midpoint of their timing, aligns every pair
of engines within each window, and counts per word how many other engines have exactly
that word there ("support"). It prints the share of each engine's words supported by a
majority of the others and by none, plus per-window against whole-file WER per pair. With
`--out` it writes a Markdown file per video: every window with each engine's text, words
supported by at most one other engine in bold with their count.

Bug found in testing: sorting words on (start, end, word) put Canary's and Voxtral's words
in alphabetical order within a window, since they all share their piece's timing. Now
sorted on start time only (stable sort keeps the spoken order).

Test on `vos/LeesWijs-bladerboek-33` (59 windows):

| engine | words | supported by ≥ 3 of 5 | supported by none |
|---|---|---|---|
| whisper | 1019 | 95.7% | 1.5% |
| parakeet | 1016 | 94.6% | 2.1% |
| wav2vec2 | 1013 | 84.6% | 13.3% |
| vosk | 972 | 88.6% | 9.7% |
| canary | 1018 | 94.6% | 2.8% |
| voxtral | 1103 | 88.3% | 9.1% |

Per-window and whole-file WER differ by at most about 1 percentage point per pair (for
example Whisper/Canary 6.8% against 6.4%), so the midpoint rule rarely puts words in the
wrong window. Voxtral's invented sentence shows up as its own window full of
zero-support words during the music intro (00:00–00:07). Example of a split decision:
"gedachten" (Whisper, Parakeet) against "gedachte" (Vosk, Canary, Voxtral).

Remaining: run `align.py` on all 48 videos once the rerun with word timings is done, then
step 4 (ROVER-style multiple alignment and majority-vote transcript) and the rest of step 5
(CER, results per series).

### Systematic comparison, step 4: multiple alignment and majority vote
`align.py` now also aligns all engines of a window together into columns (one per word
position), ROVER-style: it starts with the engine closest to all others in that window
(the "centre star"), then adds the others closest first, each aligned to the majority
vote of the columns so far. Each column is decided by majority vote; if most engines
have no word there, the column is dropped; ties go to the starting engine. New outputs:
- per window in the Markdown files: the voted text and a table of only the positions
  where the engines disagree;
- with `--vote DIR`: a majority-vote `.srt` per video, as a starting point for
  hand-correcting. The winning word keeps the spelling and punctuation of the first engine
  (in command-line order) that has it; its timing is the median of the engines with real
  word timings (words spanning more than 3 s, i.e. Canary/Voxtral piece timings, are not
  used); cues are made with the same rules as for Parakeet/wav2vec2/Vosk;
- printed: each engine's WER against the vote.

Test on `vos/LeesWijs-bladerboek-33`, WER against the majority vote: Whisper 3.2%,
Parakeet 4.3%, Canary 4.8%, Voxtral 11.2%, Vosk 15.8%, wav2vec2 17.9%. Voxtral's invented
sentence in the intro is voted away (only Voxtral has it). The vote `.srt` has proper start
times (first cue at 8.685 s, where Whisper's own `.srt` starts at 0:00) and keeps Whisper's
capitals and punctuation.

Limitations seen in the test:
- **The majority can be wrong.** "oplaasboot" (Parakeet, Canary, Voxtral) beats
  "opblaasboot" (Whisper, Vosk), though "opblaasboot" is the real word. The vote is a draft
  to correct, not a reference.
- When the winning spelling comes from an engine without punctuation, the punctuation is
  lost ("gedachte" from Vosk, where Whisper had "gedachten.").
- Cues can break mid-sentence, since cue cutting ignores sentence ends.
- WER against the vote favours engines that are part of the majority; it is agreement.

### Systematic comparison, step 5: CER and results per series
`align.py` now also prints each engine's character error rate (CER) against the vote, and
the WER against the vote per series (the folder a video is in). Test video:

| engine | WER vs vote | CER vs vote |
|---|---|---|
| whisper | 3.2% | 1.2% |
| parakeet | 4.3% | 2.0% |
| wav2vec2 | 17.9% | 8.0% |
| vosk | 15.8% | 9.8% |
| canary | 4.8% | 1.6% |
| voxtral | 11.2% | 11.0% |

wav2vec2's errors are mostly run-together words and near-misses (WER more than twice its
CER), Vosk more often has entirely different words, and Voxtral's CER stays high because
its invented sentence counts in full.

Status at this point: all five steps implemented and tested on one video; the full run
follows below.

### Systematic comparison: results on all 48 videos
Rerun with word timings (`work/run_words.sh`): Whisper 545 s (slower with word
timestamps), Parakeet 211 s, wav2vec2 197 s, Canary 362 s, Voxtral 2361 s, Vosk 693 s (one
process per video). Compared with the first run (`work/run1/`): Parakeet, wav2vec2, Canary
and Voxtral are byte-identical in all 48 `.srt` files. Whisper changed in all 48 (word
timestamps change its segmentation, and it isn't deterministic anyway). Vosk changed in 47
of 48; the unchanged one is presumably the first video in the batch, the only one with no
video before it, which fits the order problem.

`align.py` on all 48 videos (20 s), engines with capitals and punctuation first, so the
vote takes their spelling:

| engine | words | supported by ≥ 3 of 5 | by none | WER vs vote | CER vs vote |
|---|---|---|---|---|---|
| whisper | 27486 | 92.1% | 2.0% | 6.9% | 5.2% |
| canary | 28364 | 90.6% | 4.1% | 6.9% | 3.2% |
| voxtral | 34432 | 75.4% | 19.7% | 25.9% | 27.0% |
| parakeet | 27711 | 89.3% | 5.4% | 10.8% | 5.7% |
| wav2vec2 | 27802 | 79.2% | 16.9% | 23.5% | 11.3% |
| vosk | 26494 | 84.4% | 11.9% | 21.2% | 12.6% |

WER against the vote per series (from the first `align.py` run, with the engines in the
order whisper, parakeet, wav2vec2, vosk, canary, voxtral; the order only matters for ties):

| series | whisper | parakeet | wav2vec2 | vosk | canary | voxtral |
|---|---|---|---|---|---|---|
| balotje | 7.4% | 8.9% | 15.1% | 14.6% | 5.2% | 22.5% |
| beestje | 24.8% | 16.5% | 34.0% | 38.0% | 7.4% | 67.5% |
| bentje | 4.2% | 13.6% | 18.9% | 19.5% | 7.2% | 22.8% |
| eend | 2.5% | 13.1% | 22.7% | 16.4% | 6.6% | 17.2% |
| help | 8.1% | 15.5% | 22.3% | 11.8% | 8.6% | 31.7% |
| jake | 5.2% | 12.3% | 28.7% | 19.5% | 8.0% | 29.7% |
| kerst | 8.4% | 8.4% | 27.0% | 32.5% | 7.8% | 27.1% |
| lammetje | 11.1% | 4.5% | 19.1% | 23.4% | 5.5% | 33.3% |
| rinus | 5.7% | 12.9% | 22.4% | 10.7% | 7.8% | 42.7% |
| sint | 9.2% | 8.5% | 31.6% | 37.4% | 11.2% | 23.9% |
| tim | 5.1% | 25.4% | 31.6% | 17.9% | 3.7% | 25.6% |
| vos | 5.8% | 4.2% | 17.9% | 15.3% | 4.2% | 15.7% |

Findings:
- **Whisper and Canary are tied overall** (6.9% WER against the vote); Canary has the
  lowest CER (3.2%). Canary is the most consistent across series (3.7–11.2%); Whisper is
  best in 5 series but falls to 24.8% in `beestje` (skipped openings).
- Parakeet struggles with `tim` (25.4%, it drops words there) but is best in `lammetje`.
- Voxtral's numbers are dominated by its invented sentence (see the Voxtral entries);
  the vote removes it: none of the 48 vote files contain it.
- Per-window and whole-file pairwise WER differ by at most 1.2 percentage points, so the
  midpoint rule works.
- The vote `.srt` files (`work/vote/`) contain the `beestje` openings Whisper skipped, with
  the majority spelling "Mik". 28,323 words in total.
- Caveat again: these are all measured against the majority, so engines that agree with
  the majority score well by construction. Only hand-corrected references measure accuracy.

Known weak spot of the vote `.srt`: cues are cut by pause/length only, not at sentence
ends, so 985 cues start mid-sentence with a lowercase word (Whisper's own `.srt`: 143).
Worth fixing before hand-correcting (see TODO).

### Better majority-vote drafts
Checked the vote `.srt` files before René starts correcting them, and fixed four problems.
None of them needed an engine rerun; `align.py` takes 20 s for all 48 videos.

1. **Cues now also end at the end of a sentence** (`words_to_cues` in `transcribe.py`: after
   a word ending in `.`, `!`, `?` or `…`, also behind a closing quote). This only affects
   the vote: wav2vec2 and Vosk write no punctuation, and Parakeet makes its own cues.
2. **Capitals and punctuation are voted on too.** Whisper sometimes writes a whole stretch
   without capitals or punctuation ("bentje keek om zich heen de bomen waren zo hoog…" in
   `bentje/LeesWijs-bladerboek-30`), and the vote copied that since it took spelling from
   the first engine on the command line. Now the spelling is the most common form among
   the engines with the winning word that use capitals or punctuation somewhere in that
   window. An intermediate version that only counted formatted forms capitalised
   mid-sentence words ("riep ze, Maar Sarah"), because lowercase "maar" didn't count.
3. **Two-step vote.** In `beestje/LeesWijs-bladerboek-5`, "Dag muis, zegt Mik" (a line
   Whisper skipped) came out as "muis, zegt": three engines had no word at the "Mik"
   position, and the four that did spelled it four ways (mik, mick, nik, nick), so "no
   word" won with 2 votes against 1 each. Now the vote first decides whether there is a
   word (engines with any word against engines with none; a tie keeps the word), then
   which one. Ties between spellings go to the engine listed first on the command line;
   they used to go to the window's starting engine, which was wav2vec2 in that window
   (since Whisper had nothing), giving "nik".
4. **Words only Canary/Voxtral heard** have no timing of their own. They were placed right
   after the previous word, which put "Dag," in its own cue 6 s too early. Now a run of
   such words goes just before the next timed word (at most 0.4 s per word), or after the
   previous one at the end of a window.

Effect: cues starting with a lowercase letter went from 985 to 471, most of them now
dialogue ("Wil je dit echt? ⏎ vraagt de dierenarts.", correct Dutch) or cuts at a reading
pause. The vote files grew from 28,323 to 28,685 words (words the one-step vote dropped).
Still none contain Voxtral's invented sentence.

Scores against the new vote (engines in the order whisper, canary, voxtral, parakeet,
wav2vec2, vosk):

| engine | WER vs vote | CER vs vote |
|---|---|---|
| whisper | 6.8% | 5.3% |
| canary | 7.0% | 3.4% |
| voxtral | 25.8% | 26.6% |
| parakeet | 11.2% | 6.1% |
| wav2vec2 | 23.8% | 11.3% |
| vosk | 21.6% | 13.1% |

| series | whisper | canary | voxtral | parakeet | wav2vec2 | vosk |
|---|---|---|---|---|---|---|
| balotje | 7.0% | 5.6% | 23.0% | 8.8% | 15.5% | 14.6% |
| beestje | 23.5% | 6.6% | 65.9% | 17.1% | 35.0% | 39.6% |
| bentje | 4.1% | 7.3% | 23.0% | 13.8% | 19.0% | 19.8% |
| eend | 2.5% | 6.8% | 17.2% | 13.0% | 22.8% | 16.9% |
| help | 8.7% | 8.2% | 32.0% | 15.7% | 23.0% | 11.7% |
| jake | 4.5% | 8.6% | 28.9% | 13.7% | 29.7% | 19.9% |
| kerst | 8.3% | 7.9% | 26.7% | 8.9% | 27.2% | 32.8% |
| lammetje | 11.2% | 4.9% | 33.6% | 5.5% | 19.2% | 23.7% |
| rinus | 5.4% | 7.6% | 42.6% | 12.7% | 22.5% | 11.5% |
| sint | 9.3% | 10.3% | 23.2% | 9.6% | 32.3% | 38.4% |
| tim | 5.2% | 5.7% | 23.8% | 26.5% | 31.1% | 18.3% |
| vos | 5.8% | 4.3% | 15.6% | 4.3% | 17.8% | 15.5% |

The vote drafts in `work/vote/` are now ready as a starting point for hand-correcting.
Remaining known imperfections: some cues still break at a reading pause mid-sentence, and
the majority can be wrong ("oplaasboot").

### Python upgrade: 3.11 → 3.13
Done as planned in the TODO list. MacPorts' Python 3.13.15 (`/opt/local/bin/python3.13`)
instead of the python.org 3.11.6.
- 3.14 was not tried: a check showed at least one pinned package is only published for
  Python below 3.14.
- A test environment in `work/test/venv313` installed `requirements.txt` with numpy back at
  2.4.6 without problems (`pip check`: no broken requirements). The numpy cap from
  `mistral-common` only applies to Python ≤ 3.12.
- **Root certificates work** with the MacPorts Python, so Vosk's own model download would
  work there (tested with a request to alphacephei.com).
- All six engines on `vos/LeesWijs-bladerboek-33` under 3.13: Parakeet, wav2vec2, Vosk,
  Canary and Voxtral byte-identical to the 3.11 outputs in `work/`.
- Whisper differed, which led to a check whether Whisper, like Vosk, depends on earlier
  videos in the same run rather than being random. Two single runs happened to give
  identical words, but a third single run gave 15 different words and a different number
  of cues (134 instead of 112), while a run after another video matched the first. So
  Whisper really does vary at random between runs, as noted before; not an order effect.
- The old environment was renamed to `venv-py311/` as a fallback (still runs; git-ignored
  via `venv-*/`), and a fresh `venv/` was created with 3.13 (a venv can't simply be moved,
  its scripts contain their own path). In the new `venv`: Parakeet and Canary identical
  again, and `compare.py` and `align.py` give the same results.
- `requirements.txt` changes: numpy 2.3.5 → 2.4.6, plus `setuptools` and replacements for
  standard modules Python 3.13 removed (`audioop-lts`, `standard-aifc`, `standard-chunk`,
  `standard-sunau`), pulled in by the audio libraries. It now needs Python 3.13.

### Reproducible Whisper (and Vosk)
Whisper decodes greedily first, but when a stretch looks poor (too repetitive, or low
confidence) it retries at higher "temperatures", sampling words at random with an
unseeded generator (`mlx_whisper.transcribe`'s default `temperature=(0.0, 0.2, …, 1.0)`).
Two ways to make it reproducible were tested with `work/test/whisper_variants.py`:

- **Greedy only** (`temperature=0`): fully reproducible, alone and after another video.
  Same result as the default in 43 of 48 videos, but in 3 it loses text. In
  `vos/LeesWijs-bladerboek-35` it dropped a 20-second passage of seven sentences
  (3:26–3:48) and put an invented sentence in its place ("De volgende keer was het.").
  Against the vote of the other five engines: 9.6% WER, the default 9.1–9.2%. Rejected.
- **Retries kept, fixed seed** (`mx.random.seed(0)` before each video): reproducible for
  a single video (three runs byte-identical), but after another video in the same process
  the output still differs. So Whisper, like Vosk, carries some state from one video to
  the next, besides the randomness.

What was chosen, in `transcribe.py`: a fixed seed (`WHISPER_SEED`), and for Whisper and
Vosk (`ISOLATE_ENGINES`) a separate process per video when given several videos. The
Vosk loop in the shell from before is no longer needed.

Checks:
- Whisper on `vos/LeesWijs-bladerboek-33` alone and after `…-34`: byte-identical. Vosk the
  same way: identical to the existing per-process output in `work/vosk`.
- Seeded Whisper twice on all 48 videos: **all 96 files byte-identical** (`.srt` and
  `.words.json`). 574 s per run (545 s before; the extra processes cost about 30 s).
- Against the vote of the other five engines: 9.2% WER, the same as the default runs
  (9.1%, 9.2%). Reproducibility costs no accuracy.

Important finding along the way: **whether Whisper skips a passage is partly luck.** With
seed 0 it also loses the passage in video 35, like greedy; the earlier default run got it
by a lucky draw that shifted Whisper's 30-second windows. A fixed seed makes the luck
repeatable, not better. Robustness is a separate problem (see TODO).

The seeded output is now the standard `work/whisper/`; the previous one is in
`work/run2/whisper/`. The vote was regenerated; the scores against it are unchanged
(Whisper 6.8% WER, 5.3% CER; 28,688 words in the vote files).

### wav2vec2 with its language model (`wav2vec2-lm`)
The wav2vec2 model repo contains a Dutch 5-gram language model (`language_model/lm.binary`,
1.4 GB, plus a word list `unigrams.txt` of 1.4 million entries). The model author's own
results on Common Voice 6 (Dutch test set): WER 15.7% without it, 12.8% with it.

Setup problems and solutions:
- `pyctcdecode` 0.5.0 (latest, from before numpy 2) pins `numpy<2.0`, which would have
  downgraded numpy for all engines. Its code uses none of the numpy features that numpy 2
  removed, so it was installed with `--no-deps` (its other dependencies, `pygtrie` and
  `hypothesis`, normally). `pip check` now reports that pin; that is expected.
- `kenlm` 0.3.0 from PyPI, and also the GitHub source, fail to compile on Python 3.13: they
  ship C++ code that an old Cython generated, using CPython internals 3.13 removed
  (`_PyGC_FINALIZED`, `_PyDict_SetItem_KnownHash`, `_PyLong_AsByteArray`). Fixed by
  regenerating that code from `kenlm.pyx` with Cython 3.3.0 (in a throwaway environment)
  before building.
- Both are excluded from `requirements.txt` (`pip freeze --exclude kenlm --exclude
  pyctcdecode`), since `pip install -r` can't install them; the README has the steps.
- With `kenlm` and `pyctcdecode` installed, the `transformers` pipeline loads the language
  model on its own. `transcribe.py` now passes the feature extractor itself, which stops
  that, and adds the decoder only for the new engine `wav2vec2-lm`. The plain `wav2vec2`
  output stayed byte-identical.
- pyctcdecode caches the language model in `~/.cache/pyctcdecode`, not in the Hugging Face
  cache.

**René's observation about visible and invisible errors.** Without a language model,
wav2vec2's mistakes are visible ("aardepelburee"): a reader recognises the intended word.
A language model turns uncertainty into real words; when that word is wrong, the mistake
is invisible and only listening reveals it. WER counts both the same. So for each engine,
every wrong or extra word (against the vote) was checked against the language model's
word list: real word (invisible) or not (visible). Test video `vos/LeesWijs-bladerboek-33`:

| engine | WER vs vote | wrong words | real word (invisible) | non-word (visible) | missing |
|---|---|---|---|---|---|
| whisper | 2.1% | 13 | 12 | 1 | 8 |
| canary | 4.6% | 38 | 28 | 10 | 9 |
| parakeet | 4.2% | 33 | 29 | 4 | 10 |
| voxtral | 11.4% | 110 | 95 | 15 | 7 |
| wav2vec2 | 17.7% | 147 | 111 | 36 | 34 |
| wav2vec2-lm | 15.2% | 132 | 115 | 17 | 24 |
| vosk | 15.7% | 101 | 99 | 2 | 60 |

- The language model helps wav2vec2 somewhat and halves its visible mistakes, while its
  invisible ones rise slightly: just as René predicted.
- Vosk's mistakes are nearly always invisible: Kaldi can only output words from its fixed
  vocabulary. It also drops the most words.
- Caveat: the word list comes from web text and includes typos, so it counts some garbled
  words as real; the invisible counts are upper limits. It does contain "aardappelpuree"
  and "opblaasboot" but not "aardepelburee" or "oplaasboot". The vote (which included plain
  wav2vec2 here) is the yardstick, not a verified reference.
- "visvos" became "vis vos"; "aardepelburee" stayed, although "aardappelpuree" is in the
  word list: the beam search didn't reach it.

### wav2vec2-lm on all 48 videos
281 s for all 48 videos (plain wav2vec2: about 200 s), so the language model search is
cheap. Scored against the vote of the five other engines (without either wav2vec2, since
both share the acoustic model and would count double):

| | WER | wrong words | real word (invisible) | non-word (visible) | missing |
|---|---|---|---|---|---|
| wav2vec2 | 23.9% | 5,583 | 3,716 (67%) | 1,867 (33%) | 1,212 |
| wav2vec2-lm | 19.4% | 4,668 | 3,884 (83%) | 784 (17%) | 849 |

A relative gain of about 19%, in line with the author's own Common Voice results (15.7% →
12.8%). Visible mistakes drop by 58%, invisible ones rise by 5%. Still well behind Whisper
and Canary.

`wav2vec2-lm` replaces plain `wav2vec2` in the standard vote (never both). `align.py` got
`--lexicon FILE`, which splits each engine's wrong and extra words against the vote into
real words and non-words (a bug in the first version skipped windows where the vote was
empty, so Voxtral's invented sentences weren't counted). Six engines, all 48 videos:

| engine | WER vs vote | CER vs vote | wrong words | real word (invisible) | non-word (visible) |
|---|---|---|---|---|---|
| whisper | 6.7% | 5.2% | 647 | 82.5% | 17.5% |
| canary | 7.0% | 3.4% | 1,598 | 72.1% | 27.9% |
| voxtral | 25.8% | 26.6% | 7,185 | 84.5% | 15.5% |
| parakeet | 11.2% | 6.0% | 2,036 | 76.0% | 24.0% |
| wav2vec2-lm | 18.9% | 8.7% | 4,417 | 82.8% | 17.2% |
| vosk | 21.5% | 13.1% | 3,671 | 98.3% | 1.7% |

- Whisper makes by far the fewest mistakes, but they are mostly invisible.
- Canary's mistakes are the most often visible: easier to catch when correcting.
- Vosk's are almost never visible (fixed vocabulary).
- Same caveat as before: the word list is generous, so "real word" shares are upper limits.

The vote now has 28,656 words (it had 28,688 with plain wav2vec2).

### Voxtral Mini 4B Realtime (`voxtral-rt`)
`mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` (Apache-2.0, 8.3 GB in the cache),
through mlx-audio. The download used Hugging Face's newer "xet" transfer: the blob file
stays at 0 bytes until the end, while progress shows only in
`~/.cache/huggingface/xet/logs/`.

How it differs from Voxtral Mini 3B: it is a streaming model that emits one token per
80 ms of audio (12.5 per second), including the silent stretches, with a fixed delay of
480 ms. Consequences:
- A 10-minute video needs about 7,500 tokens, beyond mlx-audio's default limit of 4,096,
  so it gets the same 15-second pieces as Canary and Voxtral 3B (engine `voxtral-rt`).
- Since every token belongs to a fixed moment, real word timings could be derived from
  token positions; mlx-audio's streaming loop only reports text, so that would mean
  reimplementing about 30 lines of it. Postponed until the quality is known.
- **It is slow on this Mac**: 98 ms per token, so 20 s for 14 s of audio, slower than
  real time (about 8 hours for all 48 videos). Voxtral 3B is fast because it only emits
  the text (about 40 tokens per piece).

First tests on `vos/LeesWijs-bladerboek-33`: the 8–22 s piece was transcribed perfectly
("Vos en vis. Vos had aardappelpuree gemaakt. Dat bedacht hij zou erg lekker zijn met
gebakken vis."). **The music intro (0–7.5 s) gave no text at all**, where Voxtral 3B
invents its "mijnbouwplaats" sentence.

A first full-video test was stopped after 10 minutes to find out why it was slow.
René chose to run a sample first: the first video of each series (12 videos, 1.4 hours of
video), in `work/voxtral-rt/`. Evaluation plan: against the vote of the five engines
without either Voxtral (Whisper, Canary, Parakeet, wav2vec2-lm, Vosk), so the two Voxtrals
can be compared fairly on the same videos.

### Voxtral Realtime: results on the 12-video sample
The sample took from 20:47 to about 22:35. Processing speed about 1.3 s per second of
audio (1.41 for the first video, which includes loading the model and ran partly before
René disconnected his external display; the display probably mattered only a few percent).

Against the vote of the five engines without either Voxtral (Whisper, Canary, Parakeet,
wav2vec2-lm, Vosk; this favours those five, since they are part of it):

| whisper | canary | parakeet | wav2vec2-lm | vosk | voxtral | voxtral (cleaned) | voxtral-rt |
|---|---|---|---|---|---|---|---|
| 6.2% | 7.0% | 9.8% | 18.2% | 21.3% | 22.3% | 6.7% | 7.7% |

Per video (one per series), same yardstick:

| series | whisper | canary | parakeet | voxtral (cleaned) | voxtral-rt |
|---|---|---|---|---|---|
| balotje | 10.8% | 4.9% | 8.7% | 4.6% | 6.5% |
| beestje | 27.0% | 11.1% | 18.6% | 13.7% | 14.6% |
| bentje | 2.2% | 6.7% | 13.1% | 6.1% | 7.8% |
| eend | 2.3% | 5.6% | 11.1% | 6.1% | 5.6% |
| help | 9.3% | 7.7% | 15.3% | 7.7% | 6.1% |
| jake | 4.8% | 9.0% | 9.9% | 7.3% | 7.7% |
| kerst | 7.9% | 9.9% | 7.6% | 8.7% | 10.4% |
| lammetje | 10.6% | 5.6% | 4.3% | 3.1% | 6.5% |
| rinus | 7.0% | 4.7% | 5.6% | 6.4% | 7.5% |
| sint | 6.8% | 9.5% | 9.0% | 9.3% | 12.2% |
| tim | 5.8% | 4.8% | 26.2% | 8.8% | 7.8% |
| vos | 1.5% | 5.0% | 3.9% | 4.4% | 4.3% |

With Voxtral Realtime as a full member of the six-engine vote (instead of Voxtral 3B), on
the same 12 videos, it comes out best:

| engine | ≥ 3 of 5 support | WER vs vote | CER vs vote | wrong words | real word | non-word |
|---|---|---|---|---|---|---|
| whisper | 92.1% | 6.8% | 5.2% | 168 | 82.7% | 17.3% |
| canary | 90.5% | 7.0% | 3.0% | 386 | 74.4% | 25.6% |
| voxtral-rt | 91.5% | 6.6% | 2.9% | 329 | 74.2% | 25.8% |
| parakeet | 89.4% | 10.9% | 5.8% | 491 | 75.4% | 24.6% |
| wav2vec2-lm | 82.4% | 18.7% | 8.3% | 1,103 | 83.7% | 16.3% |
| vosk | 84.7% | 21.6% | 13.2% | 913 | 98.1% | 1.9% |

Conclusions:
- **Voxtral Realtime is at least as accurate as Whisper and Canary**, and needs no
  cleaning: it does not invent text on music or silence. Its most repeated cues are real
  refrains ("Ik moet op de tegels blijven, zei Tim.", "Vroeg Vos.").
- Like Canary, it handles the `beestje` openings that Whisper skips (14.6% there against
  Whisper's 27.0%), and it doesn't have Parakeet's problem with `tim`.
- Its drawback is speed: slower than real time on this Mac (about 1.3×), so the remaining
  36 videos (4.3 hours of video) would take about 5.6 hours.

### Voxtral Realtime on all 48 videos
The remaining 36 videos took 20,913 s (22:40 to about 04:25), about 1.35 s per second of
audio. Since it doesn't invent text, Voxtral Realtime replaces Voxtral 3B in the standard
six-engine vote (Whisper, Canary, Voxtral Realtime, Parakeet, wav2vec2-lm, Vosk). All 48
videos:

| engine | ≥ 3 of 5 support | WER vs vote | CER vs vote | wrong words | real word | non-word |
|---|---|---|---|---|---|---|
| whisper | 92.2% | 6.6% | 5.1% | 631 | 83.4% | 16.6% |
| canary | 90.5% | 7.0% | 3.3% | 1,615 | 72.1% | 27.9% |
| voxtral-rt | 91.4% | 6.2% | 2.8% | 1,290 | 76.7% | 23.3% |
| parakeet | 89.3% | 11.1% | 6.0% | 2,034 | 75.8% | 24.2% |
| wav2vec2-lm | 82.4% | 19.0% | 8.8% | 4,433 | 82.7% | 17.3% |
| vosk | 84.6% | 21.5% | 13.0% | 3,669 | 98.3% | 1.7% |

| series | whisper | canary | voxtral-rt | parakeet | wav2vec2-lm | vosk |
|---|---|---|---|---|---|---|
| balotje | 7.0% | 5.6% | 5.5% | 9.0% | 12.2% | 14.6% |
| beestje | 23.8% | 7.6% | 9.7% | 16.5% | 28.4% | 40.1% |
| bentje | 3.8% | 7.2% | 5.9% | 14.1% | 15.9% | 19.8% |
| eend | 2.6% | 6.5% | 4.2% | 13.0% | 18.3% | 17.0% |
| help | 8.5% | 8.9% | 4.3% | 15.3% | 17.4% | 11.3% |
| jake | 4.0% | 8.9% | 7.0% | 13.5% | 25.5% | 20.0% |
| kerst | 7.7% | 7.6% | 8.6% | 8.8% | 20.2% | 32.9% |
| lammetje | 11.1% | 5.1% | 5.6% | 5.6% | 15.4% | 23.5% |
| rinus | 5.0% | 8.1% | 5.9% | 12.8% | 16.7% | 11.6% |
| sint | 9.4% | 10.4% | 11.3% | 8.9% | 24.5% | 37.9% |
| tim | 4.1% | 4.5% | 4.7% | 26.1% | 23.4% | 17.9% |
| vos | 5.6% | 4.2% | 3.4% | 4.6% | 15.5% | 15.3% |

- **Voxtral Realtime has the lowest error rates** against the vote, and is the most
  consistent across series (at most 11.3%; Whisper goes up to 23.8% in `beestje`).
- Whisper makes by far the fewest wrong words (631), but WER also counts missing words,
  and Whisper skips passages.
- Voxtral Realtime is the closest to both Whisper and Canary (about 10.5% pairwise).
- As always: these are measured against the majority. Only the hand-corrected references
  will tell which engine is actually most accurate.

The vote files now have 28,637 words. The previous summary (with Voxtral 3B in the vote)
is kept in `work/test/summary-with-voxtral3b.txt`.

### Voxtral Realtime 4-bit
`mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` (3.1 GB, download 527 s), run by
`work/run_4bit.sh` right after the full-precision run: 04:28 to 06:13, output in
`work/voxtral-rt-4bit/`. Same engine (`--engine voxtral-rt --model …-4bit`).

Speed:

| | 12 sample videos (1.4 h) | other 36 (4.3 h) | per second of audio |
|---|---|---|---|
| full precision | about 6,500 s | 20,913 s | about 1.3 s |
| 4-bit | 1,527 s | 4,790 s | about 0.3 s |

**4.3× faster, and 3× faster than real time**: all 48 videos in 1 hour 45 minutes instead of
7.5 hours.

Accuracy, against the vote of the five engines without either Voxtral
(`work/test/vote-no-voxtral`): full precision 7.6% WER, 4-bit 7.7%. The two versions differ
in 2.5% of their words (no video is word-for-word identical; 28,195 against 28,202 words).
Per series within half a point, except `beestje` (12.1% → 14.2%):

| series | full | 4-bit |
|---|---|---|
| balotje | 6.8% | 6.8% |
| beestje | 12.1% | 14.2% |
| bentje | 7.2% | 7.8% |
| eend | 5.5% | 6.0% |
| help | 5.4% | 5.4% |
| jake | 9.1% | 8.7% |
| kerst | 9.4% | 9.0% |
| lammetje | 6.1% | 5.9% |
| rinus | 7.6% | 7.9% |
| sint | 12.9% | 12.8% |
| tim | 5.4% | 5.5% |
| vos | 4.5% | 4.7% |

No invented text either: its most repeated cues are the story refrains ("Vroeg Vos.",
"Vraagt Otter.", "Ik moet op de tegels blijven, zei Tim.").

**Conclusion: the 4-bit version is the practical choice** for Voxtral Realtime on this
Mac: nearly the same accuracy, faster than real time. The full-precision output stays the
one in the standard vote, since that is what was measured against the other engines.

## Reproducing

Every command used so far, grouped by purpose. Run from the repository root on an Apple
Silicon Mac with Python 3 and ffmpeg (`brew install ffmpeg`). The videos go in
`videos/<series>/*.mp4`; they are not in git. Commands that originally wrote to a temporary
folder are shown with `work/test/` instead.

### Setup

To recreate the environment (Python 3.13; MacPorts puts it in `/opt/local/bin`):

```sh
python3.13 -m venv venv
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

### Rerun with word timings

First a copy of the previous outputs, then all engines again. Vosk runs one process per
video (see "Reproducibility checks"). This is `work/run_words.sh`:

```sh
mkdir -p work/run1
for e in whisper parakeet wav2vec2 vosk canary voxtral; do cp -R work/$e work/run1/$e; done

for e in whisper parakeet wav2vec2 canary voxtral; do
  start=$SECONDS
  ./venv/bin/python transcribe.py --engine $e --output-dir work/$e videos/*/*.mp4 | grep -c -- '->'
  echo "$e done in $((SECONDS-start))s"
done
start=$SECONDS
for f in videos/*/*.mp4; do
  ./venv/bin/python transcribe.py --engine vosk --output-dir work/vosk "$f" > /dev/null
done
echo "vosk done in $((SECONDS-start))s"
```

### Aligning within time windows

```sh
./venv/bin/python align.py work/whisper work/canary work/voxtral work/parakeet \
    work/wav2vec2 work/vosk --out work/align --vote work/vote > work/align/summary.txt
```

Engines that write capitals and punctuation go first, so the vote takes their spelling.
To compare the rerun with the first run:

```sh
for e in whisper parakeet wav2vec2 vosk canary voxtral; do
  n=0
  for f in work/run1/$e/*.srt; do cmp -s "$f" "work/$e/$(basename $f)" || n=$((n+1)); done
  echo "$e: $n of 48 .srt files changed"
done
```

For the one-video test, the six `.words.json` files were first copied into one folder per
engine under `work/test/wordsets/`.

### Python upgrade

```sh
# certificates and version of the new Python
/opt/local/bin/python3.13 --version
/opt/local/bin/python3.13 -c "import urllib.request; urllib.request.urlopen('https://alphacephei.com/vosk/models/model-list.json'); print('certificates OK')"

# test environment with numpy back at 2.4.6
sed 's/^numpy==2.3.5$/numpy==2.4.6/' requirements.txt > work/test/req-new.txt
/opt/local/bin/python3.13 -m venv work/test/venv313
work/test/venv313/bin/pip install --upgrade pip
work/test/venv313/bin/pip install -r work/test/req-new.txt
work/test/venv313/bin/pip check

# every engine on the test video, compared with the 3.11 output
for e in whisper parakeet wav2vec2 vosk canary voxtral; do
  work/test/venv313/bin/python transcribe.py --engine $e --output-dir work/test/py313/$e \
      videos/vos/LeesWijs-bladerboek-33.mp4
  n=LeesWijs-bladerboek-33; [ $e = whisper ] && f=$n.srt || f=$n.$e.srt
  cmp -s work/test/py313/$e/$f work/$e/$f && echo "$e identical" || echo "$e differs"
done

# the switch
mv venv venv-py311
/opt/local/bin/python3.13 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r work/test/req-new.txt
./venv/bin/pip freeze > requirements.txt
```

The Whisper checks: two more single runs of the test video (`--output-dir
work/test/whisper-single2`) and one after another video in the same call
(`videos/vos/LeesWijs-bladerboek-34.mp4 videos/vos/LeesWijs-bladerboek-33.mp4`), compared by
word with `compare.error_counts`.

### Reproducible Whisper

`work/test/whisper_variants.py` (not in git) runs Whisper with `default`, `seed` or
`greedy` decoding on videos in one process: it calls `transcribe.run_whisper` with
`mlx_whisper.transcribe` wrapped to set `mx.random.seed(0)` first (seed) or pass
`temperature=0.0` (greedy).

```sh
V=videos/vos/LeesWijs-bladerboek-33.mp4; W=videos/vos/LeesWijs-bladerboek-34.mp4
for variant in seed greedy; do
  for run in 1 2; do
    ./venv/bin/python work/test/whisper_variants.py $variant work/test/wv/$variant-single$run $V
  done
  ./venv/bin/python work/test/whisper_variants.py $variant work/test/wv/$variant-after34 $W $V
done
./venv/bin/python work/test/whisper_variants.py greedy work/test/wv/greedy videos/*/*.mp4

# the vote of the other five engines as a yardstick
./venv/bin/python align.py work/canary work/voxtral work/parakeet work/wav2vec2 work/vosk \
    --vote work/test/vote5
./venv/bin/python compare.py work/test/wv/default-run1 work/test/wv/default-run2 \
    work/test/wv/greedy work/test/wv/seeded --reference work/test/vote5
```

(`default-run1` and `default-run2` are copies of the Whisper `.srt` files from `work/run1`
and from the rerun with word timings; `seeded` is a copy of `whisper-seeded-1`.)

With the fix in `transcribe.py`: alone against in a batch, then twice on everything:

```sh
./venv/bin/python transcribe.py --output-dir work/test/iso/single $V
./venv/bin/python transcribe.py --output-dir work/test/iso/batch $W $V
cmp work/test/iso/single/LeesWijs-bladerboek-33.srt work/test/iso/batch/LeesWijs-bladerboek-33.srt

for i in 1 2; do
  ./venv/bin/python transcribe.py --output-dir work/test/whisper-seeded-$i videos/*/*.mp4
done
n=0
for f in work/test/whisper-seeded-1/*; do
  cmp -s "$f" "work/test/whisper-seeded-2/$(basename $f)" || n=$((n+1))
done
echo "$n files differ"

mkdir -p work/run2 && mv work/whisper work/run2/whisper
cp -R work/test/whisper-seeded-1 work/whisper
./venv/bin/python align.py work/whisper work/canary work/voxtral work/parakeet \
    work/wav2vec2 work/vosk --out work/align --vote work/vote > work/align/summary.txt
```

### wav2vec2 with its language model

The author's evaluation results and the repo contents:

```sh
./venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
for f in HfApi().model_info(repo, files_metadata=True).siblings:
    print(f.rfilename, f.size)
for f in ["mozilla-foundation_common_voice_6_0_nl_test_eval_results_greedy.txt",
          "mozilla-foundation_common_voice_6_0_nl_test_eval_results.txt",
          "language_model/attrs.json"]:
    print(f, open(hf_hub_download(repo, f)).read())
PY
```

Packages (see the entry above for why they're installed this way):

```sh
./venv/bin/pip install --dry-run kenlm pyctcdecode      # would downgrade numpy to 1.26.4
./venv/bin/pip download --no-deps pyctcdecode==0.5.0 -d work/test/pyctc
unzip -p work/test/pyctc/pyctcdecode-0.5.0*.whl '*/METADATA' | grep '^Requires-Dist'
./venv/bin/pip install --no-deps pyctcdecode==0.5.0
./venv/bin/pip install pygtrie hypothesis

./venv/bin/pip install kenlm                             # fails on Python 3.13
cd work/test
curl -sSfL -o kenlm.zip https://github.com/kpu/kenlm/archive/master.zip && unzip -q kenlm.zip
/opt/local/bin/python3.13 -m venv cy && cy/bin/pip install cython
cy/bin/cython --cplus -3 kenlm-master/python/kenlm.pyx -o kenlm-master/python/kenlm.cpp
cd ../.. && ./venv/bin/pip install work/test/kenlm-master

./venv/bin/pip freeze --exclude kenlm --exclude pyctcdecode > requirements.txt
```

Test and full run:

```sh
V=videos/vos/LeesWijs-bladerboek-33.mp4
./venv/bin/python transcribe.py --engine wav2vec2 --output-dir work/test/lm/raw $V
cmp work/test/lm/raw/LeesWijs-bladerboek-33.wav2vec2.srt work/wav2vec2/LeesWijs-bladerboek-33.wav2vec2.srt
./venv/bin/python transcribe.py --engine wav2vec2-lm --output-dir work/test/lm/lm $V
./venv/bin/python transcribe.py --engine wav2vec2-lm --output-dir work/wav2vec2-lm videos/*/*.mp4
```

Visible and invisible errors (the word list is the language model's `unigrams.txt`,
copied from `~/.cache/pyctcdecode/.../language_model/`; `work/test/lm/one/<engine>/` holds
each engine's `.srt` for the test video, `work/test/lm/one/vote/` the vote):

```sh
cp ~/.cache/pyctcdecode/models--jonatasgrosman--wav2vec2-large-xlsr-53-dutch/snapshots/*/language_model/unigrams.txt \
    work/test/nl-words.txt
./venv/bin/python - <<'PY'
from pathlib import Path
import jiwer
import compare as c
lexicon = {w.strip().lower() for w in open("work/test/nl-words.txt", encoding="utf-8")}
ref = c.words_of(c.read_srt(Path("work/test/lm/one/vote/LeesWijs-bladerboek-33.srt")))
for e in ["whisper", "canary", "voxtral", "parakeet", "wav2vec2", "wav2vec2-lm", "vosk"]:
    hyp = c.words_of(c.read_srt(next(Path(f"work/test/lm/one/{e}").glob("*.srt"))))
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    wrong = [w for ch in out.alignments[0] if ch.type in ("substitute", "insert")
             for w in hyp[ch.hyp_start_idx:ch.hyp_end_idx]]
    real = sum(w in lexicon for w in wrong)
    errors = out.substitutions + out.deletions + out.insertions
    print(e, f"{errors/len(ref):.1%}", len(wrong), real, len(wrong) - real, out.deletions)
PY
```

Scoring both wav2vec2 variants against the other five, and the standard run:

```sh
./venv/bin/python align.py work/whisper work/canary work/voxtral work/parakeet work/vosk \
    --vote work/test/vote-no-w2v
./venv/bin/python compare.py work/wav2vec2 work/wav2vec2-lm --reference work/test/vote-no-w2v

./venv/bin/python align.py work/whisper work/canary work/voxtral work/parakeet \
    work/wav2vec2-lm work/vosk --out work/align --vote work/vote \
    --lexicon work/test/nl-words.txt > work/align/summary.txt
```

### Voxtral Mini 4B Realtime

```sh
./venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16')"

# one piece, with the model's own timing output (work/test/vos33.wav: the test video's
# audio, extracted with ffmpeg as in "Test runs on one video")
./venv/bin/python - <<'PY'
import soundfile
from mlx_audio.stt.utils import load
model = load("mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16")
audio, rate = soundfile.read("work/test/vos33.wav", dtype="float32")
print(model.generate(audio[8*rate:22*rate], verbose=True).text)
print(repr(model.generate(audio[0:int(7.5*rate)]).text))   # the music intro
PY

# the sample: first video of each series
for d in videos/*/; do ls "$d"*.mp4 | head -1; done > work/test/rt-sample.txt
./venv/bin/python transcribe.py --engine voxtral-rt --output-dir work/voxtral-rt \
    $(cat work/test/rt-sample.txt)
```

Evaluating the sample (`work/test/vote-no-voxtral`: the vote without either Voxtral;
`work/test/rt12/<engine>/` holds each engine's `.srt` files for the 12 sample videos, copied
from `work/`):

```sh
./venv/bin/python align.py work/whisper work/canary work/parakeet work/wav2vec2-lm work/vosk \
    --vote work/test/vote-no-voxtral
./venv/bin/python compare.py work/test/rt12/whisper work/test/rt12/canary \
    work/test/rt12/parakeet work/test/rt12/wav2vec2-lm work/test/rt12/vosk \
    work/test/rt12/voxtral work/test/rt12/voxtral-clean work/test/rt12/voxtral-rt \
    --reference work/test/rt12/ref

# Voxtral Realtime as a full member of the vote (align.py uses the 12 videos all have)
./venv/bin/python align.py work/whisper work/canary work/voxtral-rt work/parakeet \
    work/wav2vec2-lm work/vosk --lexicon work/test/nl-words.txt
```

The remaining 36 videos:

```sh
ls videos/*/*.mp4 | grep -v -x -F -f work/test/rt-sample.txt > work/test/rt-rest.txt
./venv/bin/python transcribe.py --engine voxtral-rt --output-dir work/voxtral-rt \
    $(cat work/test/rt-rest.txt)
```

The 4-bit version, chained after the full-precision run (`work/run_4bit.sh`, started with the
PID of the running full-precision job; it downloads the model first, waits for that process
to end, then transcribes the 12 sample videos and then the other 36):

```sh
MODEL=mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
./venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
while kill -0 $PID 2>/dev/null; do sleep 60; done
for list in work/test/rt-sample.txt work/test/rt-rest.txt; do
  ./venv/bin/python transcribe.py --engine voxtral-rt --model $MODEL \
      --output-dir work/voxtral-rt-4bit $(cat $list)
done
```

The standard comparison with Voxtral Realtime instead of Voxtral 3B:

```sh
./venv/bin/python align.py work/whisper work/canary work/voxtral-rt work/parakeet \
    work/wav2vec2-lm work/vosk --out work/align --vote work/vote \
    --lexicon work/test/nl-words.txt > work/align/summary.txt
```

Comparing the 4-bit and full-precision versions:

```sh
./venv/bin/python compare.py work/voxtral-rt work/voxtral-rt-4bit --reference work/test/vote-no-voxtral
```

(The per-series table used `compare.load_engine`, `words_of` and `error_counts` on the same
folders, grouping videos by their folder under `videos/`.)
