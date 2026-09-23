#!/usr/bin/env python3
"""Compare engines word by word within short time windows, without a pivot engine.

Usage:
    ./venv/bin/python align.py work/whisper work/parakeet work/wav2vec2 work/vosk \\
        work/canary work/voxtral --out work/align --vote work/vote

Each argument is a folder with the .words.json files that transcribe.py writes next
to its .srt files; the folder name is used as the engine's name.

How it works:
  1. Each video is cut into windows of at most 15 s at the quietest moments, with
     split_at_pauses from transcribe.py (the same pieces Canary and Voxtral use).
  2. Every engine's words go into the window that holds the midpoint of their timing.
     Words in different windows are never aligned with each other.
  3. Within each window, every pair of engines is aligned (word-level edit distance).
     Each word's "support" is the number of other engines that have exactly that word
     at that position.
  4. Within each window, all engines are also aligned together into columns, one per
     word position (ROVER-style progressive alignment). It starts from the engine
     closest to all others in that window and adds the others closest first. Each
     column is decided by majority vote; a column where most engines have no word is
     dropped. Ties go to the starting engine.

Output:
  - Printed: per engine, how many of its words are supported by a majority of the other
    engines and how many by none; each engine's word and character error rate (WER,
    CER) against the majority vote, overall and per series (the folder a video is in);
    and pairwise WER computed per window next to the whole-file WER (a big difference
    means words land in the wrong window). CER is less sensitive to words written
    together or apart ("daarnet"/"daar net").
  - With --out: one Markdown file per video: per window the majority-vote text and a
    table of the word positions where engines disagree.
  - With --vote: one .srt per video with the majority-vote transcript, as a starting
    point for hand-correcting. Spelling and punctuation come from the first engine (in
    the order given) that has the winning word; timing is the median of the engines
    with real word timings.

Words are compared lowercased and without punctuation (see normalize in compare.py).
"""

import argparse
import json
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import jiwer

from compare import error_counts, normalize
from transcribe import extract_audio, split_at_pauses, words_to_cues, write_srt

# A word supported by at most this many other engines is marked in the Markdown output.
MARK_SUPPORT = 1

# Canary and Voxtral words carry the timing of their whole 15 s piece. Timings this
# long are not used for the majority-vote subtitles.
MAX_WORD_SECONDS = 3.0


@dataclass
class Token:
    norm: str   # lowercased, no punctuation: what is compared
    orig: str   # as the engine wrote it, when it normalizes to a single token
    start: float
    end: float


def load_words(folder: Path) -> dict[str, list[tuple[float, float, str]]]:
    words = {}
    for path in sorted(folder.glob("*.words.json")):
        words[path.name.split(".")[0]] = [
            (w["start"], w["end"], w["word"]) for w in json.loads(path.read_text("utf-8"))]
    return words


def find_video(name: str) -> Path | None:
    return next(Path("videos").glob(f"*/{name}.mp4"), None)


def windows_of(video: Path) -> list[tuple[float, float]]:
    import soundfile

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio(video, wav)
        audio, rate = soundfile.read(wav, dtype="float32")
    return [(a / rate, b / rate) for a, b in split_at_pauses(audio, rate)]


def split_into_windows(words: list[tuple[float, float, str]],
                       windows: list[tuple[float, float]]) -> list[list[Token]]:
    """Tokens per window, placed by the midpoint of their timing."""
    result: list[list[Token]] = [[] for _ in windows]
    i = 0
    # Sort on start time only: Canary's and Voxtral's words all share their piece's
    # timing, and a stable sort keeps them in spoken order.
    for start, end, word in sorted(words, key=lambda w: w[0]):
        mid = (start + end) / 2
        while i < len(windows) - 1 and mid >= windows[i][1]:
            i += 1
        parts = normalize(word)
        for part in parts:
            result[i].append(Token(part, word if len(parts) == 1 else part, start, end))
    return result


def norms(tokens: list[Token]) -> list[str]:
    return [t.norm for t in tokens]


def alignment(a: list[str], b: list[str]):
    """jiwer's alignment chunks of b against a; empty when either side is empty."""
    if not a or not b:
        return []
    return jiwer.process_words(" ".join(a), " ".join(b)).alignments[0]


def matches(a: list[str], b: list[str]) -> list[bool]:
    """For every word of a: is it aligned to an identical word of b?"""
    hit = [False] * len(a)
    for chunk in alignment(a, b):
        if chunk.type == "equal":
            for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
                hit[k] = True
    return hit


def vote(column: dict[str, Token | None], first: str) -> str | None:
    """Majority word of a column (None: most engines have no word here)."""
    counts = Counter(t.norm if t else None for t in column.values())
    best = max(counts.values())
    winners = [w for w, c in counts.items() if c == best]
    if len(winners) == 1:
        return winners[0]
    pivot = column.get(first)
    return pivot.norm if pivot else None


def multi_align(tokens: dict[str, list[Token]]) -> tuple[str, list[dict[str, Token | None]]]:
    """Align all engines of one window into columns. Returns the starting engine too."""
    names = list(tokens)
    distance = {n: sum(error_counts(norms(tokens[n]), norms(tokens[o]))[0]
                       for o in names if o != n) for n in names}
    first = min(names, key=lambda n: distance[n])
    columns: list[dict[str, Token | None]] = [{first: t} for t in tokens[first]]
    order = sorted((n for n in names if n != first),
                   key=lambda n: error_counts(norms(tokens[first]), norms(tokens[n]))[0])
    added = [first]

    for name in order:
        consensus = [vote(c, first) or next(t.norm for t in c.values() if t) for c in columns]
        mine = tokens[name]
        if not columns:
            columns = [{name: t} for t in mine]
        elif mine:
            new: list[dict[str, Token | None]] = []
            done = 0  # columns copied so far
            for chunk in alignment(consensus, norms(mine)):
                new += columns[done:chunk.ref_start_idx]
                done = chunk.ref_start_idx
                if chunk.type == "insert":
                    new += [{name: t} for t in mine[chunk.hyp_start_idx:chunk.hyp_end_idx]]
                    continue
                for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    column = columns[k]
                    if chunk.type != "delete":
                        column[name] = mine[chunk.hyp_start_idx + k - chunk.ref_start_idx]
                    new.append(column)
                done = chunk.ref_end_idx
            columns = new + columns[done:]
        added.append(name)
        for column in columns:
            for n in added:
                column.setdefault(n, None)
    return first, columns


def voted_tokens(first: str, columns: list[dict[str, Token | None]],
                 names: list[str]) -> list[tuple[float | None, float | None, str]]:
    """The majority-vote words of a window, with original spelling and timing."""
    result = []
    for column in columns:
        winner = vote(column, first)
        if winner is None:
            continue
        voters = [column[n] for n in names if column.get(n) and column[n].norm == winner]
        timed = [t for t in voters if t.end - t.start <= MAX_WORD_SECONDS]
        start = statistics.median(t.start for t in timed) if timed else None
        end = statistics.median(t.end for t in timed) if timed else None
        result.append((start, end, voters[0].orig))
    return result


def fill_times(words, window: tuple[float, float]) -> list[tuple[float, float, str]]:
    """Give words without timing (only voted for by Canary/Voxtral) a spot in between."""
    filled, prev_end = [], window[0]
    for i, (start, end, word) in enumerate(words):
        if start is None:
            nxt = next((s for s, _, _ in words[i + 1:] if s is not None), window[1])
            start, end = prev_end, max(prev_end, min(nxt, prev_end + 0.4))
        filled.append((start, end, word))
        prev_end = end
    return filled


def char_errors(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    """(character edits, reference characters) between two word lists."""
    ref, hyp = " ".join(reference), " ".join(hypothesis)
    if not ref:
        return len(hyp), 0
    if not hyp:
        return len(ref), len(ref)
    out = jiwer.process_characters(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, len(ref)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02}:{s:02}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engines", nargs="+", type=Path,
                    help="folders of .words.json files, one per engine")
    ap.add_argument("--out", type=Path, help="write one Markdown file per video here")
    ap.add_argument("--vote", type=Path, help="write majority-vote .srt files here")
    args = ap.parse_args()

    engines = {folder.name: load_words(folder) for folder in args.engines}
    names = list(engines)
    videos = sorted(set.intersection(*(set(w) for w in engines.values())))
    if not videos:
        print("no video has .words.json files from every engine", file=sys.stderr)
        return 1
    print(f"{len(videos)} videos with all of: {', '.join(names)}")
    for folder in (args.out, args.vote):
        if folder:
            folder.mkdir(parents=True, exist_ok=True)

    others = len(names) - 1
    majority = others // 2 + 1
    stats = {n: {"words": 0, "majority": 0, "none": 0} for n in names}
    # Per series and engine: [word errors, vote words, char errors, vote chars].
    by_series: dict[str, dict[str, list[int]]] = {}
    pair_errors = {p: [0, 0, 0] for p in combinations(names, 2)}  # windowed, whole, ref words

    for video in videos:
        path = find_video(video)
        if path is None:
            print(f"skip (video not found): {video}", file=sys.stderr)
            continue
        windows = windows_of(path)
        series = by_series.setdefault(path.parent.name, {n: [0, 0, 0, 0] for n in names})
        per_window = {n: split_into_windows(engines[n][video], windows) for n in names}

        for a, b in pair_errors:
            whole_a = [t.norm for win in per_window[a] for t in win]
            whole_b = [t.norm for win in per_window[b] for t in win]
            pair_errors[(a, b)][1] += error_counts(whole_a, whole_b)[0]
            pair_errors[(a, b)][2] += len(whole_a)
            for wa, wb in zip(per_window[a], per_window[b]):
                pair_errors[(a, b)][0] += error_counts(norms(wa), norms(wb))[0]

        lines = [f"# {video}", "",
                 f"{len(windows)} windows. Per window: the majority vote, then every word "
                 f"position where the engines disagree (– means no word).", ""]
        vote_srt_words = []
        for i, window in enumerate(windows):
            tokens = {n: per_window[n][i] for n in names}
            for n in names:
                support = [0] * len(tokens[n])
                for o in names:
                    if o != n:
                        for k, hit in enumerate(matches(norms(tokens[n]), norms(tokens[o]))):
                            support[k] += hit
                s = stats[n]
                s["words"] += len(support)
                s["majority"] += sum(c >= majority for c in support)
                s["none"] += sum(c == 0 for c in support)

            if not any(tokens.values()):
                continue
            first, columns = multi_align(tokens)
            voted = voted_tokens(first, columns, names)
            vote_norms = [w for _, _, word in voted for w in normalize(word)]
            for n in names:
                e, total = error_counts(vote_norms, norms(tokens[n]))
                ce, ctotal = char_errors(vote_norms, norms(tokens[n]))
                for k, v in enumerate((e, total, ce, ctotal)):
                    series[n][k] += v
            vote_srt_words += fill_times(voted, window)

            lines += [f"## {fmt_time(window[0])}–{fmt_time(window[1])}", "",
                      f"**Vote:** {' '.join(w for _, _, w in voted) or '∅'}", ""]
            disagree = [c for c in columns if len({t.norm if t else None
                                                   for t in c.values()}) > 1]
            if disagree:
                lines += ["| " + " | ".join(names) + " | vote |",
                          "|" + "---|" * (len(names) + 1)]
                for c in disagree:
                    cells = [c[n].norm if c.get(n) else "–" for n in names]
                    lines.append("| " + " | ".join(cells) + f" | {vote(c, first) or '–'} |")
                lines.append("")
        if args.out:
            (args.out / f"{video}.md").write_text("\n".join(lines), encoding="utf-8")
        if args.vote:
            write_srt(words_to_cues(vote_srt_words), args.vote / f"{video}.srt")

    totals = {n: [sum(by_series[x][n][k] for x in by_series) for k in range(4)]
              for n in names}
    print(f"\nSupport per engine (a word is supported by an other engine that has the same "
          f"word at the same place), and error rates against the majority vote\n")
    print(f"| {'engine':<10} | {'words':>7} | {f'≥{majority} of {others}':>9} | {'none':>6} "
          f"| {'WER':>6} | {'CER':>6} |")
    print(f"|{'-' * 12}|{'-' * 9}|{'-' * 11}|{'-' * 8}|{'-' * 8}|{'-' * 8}|")
    for n, s in stats.items():
        w = max(s["words"], 1)
        e, total, ce, ctotal = totals[n]
        print(f"| {n:<10} | {s['words']:>7} | {s['majority'] / w:>9.1%} | {s['none'] / w:>6.1%} "
              f"| {e / max(total, 1):>6.1%} | {ce / max(ctotal, 1):>6.1%} |")

    print("\nWER against the majority vote per series\n")
    print(f"| {'series':<10} | " + " | ".join(f"{n:>8}" for n in names) + " |")
    print(f"|{'-' * 12}|" + "|".join("-" * 10 for _ in names) + "|")
    for x in sorted(by_series):
        cells = [f"{e / max(t, 1):>8.1%}" for e, t, _, _ in (by_series[x][n] for n in names)]
        print(f"| {x:<10} | " + " | ".join(cells) + " |")

    print("\nPairwise WER (second engine against first), per window vs whole file\n")
    print(f"| {'pair':<22} | {'windows':>8} | {'whole':>7} |")
    print(f"|{'-' * 24}|{'-' * 10}|{'-' * 9}|")
    for (a, b), (windowed, whole, total) in pair_errors.items():
        t = max(total, 1)
        print(f"| {a + ' / ' + b:<22} | {windowed / t:>8.1%} | {whole / t:>7.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
