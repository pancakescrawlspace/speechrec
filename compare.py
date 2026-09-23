#!/usr/bin/env python3
"""Compare .srt output of several speech recognition engines, with each other and with references.

Usage:
    ./venv/bin/python compare.py work/whisper work/parakeet work/wav2vec2 \\
        [--reference references] [--review work/review]

Each argument is a folder of .srt files from one engine; the folder name is used as
the engine's name. Files are matched on the video name (the part before the first
dot), so LeesWijs-bladerboek-5.srt and LeesWijs-bladerboek-5.parakeet.srt match.

Reports:
  - Agreement: word error rate (WER) of every engine measured against every other.
    This only shows how much engines differ, not which one is right.
  - Accuracy (with --reference): WER of every engine against hand-corrected .srt files.
  - Review sheets (with --review): one Markdown file per video listing the cues of the
    first engine where other engines disagree, as a guide for hand-correcting. With
    --min-disagree N, only cues where at least N other engines disagree are listed.

Before comparing, text is lowercased and stripped of punctuation, since engines
differ in capitalisation and punctuation (wav2vec2 has neither).
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer


@dataclass
class Cue:
    number: int
    start: str
    end: str
    text: str


def read_srt(path: Path) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").replace("\r", "")):
        lines = block.strip().split("\n")
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start, end = (t.strip() for t in lines[1].split("-->"))
        cues.append(Cue(int(lines[0]), start, end, " ".join(lines[2:])))
    return cues


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    # Keep apostrophes inside words (zo'n, 's avonds), drop all other punctuation.
    text = re.sub(r"[^\w\s']|(?<!\w)'|'(?!\w)", " ", text)
    return text.split()


def load_engine(folder: Path) -> dict[str, list[Cue]]:
    return {p.name.split(".")[0]: read_srt(p) for p in sorted(folder.glob("*.srt"))}


def words_of(cues: list[Cue]) -> list[str]:
    return [w for c in cues for w in normalize(c.text)]


def error_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    """Return (errors, reference length); jiwer refuses empty input."""
    if not reference:
        return len(hypothesis), 0
    if not hypothesis:
        return len(reference), len(reference)
    out = jiwer.process_words(" ".join(reference), " ".join(hypothesis))
    return out.substitutions + out.deletions + out.insertions, len(reference)


def wer_table(title: str, rows: dict[str, dict[str, list[str]]],
              columns: dict[str, dict[str, list[str]]]) -> None:
    """Print WER of each column engine (hypothesis) against each row (reference)."""
    names = list(columns)
    print(f"\n{title}\n")
    print(f"| {'reference ↓ / engine →':<24} | " + " | ".join(f"{n:>9}" for n in names) + " |")
    print(f"|{'-' * 26}|" + "|".join("-" * 11 for _ in names) + "|")
    for ref_name, ref in rows.items():
        cells = []
        for name in names:
            hyp = columns[name]
            videos = [v for v in ref if v in hyp]
            if ref_name == name or not videos:
                cells.append("—")
                continue
            errors = total = 0
            for v in videos:
                e, n = error_counts(ref[v], hyp[v])
                errors += e
                total += n
            cells.append(f"{errors / total:.1%}" if total else "—")
        print(f"| {ref_name:<24} | " + " | ".join(f"{c:>9}" for c in cells) + " |")


def aligned_cues(base: list[Cue], other: list[str]) -> list[tuple[bool, list[str]]]:
    """For every base cue: whether `other` differs there, and `other`'s words for it."""
    spans, words = [], []
    for cue in base:
        w = normalize(cue.text)
        spans.append((len(words), len(words) + len(w)))
        words.extend(w)

    if not words or not other:
        return [(bool(words or other), []) for _ in base]
    chunks = jiwer.process_words(" ".join(words), " ".join(other)).alignments[0]

    result = []
    for a, b in spans:
        differs, hyp = False, []
        for ch in chunks:
            if ch.type == "insert":
                # Words the other engine adds go with the base cue they precede
                # (or the last cue, when they come at the very end).
                pos = ch.ref_start_idx
                if a <= pos < b or (pos == len(words) and b == len(words) and a < b):
                    differs = True
                    hyp.extend(other[ch.hyp_start_idx:ch.hyp_end_idx])
                continue
            lo, hi = max(a, ch.ref_start_idx), min(b, ch.ref_end_idx)
            if lo >= hi:
                continue
            if ch.type != "equal":
                differs = True
            if ch.type != "delete":
                # equal and substitute chunks map base words one-to-one.
                offset = ch.hyp_start_idx - ch.ref_start_idx
                hyp.extend(other[lo + offset:hi + offset])
        result.append((differs, hyp))
    return result


def write_review(video: str, engines: dict[str, dict[str, list[Cue]]], out: Path,
                 min_disagree: int) -> tuple[int, int]:
    base_name, *other_names = engines
    base = engines[base_name][video]
    others = {n: aligned_cues(base, words_of(engines[n].get(video, []))) for n in other_names}

    sections = []
    for i, cue in enumerate(base):
        differing = [n for n in other_names if others[n][i][0]]
        if len(differing) < min_disagree:
            continue
        lines = [f"## {cue.number} · {cue.start} → {cue.end} · "
                 f"{len(differing)}/{len(other_names)} differ", "",
                 f"- **{base_name}:** {cue.text}"]
        for n in other_names:
            differs, hyp = others[n][i]
            mark = "" if differs else " (same)"
            lines.append(f"- {n}{mark}: {' '.join(hyp) or '∅'}")
        sections.append("\n".join(lines))

    header = (f"# {video}\n\n{len(sections)} of {len(base)} {base_name} cues differ from at "
              f"least {min_disagree} of: {', '.join(other_names)}.\n"
              f"Other engines' text is lowercased and without punctuation.\n")
    out.write_text(header + "\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    return len(sections), len(base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engines", nargs="+", type=Path, help="folders of .srt files, one per engine")
    ap.add_argument("--reference", type=Path, help="folder of hand-corrected .srt files")
    ap.add_argument("--review", type=Path, help="write per-video review sheets to this folder")
    ap.add_argument("--min-disagree", type=int, default=1, metavar="N",
                    help="review sheets list a cue when at least N other engines differ "
                         "(default: %(default)s)")
    args = ap.parse_args()

    engines = {folder.name: load_engine(folder) for folder in args.engines}
    for name, videos in engines.items():
        print(f"{name}: {len(videos)} videos")
    words = {name: {v: words_of(c) for v, c in videos.items()} for name, videos in engines.items()}

    wer_table("Agreement (WER of each engine against each other engine)", words, words)

    if args.reference:
        refs = {v: words_of(c) for v, c in load_engine(args.reference).items()}
        print(f"\nreferences: {len(refs)} videos")
        wer_table("Accuracy (WER against hand-corrected references)", {"references": refs}, words)

    if args.review:
        if len(engines) < 2:
            print("review sheets need at least two engines", file=sys.stderr)
            return 1
        args.review.mkdir(parents=True, exist_ok=True)
        base_name = next(iter(engines))
        flagged = total = 0
        for video in engines[base_name]:
            f, t = write_review(video, engines, args.review / f"{video}.md", args.min_disagree)
            flagged += f
            total += t
        print(f"\nreview sheets: {len(engines[base_name])} written to {args.review}, "
              f"{flagged} of {total} {base_name} cues flagged ({flagged / max(total, 1):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
