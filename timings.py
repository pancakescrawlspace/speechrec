#!/usr/bin/env python3
"""Compare one engine's word timings with other engines', on the words they agree on.

Usage:
    ./venv/bin/python timings.py work/canary work/whisper work/parakeet
    ./venv/bin/python timings.py work/test/rt-offset0 work/whisper --shift 0.64 0.20

The first folder holds the .words.json files of the engine to check; the others are
the engines to compare it with. Per video, the words (normalized as in compare.py) are
aligned with jiwer, and for every pair of equal words the difference in start time
and in end time is taken (checked engine minus the other). Prints the median,
quartiles and 5–95 percentiles of those differences, and the share within 0.2 s.

--shift START END first moves every word of the checked engine: its end END seconds
earlier, its start START seconds earlier but not before the previous word's end. With
timings made with transcribe.py's offsets set to 0, this tries out offsets without
transcribing again.
"""

import argparse
import json
import sys
from pathlib import Path

import jiwer
import numpy as np

from compare import normalize

Word = tuple[float, float, str]


def load(folder: Path) -> dict[str, list[Word]]:
    return {path.name.split(".")[0]: [(w["start"], w["end"], w["word"])
                                      for w in json.loads(path.read_text("utf-8"))]
            for path in sorted(folder.glob("*.words.json"))}


def shifted(words: list[Word], start_offset: float, end_offset: float) -> list[Word]:
    result, previous_end = [], 0.0
    for start, end, word in words:
        end = max(0.0, end - end_offset)
        result.append((min(max(start - start_offset, previous_end), end), end, word))
        previous_end = end
    return result


def normalized(words: list[Word]) -> list[Word]:
    return [(start, end, part) for start, end, word in words for part in normalize(word)]


def differences(checked: list[Word], other: list[Word]):
    """(start difference, end difference) for each pair of equal aligned words."""
    checked, other = normalized(checked), normalized(other)
    if not checked or not other:
        return
    out = jiwer.process_words(" ".join(w[2] for w in other), " ".join(w[2] for w in checked))
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                c, o = checked[chunk.hyp_start_idx + k], other[chunk.ref_start_idx + k]
                yield c[0] - o[0], c[1] - o[1]


def report(name: str, diffs: list[float]) -> None:
    if not diffs:
        print(f"{name:<22} no words in common")
        return
    d = np.array(diffs)
    p5, q1, median, q3, p95 = np.percentile(d, [5, 25, 50, 75, 95])
    print(f"{name:<22} {len(d):>6} words  median {median:+.3f}  "
          f"quartiles [{q1:+.3f}, {q3:+.3f}]  5–95% [{p5:+.3f}, {p95:+.3f}]  "
          f"within 0.2 s: {np.mean(abs(d) < 0.2):.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engine", type=Path, help="folder with the .words.json files to check")
    ap.add_argument("others", nargs="+", type=Path, help="folders of engines to compare with")
    ap.add_argument("--shift", nargs=2, type=float, metavar=("START", "END"),
                    help="move the checked words earlier by these offsets first")
    ap.add_argument("--videos", nargs="+", help="only these videos (names without extension)")
    args = ap.parse_args()

    checked = load(args.engine)
    if args.videos:
        checked = {v: w for v, w in checked.items() if v in args.videos}
    if args.shift:
        checked = {v: shifted(w, *args.shift) for v, w in checked.items()}
    for folder in args.others:
        other = load(folder)
        starts, ends = [], []
        for video, words in checked.items():
            for s, e in differences(words, other.get(video, [])):
                starts.append(s)
                ends.append(e)
        report(f"{folder.name} start", starts)
        report(f"{folder.name} end", ends)
    return 0


if __name__ == "__main__":
    sys.exit(main())
