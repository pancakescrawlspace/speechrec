#!/usr/bin/env python3
"""Generate Dutch .srt subtitles for a video using Whisper on Apple Silicon (mlx-whisper).

Usage:
    ./venv/bin/python transcribe.py beestje/LeesWijs-bladerboek-5.mp4 [more videos...]

The .srt is written next to each video with the same basename.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx_whisper
from mlx_whisper.writers import get_writer

MODEL = "mlx-community/whisper-large-v3-turbo"

# Whisper hallucinates on music/silence, producing cues like "***" or lone
# punctuation. Drop any cue with no letters in it.
def has_speech(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def extract_audio(video: Path, wav: Path) -> None:
    """Whisper expects 16 kHz mono; let ffmpeg do the decoding."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )


def transcribe(video: Path, language: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio(video, wav)
        result = mlx_whisper.transcribe(
            str(wav),
            path_or_hf_repo=MODEL,
            language=language,
            task="transcribe",
            # Whisper's built-in prompt nudges spelling/style; a short Dutch
            # sentence helps it commit to Dutch orthography from the start.
            initial_prompt="Een voorleesverhaal voor kinderen.",
            condition_on_previous_text=False,  # reduces runaway repetition
            verbose=False,
        )

    result["segments"] = [s for s in result["segments"] if has_speech(s["text"])]

    writer = get_writer("srt", str(video.parent))
    writer(result, video.stem)
    out = video.with_suffix(".srt")
    print(f"{video} -> {out}  ({len(result['segments'])} cues)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--language", default="nl")
    args = ap.parse_args()

    for video in args.videos:
        if not video.exists():
            print(f"skip (not found): {video}", file=sys.stderr)
            continue
        transcribe(video, args.language)
    return 0


if __name__ == "__main__":
    sys.exit(main())
