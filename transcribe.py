#!/usr/bin/env python3
"""Generate Dutch .srt subtitles for a video with a choice of speech recognition engines.

Usage:
    ./venv/bin/python transcribe.py videos/beestje/LeesWijs-bladerboek-5.mp4 [more videos...]
    ./venv/bin/python transcribe.py --engine parakeet --output-dir work/parakeet videos/vos/*.mp4

Engines:
    whisper   Whisper large-v3-turbo via mlx-whisper (default, the baseline)
    parakeet  NVIDIA Parakeet TDT 0.6B v3 via parakeet-mlx
    wav2vec2  A Dutch fine-tuned wav2vec2 (XLSR-53) via transformers; lowercase,
              no punctuation
    vosk      Vosk (Kaldi) with its large Dutch model; lowercase, no punctuation
    canary    NVIDIA Canary 1B v2 via mlx-audio; no timestamps of its own, so cue
              timing is approximate (see split_at_pauses)
    voxtral   Mistral Voxtral Mini 3B via mlx-audio; no timestamps either

The .srt is written next to each video with the same basename, unless
--output-dir is given. Engines other than whisper add their name to the file
name (video.parakeet.srt) so they never overwrite the whisper subtitles.

Next to each .srt goes a .words.json (video.words.json, video.parakeet.words.json)
with every recognised word and its timing in seconds, for comparing engines word
by word. Canary and Voxtral have no word timings: their words get the start and
end of the audio piece they came from.
"""

import argparse
import functools
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = {
    "whisper": "mlx-community/whisper-large-v3-turbo",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
    "wav2vec2": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    # Not on Hugging Face: vosk downloads its models to ~/.cache/vosk by name.
    "vosk": "vosk-model-nl-spraakherkenning-0.6",
    "canary": "CogniSoftOrg/canary-1b-v2-mlx-bf16",
    "voxtral": "mlx-community/Voxtral-Mini-3B-2507-bf16",
}

# Whisper's built-in prompt nudges spelling/style; a short Dutch sentence helps
# it commit to Dutch orthography from the start.
PROMPT = "Een voorleesverhaal voor kinderen."

# Parakeet, wav2vec2 and Vosk return word timings, not subtitle-sized segments, so we
# cut cues ourselves: at a pause, after the end of a sentence, or when a cue gets too
# long to read.
MAX_CUE_WORDS = 14
MAX_CUE_SECONDS = 7.0
CUE_PAUSE_SECONDS = 0.8

# Engines without timestamps (Canary, Voxtral) get the audio in pieces of at most this
# length, cut at the quietest moment; each piece's text is spread over its span.
MAX_PIECE_SECONDS = 15.0


@dataclass
class Cue:
    start: float
    end: float
    text: str


# (start, end, word), times in seconds.
Word = tuple[float, float, str]


# Whisper hallucinates on music/silence, producing cues like "***" or lone
# punctuation. Drop any cue with no letters in it.
def has_speech(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def extract_audio(video: Path, wav: Path) -> None:
    """All engines expect 16 kHz mono; let ffmpeg do the decoding."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )


def words_to_cues(words: list[Word]) -> list[Cue]:
    cues: list[Cue] = []
    current: list[Word] = []
    for word in words:
        if current and (
            current[-1][2].rstrip("\"')»”").endswith((".", "!", "?", "…"))
            or word[0] - current[-1][1] >= CUE_PAUSE_SECONDS
            or len(current) >= MAX_CUE_WORDS
            or word[1] - current[0][0] > MAX_CUE_SECONDS
        ):
            cues.append(Cue(current[0][0], current[-1][1], " ".join(w[2] for w in current)))
            current = []
        current.append(word)
    if current:
        cues.append(Cue(current[0][0], current[-1][1], " ".join(w[2] for w in current)))
    return cues


def split_at_pauses(audio, rate: int) -> list[tuple[int, int]]:
    """Cut audio into pieces of at most MAX_PIECE_SECONDS, as sample ranges.

    Each cut goes at the quietest 10 ms (averaged over 200 ms) in the second half
    of the allowed span, so it usually falls between sentences or words.
    """
    import numpy as np

    frame = rate // 100
    n = len(audio) // frame
    energy = np.sqrt(np.mean(audio[:n * frame].reshape(n, frame) ** 2, axis=1))
    energy = np.convolve(energy, np.ones(20) / 20, mode="same")
    max_frames = int(MAX_PIECE_SECONDS * 100)

    pieces, start = [], 0
    while n - start > max_frames:
        lo = start + max_frames // 2
        cut = lo + int(np.argmin(energy[lo:start + max_frames]))
        pieces.append((start * frame, cut * frame))
        start = cut
    pieces.append((start * frame, len(audio)))
    return pieces


def text_to_cues(text: str, start: float, end: float) -> list[Cue]:
    """Split text into sentence-sized cues, sharing out the time by text length."""
    parts = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        parts += [" ".join(words[i:i + MAX_CUE_WORDS])
                  for i in range(0, len(words), MAX_CUE_WORDS)]
    total = sum(len(p) for p in parts)
    cues, t = [], start
    for part in parts:
        duration = (end - start) * len(part) / total
        cues.append(Cue(t, t + duration, part))
        t += duration
    return cues


def run_whisper(wav: Path, model: str, language: str,
                prompt: str) -> tuple[list[Cue], list[Word]]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=model,
        language=language,
        task="transcribe",
        initial_prompt=prompt or None,
        condition_on_previous_text=False,  # reduces runaway repetition
        word_timestamps=True,
        verbose=False,
    )
    segments = result["segments"]
    words = [(w["start"], w["end"], w["word"].strip()) for s in segments for w in s["words"]]
    return [Cue(s["start"], s["end"], s["text"].strip()) for s in segments], words


@functools.cache
def load_parakeet(model: str):
    from parakeet_mlx import from_pretrained

    return from_pretrained(model)


def run_parakeet(wav: Path, model: str, language: str,
                 prompt: str) -> tuple[list[Cue], list[Word]]:
    from parakeet_mlx import DecodingConfig, SentenceConfig

    # Parakeet detects the language itself; --language and --prompt don't apply.
    config = DecodingConfig(sentence=SentenceConfig(
        max_words=MAX_CUE_WORDS,
        max_duration=MAX_CUE_SECONDS,
        silence_gap=CUE_PAUSE_SECONDS,
    ))
    # Chunking keeps memory bounded on long videos.
    result = load_parakeet(model).transcribe(
        wav, decoding_config=config, chunk_duration=120.0, overlap_duration=15.0)
    # Parakeet's tokens are word pieces; a piece starting with a space starts a word.
    words: list[Word] = []
    for token in result.tokens:
        if words and not token.text.startswith(" "):
            start, _, text = words[-1]
            words[-1] = (start, token.end, text + token.text)
        else:
            words.append((token.start, token.end, token.text.strip()))
    return [Cue(s.start, s.end, s.text.strip()) for s in result.sentences], words


@functools.cache
def load_wav2vec2(model: str):
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=model, device=device)


def run_wav2vec2(wav: Path, model: str, language: str,
                 prompt: str) -> tuple[list[Cue], list[Word]]:
    import soundfile

    # The model is Dutch-only; --language and --prompt don't apply.
    audio, rate = soundfile.read(wav, dtype="float32")
    result = load_wav2vec2(model)(
        {"raw": audio, "sampling_rate": rate},
        chunk_length_s=30, stride_length_s=5, return_timestamps="word",
    )
    words = [(c["timestamp"][0], c["timestamp"][1], c["text"]) for c in result["chunks"]]
    return words_to_cues(words), words


@functools.cache
def load_vosk(model: str):
    import vosk

    vosk.SetLogLevel(-1)
    return vosk.Model(model_name=model)


def run_vosk(wav: Path, model: str, language: str,
             prompt: str) -> tuple[list[Cue], list[Word]]:
    import wave

    from vosk import KaldiRecognizer

    # The model is Dutch-only; --language and --prompt don't apply.
    result = []
    with wave.open(str(wav), "rb") as audio:
        recognizer = KaldiRecognizer(load_vosk(model), audio.getframerate())
        recognizer.SetWords(True)
        while data := audio.readframes(4000):
            if recognizer.AcceptWaveform(data):
                result += json.loads(recognizer.Result()).get("result", [])
        result += json.loads(recognizer.FinalResult()).get("result", [])
    # <unk> marks sounds the model couldn't match to any word.
    words = [(w["start"], w["end"], w["word"]) for w in result if w["word"] != "<unk>"]
    return words_to_cues(words), words


@functools.cache
def load_mlx_audio(model: str):
    from mlx_audio.stt.utils import load

    return load(model)


def run_canary(wav: Path, model: str, language: str,
               prompt: str) -> tuple[list[Cue], list[Word]]:
    import soundfile

    # Canary has no prompt; --prompt doesn't apply.
    audio, rate = soundfile.read(wav, dtype="float32")
    cues, words = [], []
    for a, b in split_at_pauses(audio, rate):
        text = load_mlx_audio(model).generate(
            audio[a:b], source_lang=language, target_lang=language, use_pnc=True).text
        cues += text_to_cues(text, a / rate, b / rate)
        words += [(a / rate, b / rate, w) for w in text.split()]
    return cues, words


def run_voxtral(wav: Path, model: str, language: str,
                prompt: str) -> tuple[list[Cue], list[Word]]:
    import soundfile

    # Voxtral takes no prompt in transcription mode; --prompt doesn't apply.
    audio, rate = soundfile.read(wav, dtype="float32")
    piece = wav.with_name("piece.wav")
    cues, words = [], []
    for a, b in split_at_pauses(audio, rate):
        # mlx-audio's Voxtral only accepts a file path, not an array.
        soundfile.write(piece, audio[a:b], rate)
        text = load_mlx_audio(model).generate(str(piece), language=language).text
        cues += text_to_cues(text, a / rate, b / rate)
        words += [(a / rate, b / rate, w) for w in text.split()]
    return cues, words


ENGINES = {
    "whisper": run_whisper,
    "parakeet": run_parakeet,
    "wav2vec2": run_wav2vec2,
    "vosk": run_vosk,
    "canary": run_canary,
    "voxtral": run_voxtral,
}


def srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(cues: list[Cue], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{cue.text}\n\n")


def transcribe(video: Path, engine: str, model: str, language: str, prompt: str,
               output_dir: Path | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio(video, wav)
        cues, words = ENGINES[engine](wav, model, language, prompt)

    cues = [c for c in cues if has_speech(c.text)]

    name = video.stem if engine == "whisper" else f"{video.stem}.{engine}"
    out = (output_dir or video.parent) / f"{name}.srt"
    write_srt(cues, out)
    (out.parent / f"{name}.words.json").write_text(json.dumps(
        [{"start": round(a, 3), "end": round(b, 3), "word": w} for a, b, w in words],
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{video} -> {out}  ({len(cues)} cues, {len(words)} words)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--engine", choices=ENGINES, default="whisper",
                    help="speech recognition engine (default: %(default)s)")
    ap.add_argument("--language", default="nl",
                    help="whisper, canary and voxtral only (default: %(default)s)")
    ap.add_argument("--model",
                    help="model for the chosen engine: a Hugging Face repo, or a Vosk "
                         "model name (default: see DEFAULT_MODELS in this script)")
    ap.add_argument("--prompt", default=PROMPT,
                    help="whisper only: initial prompt describing the material; "
                         "pass '' for none (default: %(default)r)")
    ap.add_argument("--output-dir", type=Path,
                    help="write .srt files here instead of next to each video")
    args = ap.parse_args()

    model = args.model or DEFAULT_MODELS[args.engine]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for video in args.videos:
        if not video.exists():
            print(f"skip (not found): {video}", file=sys.stderr)
            continue
        transcribe(video, args.engine, model, args.language, args.prompt, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
