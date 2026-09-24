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
    wav2vec2-lm  The same, decoded with the model's own Dutch 5-gram language model
              (needs kenlm and pyctcdecode, see JOURNAL.md)
    vosk      Vosk (Kaldi) with its large Dutch model; lowercase, no punctuation
    canary    NVIDIA Canary 1B v2 via mlx-audio, on pieces of the audio (see
              split_at_pauses); word timings by aligning its text to its CTC model
    voxtral   Mistral Voxtral Mini 3B via mlx-audio; no timestamps either
    voxtral-rt  Mistral Voxtral Mini 4B Realtime via mlx-audio, on the same pieces;
              word timings from its token positions

The .srt is written next to each video with the same basename, unless
--output-dir is given. Engines other than whisper add their name to the file
name (video.parakeet.srt) so they never overwrite the whisper subtitles.

Next to each .srt goes a .words.json (video.words.json, video.parakeet.words.json)
with every recognised word and its timing in seconds, for comparing engines word
by word. Voxtral 3B has no word timings: its words get the start and end of the
audio piece they came from. Canary's words are timed with its CTC model, Voxtral
Realtime's from the positions of its tokens.
"""

import argparse
import bisect
import functools
import itertools
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
    "wav2vec2-lm": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    # Not on Hugging Face: vosk downloads its models to ~/.cache/vosk by name.
    "vosk": "vosk-model-nl-spraakherkenning-0.6",
    "canary": "CogniSoftOrg/canary-1b-v2-mlx-bf16",
    "voxtral": "mlx-community/Voxtral-Mini-3B-2507-bf16",
    "voxtral-rt": "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
}

# Whisper's built-in prompt nudges spelling/style; a short Dutch sentence helps
# it commit to Dutch orthography from the start.
PROMPT = "Een voorleesverhaal voor kinderen."

# When a stretch decodes poorly, Whisper retries it with random sampling. A fixed
# seed makes those retries, and so the output, the same on every run.
WHISPER_SEED = 0

# Whisper and Vosk give different output for a video depending on which videos were
# transcribed before it in the same process. With several videos, each one gets a
# process of its own, so the output is the same alone or in a batch.
ISOLATE_ENGINES = {"whisper", "vosk"}

# Parakeet, wav2vec2 and Vosk return word timings, not subtitle-sized segments, so we
# cut cues ourselves: at a pause, after the end of a sentence, or when a cue gets too
# long to read.
MAX_CUE_WORDS = 14
MAX_CUE_SECONDS = 7.0
CUE_PAUSE_SECONDS = 0.8

# Engines without timestamps of their own (Canary, both Voxtrals) get the audio in pieces
# of at most this length, cut at the quietest moment.
MAX_PIECE_SECONDS = 15.0

# Voxtral Realtime emits one token per 1,280 samples (80 ms) of 16 kHz audio. A word's
# first token comes once most of the word has been heard, so its start is earlier than
# its first token by more than its end is earlier than its last. Offsets in seconds,
# calibrated against Whisper and Parakeet (see JOURNAL.md).
RAW_SAMPLES_PER_VOXTRAL_RT_TOKEN = 1280
VOXTRAL_RT_START_OFFSET = 0.64
VOXTRAL_RT_END_OFFSET = 0.20

# Canary's words are timed by aligning its text to the frames (80 ms) of its CTC model.
# A token's CTC spike is short, so its end is moved later (a negative offset). Offsets
# in seconds, calibrated against Whisper and Parakeet (see JOURNAL.md).
CANARY_START_OFFSET = 0.12
CANARY_END_OFFSET = -0.04


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
    import mlx.core as mx
    import mlx_whisper

    mx.random.seed(WHISPER_SEED)

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
def load_wav2vec2(model: str, with_lm: bool):
    import torch
    from transformers import AutoFeatureExtractor, pipeline

    # Passing the feature extractor ourselves stops the pipeline from loading the
    # language model on its own whenever kenlm and pyctcdecode are installed.
    kwargs = {"feature_extractor": AutoFeatureExtractor.from_pretrained(model)}
    if with_lm:
        from pyctcdecode import BeamSearchDecoderCTC

        kwargs["decoder"] = BeamSearchDecoderCTC.load_from_hf_hub(
            model, allow_patterns=["language_model/*", "alphabet.json"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=model, device=device, **kwargs)


def run_wav2vec2(wav: Path, model: str, language: str, prompt: str,
                 with_lm: bool = False) -> tuple[list[Cue], list[Word]]:
    import soundfile

    # The model is Dutch-only; --language and --prompt don't apply.
    audio, rate = soundfile.read(wav, dtype="float32")
    result = load_wav2vec2(model, with_lm)(
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


@functools.cache
def load_canary_ctc(model: str):
    """Canary's separate CTC model (in the ctc/ folder of the model repo): encoder and head.

    Its own recognised text is poor; it is only used to time Canary's text. The encoder
    is the same FastConformer as Canary's, without biases, so mlx-audio's classes and
    weight mapping for Canary load it.
    """
    import mlx.core as mx
    import numpy as np
    from huggingface_hub import snapshot_download
    from mlx_audio.stt.models.canary.canary import CanaryEncoder, Model
    from mlx_audio.stt.models.canary.config import ModelConfig

    path = Path(snapshot_download(model, allow_patterns=["ctc/*"])) / "ctc"
    weights = Model._sanitize_nemo(None, mx.load(str(path / "model.safetensors")))
    # Its config.json is a copy of Canary's: the CTC encoder has fewer layers.
    config = json.loads((path / "config.json").read_text())
    config["encoder"]["use_bias"] = False
    config["encoder"]["n_layers"] = 1 + max(
        int(k.split(".")[3]) for k in weights if k.startswith("encoder.conformer.layers."))
    config = ModelConfig.from_dict(config)
    encoder = CanaryEncoder(config)
    encoder.load_weights([(k[len("encoder."):], v) for k, v in weights.items()
                          if k.startswith("encoder.")])
    encoder.eval()
    head = np.load(path / "ctc_head.npz")  # blank is the last class
    return config, encoder, mx.array(head["W"]), mx.array(head["b"])


def ctc_align(log_probs, tokens: list[int], blank: int) -> list[tuple[int, int]] | None:
    """The best path of tokens through CTC frames (Viterbi): first and last frame per token.

    None if the tokens don't fit in the frames.
    """
    import numpy as np

    n_frames, n = len(log_probs), 2 * len(tokens) + 1
    states = np.full(n, blank)
    states[1::2] = tokens
    # A token may follow the token before the previous blank directly, unless equal.
    skip = np.zeros(n, bool)
    skip[3::2] = states[3::2] != states[1:-2:2]
    score = np.full(n, -np.inf)
    score[:2] = log_probs[0, states[:2]]
    back = np.zeros((n_frames, n), np.int8)  # how many states back each came from
    for t in range(1, n_frames):
        options = np.stack([score,
                            np.concatenate([[-np.inf], score[:-1]]),
                            np.where(skip, np.concatenate([[-np.inf] * 2, score[:-2]]), -np.inf)])
        back[t] = np.argmax(options, axis=0)
        score = options[back[t], np.arange(n)] + log_probs[t, states]
    state = n - 1 if n == 1 or score[-1] >= score[-2] else n - 2
    if not np.isfinite(score[state]):
        return None
    path = [0] * n_frames
    for t in range(n_frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    spans = [[None, None] for _ in tokens]
    for t, state in enumerate(path):
        if state % 2:
            span = spans[state // 2]
            span[0] = t if span[0] is None else span[0]
            span[1] = t
    return [tuple(span) for span in spans]


def canary_words(model: str, audio, rate: int, text: str) -> list[Word] | None:
    """Time Canary's words in a piece of audio, from 0, by aligning them to the CTC model.

    A word runs from the first frame of its first token to the last of its last, moved
    by CANARY_START_OFFSET and CANARY_END_OFFSET, and starts no earlier than the
    previous word ends. None if the text can't be aligned.
    """
    import mlx.core as mx
    import numpy as np
    from types import SimpleNamespace
    from mlx_audio.stt.models.canary.canary import Model

    config, encoder, weight, bias = load_canary_ctc(model)
    mel = Model._preprocess_audio(SimpleNamespace(config=config), audio).astype(mx.bfloat16)
    frames, _ = encoder(mel, mx.array([mel.shape[1]]))
    logits = frames[0].astype(mx.float32) @ weight.T + bias
    log_probs = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))

    sp = load_mlx_audio(model)._tokenizer.sp
    tokens = sp.encode(text)
    pieces = [sp.id_to_piece(t).replace("\u2581", " ") for t in tokens]
    spans = ctc_align(log_probs, tokens, blank=weight.shape[0] - 1) if tokens else None
    if spans is None:
        return None
    # The character offset where each token's text ends, to find the token of a character.
    ends = list(itertools.accumulate(len(p) for p in pieces))
    spelled = "".join(pieces)
    found = list(re.finditer(r"\S+", spelled))
    if len(found) != len(text.split()):
        return None
    step = config.encoder.subsampling_factor * config.preprocessor.window_stride
    words, previous_end = [], 0.0
    for m, word in zip(found, text.split()):
        first = bisect.bisect_right(ends, m.start())
        last = bisect.bisect_right(ends, m.end() - 1)
        end = max(0.0, (spans[last][1] + 1) * step - CANARY_END_OFFSET)
        start = min(max(spans[first][0] * step - CANARY_START_OFFSET, previous_end), end)
        words.append((start, end, word))
        previous_end = end
    return words


def run_canary(wav: Path, model: str, language: str,
               prompt: str) -> tuple[list[Cue], list[Word]]:
    import soundfile

    # Canary has no prompt; --prompt doesn't apply. Its words are timed with its CTC
    # model; a piece whose text can't be aligned gives its words the piece's timing.
    audio, rate = soundfile.read(wav, dtype="float32")
    words = []
    for a, b in split_at_pauses(audio, rate):
        text = load_mlx_audio(model).generate(
            audio[a:b], source_lang=language, target_lang=language, use_pnc=True).text
        timed = canary_words(model, audio[a:b], rate, text) if text.strip() else []
        if timed is None:
            print(f"  could not align {a / rate:.1f}-{b / rate:.1f} s: {text!r}", file=sys.stderr)
            timed = [(0.0, (b - a) / rate, w) for w in text.split()]
        words += [(a / rate + s, min(b / rate, a / rate + e), w) for s, e, w in timed]
    return words_to_cues(words), words


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


def voxtral_rt_tokens(model, audio, max_tokens: int = 4096) -> list[int]:
    """Voxtral Realtime's output tokens for a piece of audio, one per 80 ms of audio.

    A copy of the decoding loop in mlx-audio's Model.generate (0.5.5, greedy), which
    only returns the text; the index of each token is what gives it its time.
    """
    import mlx.core as mx

    adapter_out, n_audio, prompt_len, logits, cache, enc_chunk_gen, _ = (
        model._encode_and_prefill(audio))
    adapter_len = adapter_out.shape[0]
    eos = model.config.eos_token_id
    generated = []
    next_tok = mx.argmax(logits)
    mx.async_eval(next_tok)
    for pos in range(prompt_len, n_audio):
        token = int(next_tok.item())
        generated.append(token)
        if token == eos or len(generated) > max_tokens:
            break
        if enc_chunk_gen is not None and pos >= adapter_len:
            try:
                chunk_adapter = model.encoder.downsample_and_project(next(enc_chunk_gen))
                mx.eval(chunk_adapter)
                adapter_out = mx.concatenate([adapter_out, chunk_adapter], axis=0)
                adapter_len = adapter_out.shape[0]
            except StopIteration:
                enc_chunk_gen = None
        embed = model.decoder.embed_token(token)
        if pos < adapter_len:
            embed = adapter_out[pos] + embed
        h, cache = model.decoder.forward(embed[None, :], start_pos=pos, cache=cache)
        next_tok = mx.argmax(model.decoder.logits(h.squeeze(0)))
        mx.async_eval(next_tok)
        if len(generated) % 256 == 0:
            mx.clear_cache()
    else:
        generated.append(int(next_tok.item()))
    mx.clear_cache()
    if generated and generated[-1] == eos:
        generated.pop()
    return generated


def voxtral_rt_words(model, audio, rate: int) -> tuple[str, list[Word]]:
    """Voxtral Realtime's text for a piece of audio, and its words timed from 0.

    Token i is emitted after hearing the audio up to about i × 80 ms plus the
    model's delay. A word ends VOXTRAL_RT_END_OFFSET before the end of its last
    token, and starts VOXTRAL_RT_START_OFFSET before its first token, but not
    before the previous word ends.
    """
    tokens = voxtral_rt_tokens(model, audio)
    tokenizer = model._tokenizer
    # The byte offset where each token's text ends, to find the token of any byte.
    ends, n = [], 0
    for token in tokens:
        n += len(tokenizer.token_bytes(token))
        ends.append(n)
    raw = tokenizer.decode(tokens)
    text = raw.strip()
    lead = len(raw) - len(raw.lstrip())

    def token_of(char: int) -> int:
        return bisect.bisect_right(ends, len(raw[:lead + char].encode()))

    step = RAW_SAMPLES_PER_VOXTRAL_RT_TOKEN / rate
    words, previous_end = [], 0.0
    for m in re.finditer(r"\S+", text):
        first, last = token_of(m.start()), token_of(m.end() - 1)
        end = max(0.0, (last + 1) * step - VOXTRAL_RT_END_OFFSET)
        start = min(max(first * step - VOXTRAL_RT_START_OFFSET, previous_end), end)
        words.append((start, end, m.group()))
        previous_end = end
    return text, words


def run_voxtral_rt(wav: Path, model: str, language: str,
                   prompt: str) -> tuple[list[Cue], list[Word]]:
    import soundfile

    # A streaming model: it detects the language itself and takes no prompt. It could
    # take a whole video, but emits one token per 80 ms of audio (7,500 for a 10-minute
    # video, beyond its default limit of 4,096); so it gets the same pieces as Canary
    # and Voxtral. Its word timings come from the positions of its tokens.
    audio, rate = soundfile.read(wav, dtype="float32")
    words = []
    for a, b in split_at_pauses(audio, rate):
        _, piece = voxtral_rt_words(load_mlx_audio(model), audio[a:b], rate)
        words += [(a / rate + s, min(b / rate, a / rate + e), w) for s, e, w in piece]
    return words_to_cues(words), words


ENGINES = {
    "whisper": run_whisper,
    "parakeet": run_parakeet,
    "wav2vec2": run_wav2vec2,
    "wav2vec2-lm": functools.partial(run_wav2vec2, with_lm=True),
    "vosk": run_vosk,
    "canary": run_canary,
    "voxtral": run_voxtral,
    "voxtral-rt": run_voxtral_rt,
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

    isolate = args.engine in ISOLATE_ENGINES and len(args.videos) > 1
    for video in args.videos:
        if not video.exists():
            print(f"skip (not found): {video}", file=sys.stderr)
        elif isolate:
            command = [sys.executable, __file__, str(video), "--engine", args.engine,
                       "--model", model, "--language", args.language, "--prompt", args.prompt]
            if args.output_dir:
                command += ["--output-dir", str(args.output_dir)]
            subprocess.run(command, check=True)
        else:
            transcribe(video, args.engine, model, args.language, args.prompt, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
