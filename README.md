# speechrec

A proof of concept for generating subtitles for our company's video files automatically.
It uses OpenAI's Whisper speech recognition model (`whisper-large-v3-turbo`) through
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), so it runs
locally on Apple Silicon. No audio leaves the machine.

For each input video, `transcribe.py` writes an `.srt` subtitle file with the same
basename next to the video.

## Requirements

- A Mac with Apple Silicon (M1 or later). MLX does not run on Intel Macs or other platforms.
- Python 3
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (for example `brew install ffmpeg`)

## Setup

```sh
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The first run downloads the Whisper model from Hugging Face (about 1.5 GB) and caches it
in `~/.cache/huggingface`.

## Usage

```sh
./venv/bin/python transcribe.py path/to/video.mp4 [more videos...]
```

The default language is Dutch (`nl`). To transcribe another language, pass `--language`
with an ISO 639-1 code:

```sh
./venv/bin/python transcribe.py --language en path/to/video.mp4
```

Other options:

| Option | Default | Purpose |
|---|---|---|
| `--prompt TEXT` | `"Een voorleesverhaal voor kinderen."` | Short description of the material; steers spelling and style. Pass `''` for no prompt. |
| `--model REPO` | `mlx-community/whisper-large-v3-turbo` | Hugging Face repo of any MLX Whisper model. |
| `--output-dir DIR` | next to each video | Write the `.srt` files to this folder instead (created if missing). |

The script skips files it cannot find and continues with the rest.

## How it works

1. ffmpeg extracts the audio track as 16 kHz mono WAV, the format Whisper expects.
2. Whisper transcribes the audio. Two settings help with quality:
   - A short Dutch initial prompt ("Een voorleesverhaal voor kinderen.") steers the
     model toward Dutch spelling from the first segment. This default prompt is tuned for
     read-aloud children's stories, so pass `--prompt` if your material
     is different.
   - `condition_on_previous_text=False` stops the model from repeating the same line
     over and over.
3. The script drops cues that contain no letters. Whisper sometimes produces cues like
   `***` or lone punctuation during music or silence.
4. The remaining segments are written out as an `.srt` file.

## Limitations

This is a proof of concept, so:

- It only runs on Apple Silicon.
- The cue filtering and decoding settings are hard-coded in `transcribe.py`.
- The generated subtitles have not been checked against a quality benchmark. Review
  them before publishing.
- There are no tests.

## Repository notes

Video files (`*.mp4`, `*.wmv`) and generated subtitles (`*.srt`) are excluded from git
through `.gitignore`. The videos are large, and both the videos and their subtitles are
copyrighted content, so keep them out of the repository.
