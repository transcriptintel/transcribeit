# transcribeit

A Rust CLI for speech-to-text transcription. Supports local inference via [whisper.cpp](https://github.com/ggerganov/whisper.cpp), remote transcription via OpenAI-compatible APIs, and Azure OpenAI.

Accepts any audio or video format — FFmpeg handles conversion automatically.

## Prerequisites

- Rust 1.80+ (edition 2024)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- C/C++ toolchain and CMake (for building whisper.cpp)

## Quick start

```bash
# Build
cargo build --release

# Download a model
transcribeit download-model -s base

# Transcribe any audio/video file (single file, directory, or glob)
transcribeit run -i recording.mp3 -m base

# Or pass an explicit model path
transcribeit run -i recording.mp3 -m .cache/ggml-base.bin

# Process a directory
transcribeit run -i samples/ -m .cache/ggml-base.bin -o ./output

# Process a glob
transcribeit run --input "samples/**/*.{mp3,wav,mp4}" -p azure -o ./output

# With VTT output and segmentation
transcribeit run -i meeting.mp4 -m .cache/ggml-base.bin \
  --output-format vtt --segment -o ./output

# Transcribe via OpenAI API
transcribeit run -p openai -i recording.mp3

# Transcribe via Azure OpenAI
transcribeit run -p azure -i recording.mp3 \
  --azure-deployment my-whisper -b https://myresource.openai.azure.com

# Force language and normalize before transcription
transcribeit run -i recording.wav -m .cache/ggml-base.bin --language en --normalize
```

## Features

- **Any input format** — MP3, MP4, WAV, FLAC, OGG, etc. FFmpeg converts to mono 16kHz WAV automatically.
- **3 providers** — Local whisper.cpp, OpenAI API, Azure OpenAI. Extensible via the `Transcriber` trait.
- **Language hinting** — Pass `--language` to force local and API transcription language.
- **FFmpeg audio normalization** — Optional `--normalize` to apply loudnorm before transcription.
- **Silence-based segmentation** — Splits long audio at silence boundaries for better accuracy and API compatibility.
- **Auto-split for API limits** — Files exceeding 25MB are automatically segmented when using remote providers.
- **Progress spinner** — Shows live terminal feedback during transcription (single file and segmented mode).
- **Parallel API segment transcription** — Multiple segment requests can be processed concurrently with `--segment-concurrency`.
- **VTT output** — WebVTT subtitle files with timestamps.
- **SRT output** — SubRip subtitle files with timestamps.
- **Text output** — Writes plain text transcript to stdout by default and `<input>.txt` when `--output-dir` is specified.
- **JSON manifest** — Processing metadata, segment details, and statistics.
- **Model caching** — Loaded whisper models are cached in memory for batch processing.
- **Model management** — Download and list GGML models from Hugging Face.

## Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
MODEL_CACHE_DIR=.cache
OPENAI_API_KEY=sk-your_key_here
AZURE_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
AZURE_DEPLOYMENT_NAME=whisper
AZURE_API_VERSION=2024-06-01
TRANSCRIBEIT_MAX_RETRIES=5
TRANSCRIBEIT_REQUEST_TIMEOUT_SECS=120
TRANSCRIBEIT_RETRY_WAIT_BASE_SECS=10
TRANSCRIBEIT_RETRY_WAIT_MAX_SECS=120
```

## Documentation

See the [docs](docs/) folder for detailed documentation:

- [Architecture](docs/architecture.md) — Project structure, trait design, processing pipeline
- [CLI Reference](docs/cli-reference.md) — All commands, options, and examples
- [Provider behavior](docs/provider-behavior.md) — OpenAI vs Azure argument differences
- [Troubleshooting](docs/troubleshooting.md) — Common setup/runtime issues and fixes
- [Performance benchmarks](docs/performance-benchmarks.md) — Reproducible measurement plan and templates
