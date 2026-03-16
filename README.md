# transcribeit

A Rust CLI for speech-to-text transcription. Supports local inference via [whisper.cpp](https://github.com/ggerganov/whisper.cpp), local inference via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), remote transcription via OpenAI-compatible APIs, and Azure OpenAI.

Accepts any audio or video format — FFmpeg handles conversion automatically.

## Prerequisites

- Rust 1.80+ (edition 2024)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- C/C++ toolchain and CMake (for building whisper.cpp)
- sherpa-onnx shared libraries (if using the `sherpa-onnx` provider) — set `SHERPA_ONNX_LIB_DIR` in `.env` to the directory containing them

## Quick start

```bash
# Build (reads SHERPA_ONNX_LIB_DIR from .env automatically via build.rs)
cargo build --release

# Build without sherpa-onnx (no shared library dependency needed)
cargo build --release --no-default-features

# Download a GGML model (default format, for --provider local)
transcribeit download-model -s base

# Download an ONNX model (for --provider sherpa-onnx)
transcribeit download-model -s base -f onnx

# List all downloaded models (GGML and ONNX)
transcribeit list-models

# Transcribe with local whisper.cpp (model alias resolves from MODEL_CACHE_DIR)
transcribeit run -i recording.mp3 -m base

# Transcribe with sherpa-onnx Whisper (auto-segments at ≤30s boundaries)
transcribeit run -p sherpa-onnx -i recording.mp3 -m base

# Transcribe with sherpa-onnx Moonshine (auto-detected from model files)
transcribeit run -p sherpa-onnx -i recording.mp3 -m moonshine-base

# Transcribe with sherpa-onnx SenseVoice (auto-detected from model files)
transcribeit run -p sherpa-onnx -i recording.mp3 -m sense-voice

# Or pass an explicit model path
transcribeit run -i recording.mp3 -m .cache/ggml-base.bin

# Process a directory (default output format is vtt)
transcribeit run -i samples/ -m base -o ./output

# Process a glob
transcribeit run --input "samples/**/*.{mp3,wav,mp4}" -p azure -o ./output

# Choose output format: text, vtt (default), or srt
transcribeit run -i meeting.mp4 -m base -f srt -o ./output

# Transcribe via OpenAI API
transcribeit run -p openai -i recording.mp3

# Transcribe via Azure OpenAI
transcribeit run -p azure -i recording.mp3 \
  --azure-deployment my-whisper -b https://myresource.openai.azure.com

# Force language and normalize before transcription
transcribeit run -i recording.wav -m base --language en --normalize

# VAD-based segmentation (speech-aware, avoids mid-word cuts)
transcribeit run -p sherpa-onnx -m base -i recording.mp3 --vad-model .cache/silero_vad.onnx

# Speaker diarization (2 speakers)
transcribeit run -i interview.mp3 -m base --speakers 2 \
  --diarize-segmentation-model .cache/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --diarize-embedding-model .cache/wespeaker_en_voxceleb_CAM++.onnx
```

## Features

- **Any input format** — MP3, MP4, WAV, FLAC, OGG, etc. FFmpeg converts to mono 16kHz WAV automatically.
- **4 providers** — Local whisper.cpp, sherpa-onnx, OpenAI API, Azure OpenAI. Extensible via the `Transcriber` trait.
- **3 model architectures via sherpa-onnx** — Whisper, Moonshine, and SenseVoice are auto-detected from the model directory contents. Just point `--model` at any supported model directory.
- **Model aliases** — `-m base`, `-m tiny`, etc. resolve from `MODEL_CACHE_DIR` for both `local` and `sherpa-onnx` providers. The sherpa-onnx resolver also supports glob matching (e.g., `-m moonshine-base`, `-m sense-voice`).
- **Language hinting** — Pass `--language` to force local and API transcription language.
- **FFmpeg audio normalization** — Optional `--normalize` to apply loudnorm before transcription.
- **VAD-based segmentation** — Speech-aware segmentation via Silero VAD (sherpa-onnx). Detects speech boundaries with padding and gap merging to avoid mid-word cuts. Use `--vad-model .cache/silero_vad.onnx`.
- **Silence-based segmentation** — Fallback segmentation via FFmpeg `silencedetect` for API providers or when VAD model is not available.
- **sherpa-onnx auto-segmentation** — Whisper ONNX models only support ≤30s per call; segmentation is enabled automatically.
- **sherpa-onnx is optional** — Enabled by default as a Cargo feature. Build without it: `cargo build --no-default-features`.
- **Auto-split for API limits** — Files exceeding 25MB are automatically segmented when using remote providers.
- **Progress spinner** — Shows live terminal feedback during transcription (single file and segmented mode).
- **Parallel API segment transcription** — Multiple segment requests can be processed concurrently with `--segment-concurrency`.
- **VTT output** (default) — WebVTT subtitle files with timestamps.
- **SRT output** — SubRip subtitle files with timestamps.
- **Text output** — Writes plain text transcript to stdout by default and `<input>.txt` when `--output-dir` is specified.
- **JSON manifest** — Processing metadata, segment details, and statistics.
- **Model caching** — Loaded whisper models are cached in memory for batch processing.
- **Model management** — Download and list both GGML and ONNX models. Use `--format ggml` (default) or `--format onnx` with `download-model`.

## Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
MODEL_CACHE_DIR=.cache
SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
OPENAI_API_KEY=sk-your_key_here
AZURE_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
AZURE_DEPLOYMENT_NAME=whisper
AZURE_API_VERSION=2024-06-01
TRANSCRIBEIT_MAX_RETRIES=5
TRANSCRIBEIT_REQUEST_TIMEOUT_SECS=120
TRANSCRIBEIT_RETRY_WAIT_BASE_SECS=10
TRANSCRIBEIT_RETRY_WAIT_MAX_SECS=120
VAD_MODEL=.cache/silero_vad.onnx
DIARIZE_SEGMENTATION_MODEL=.cache/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
DIARIZE_EMBEDDING_MODEL=.cache/wespeaker_en_voxceleb_CAM++.onnx
```

## License

This project is licensed under the [Business Source License 1.1](LICENSE).

- **Free** for non-commercial and evaluation use
- **Commercial/production use** requires a separate license — contact [TranscriptIntel](https://github.com/transcriptintel)
- Converts to **Apache 2.0** on March 16, 2030

## Documentation

See the [docs](docs/) folder for detailed documentation:

- [Architecture](docs/architecture.md) — Project structure, trait design, processing pipeline
- [CLI Reference](docs/cli-reference.md) — All commands, options, and examples
- [Provider behavior](docs/provider-behavior.md) — OpenAI vs Azure argument differences
- [Troubleshooting](docs/troubleshooting.md) — Common setup/runtime issues and fixes
- [Performance benchmarks](docs/performance-benchmarks.md) — Measurement plan, reference results, and templates
