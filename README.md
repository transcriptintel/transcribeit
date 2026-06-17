# transcribeit

A Rust CLI for speech-to-text transcription. Supports local inference via [whisper.cpp](https://github.com/ggerganov/whisper.cpp), local inference via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), remote transcription via OpenAI-compatible APIs, Azure OpenAI, Qwen ASR file transcription, Gemini multimodal transcription, NVIDIA hosted Riva ASR, and Deepgram.

Accepts any audio or video format — FFmpeg handles conversion automatically.

## Prerequisites

- Rust 1.96+ (edition 2024)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- C/C++ toolchain and CMake (for building whisper.cpp)
- sherpa-onnx shared libraries (if using the `sherpa-onnx` provider) — set `SHERPA_ONNX_LIB_DIR` in `.env` to the directory containing them
- S3-compatible storage credentials when using `qwen-filetrans` or Deepgram pre-signed URL mode; Cloudflare R2 is supported through `S3_ENDPOINT_URL`
- NVIDIA API key and hosted Riva function id when using `nvidia-riva`
- Deepgram API key when using `deepgram`

## Quick start

```bash
# Build the default binary
cargo build --release

# Build with sherpa-onnx (reads SHERPA_ONNX_LIB_DIR from .env automatically via build.rs)
cargo build --release --features sherpa-onnx

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

# Transcribe via OpenAI hosted diarization
transcribeit run -p openai --diarize -i meeting.mp3 -f srt -o ./output

# Transcribe via Azure OpenAI
transcribeit run -p azure -i recording.mp3 \
  --azure-deployment my-whisper -b https://myresource.openai.azure.com

# Transcribe whole files with Qwen ASR via S3/R2 pre-signed URLs
transcribeit run -p qwen-filetrans -i recording.mp3 -f vtt -o ./output

# Transcribe whole files with Gemini Files API + streamed generateContent
transcribeit run -p gemini --remote-model gemini-3.5-flash \
  -i recording.mp3 -f vtt -o ./output

# Reuse Gemini Files API uploads for repeated runs within the 48h Files API window
transcribeit run -p gemini --gemini-file-cache \
  -i recording.mp3 -f vtt -o ./output

# Use Gemini explicit cachedContent for deterministic token-cache reuse
transcribeit run -p gemini --gemini-explicit-cache --gemini-cache-ttl-secs 3600 \
  -i recording.mp3 -f vtt -o ./output

# Transcribe with Gemini and add a structured summary to the manifest
transcribeit run -p gemini --analysis summary \
  -i interview.mp4 -f vtt -o ./output

# Transcribe with NVIDIA hosted Riva ASR over gRPC
transcribeit run -p nvidia-riva -i recording.wav \
  --nvidia-api-key "$NVIDIA_API_KEY" \
  --nvidia-riva-function-id "$NVIDIA_RIVA_FUNCTION_ID" \
  -f vtt -o ./output

# Transcribe with Deepgram Nova-3 batch ASR and provider-native diarization
transcribeit run -p deepgram --remote-model nova-3 --diarize \
  -i recording.wav -f vtt -o ./output

# Transcribe with Deepgram by staging the prepared audio in S3/R2 first
transcribeit run -p deepgram --remote-model nova-3 --deepgram-use-presigned-url \
  -i recording.wav -f vtt -o ./output

# Transcribe with Deepgram Nova-3 Medical, intelligence metadata, and domain keyterms
transcribeit run -p deepgram --remote-model nova-3-medical \
  --diarize --deepgram-intelligence \
  --deepgram-keyterm Ofev --deepgram-keyterm Esbriet --deepgram-keyterm IPF \
  -i interview.wav -f vtt -o ./output

# Force language and normalize before transcription
transcribeit run -i recording.wav -m base --language en --normalize

# VAD-based segmentation (speech-aware, avoids mid-word cuts)
transcribeit run -p sherpa-onnx -m base -i recording.mp3 --vad-model .cache/silero_vad.onnx

# Speaker diarization (local Sherpa post-processing, fixed speaker count required)
transcribeit run -i interview.mp3 -m base --diarize --speakers 2 \
  --diarize-segmentation-model .cache/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --diarize-embedding-model .cache/wespeaker_en_voxceleb_CAM++.onnx
```

## Features

- **Any input format** — MP3, MP4, WAV, FLAC, OGG, etc. FFmpeg converts to mono 16kHz WAV automatically.
- **8 providers** — Local whisper.cpp, sherpa-onnx, OpenAI API, Azure OpenAI, Qwen file transcription, Gemini, NVIDIA Riva, and Deepgram. Extensible via the `Transcriber` trait.
- **Qwen ASR whole-file transcription** — `qwen-filetrans` stages audio in S3-compatible storage, passes a pre-signed URL to DashScope, polls the async task, and maps Qwen timestamps into the transcript model.
- **Stable manifest schema** — Manifests use `transcribeit.manifest.v2` with canonical millisecond timestamps, provider-neutral capabilities/quality fields, and provider-specific metadata under `provider_metadata.data`.
- **Cache telemetry** — Manifests normalize provider token-cache signals under `cache`, including Gemini `cachedContentTokenCount` and OpenAI/Azure-style `cached_tokens` when returned.
- **Qwen provider metadata** — Manifests include Qwen task timing/usage, audio info, per-segment language/emotion, and word-level timestamps. Temporary pre-signed URLs are not persisted.
- **Qwen model guardrails** — Accidental short-audio `qwen3-asr-flash` model selection is rejected before conversion and S3 upload; use `qwen3-asr-flash-filetrans` for this provider.
- **Gemini whole-file transcription** — `gemini` uploads prepared audio through Gemini Files API, streams `generateContent` response chunks with structured JSON output, and maps segment timestamps, speaker labels, language, and emotion when returned.
- **Gemini file reuse** — `--gemini-file-cache` keeps a local index of Gemini Files API uploads keyed by SHA-256 of the prepared 16 kHz mono MP3 bytes, verifies the remote file before reuse, and records reuse metadata in the manifest.
- **Gemini explicit cache** — `--gemini-explicit-cache` creates and reuses Gemini `cachedContent` objects with a configurable TTL, producing deterministic `cachedContentTokenCount` telemetry when Gemini accepts the cache.
- **Gemini summary analysis** — `--analysis summary` runs a second Gemini JSON pass over the transcript and stores a provider-neutral summary, key points, topics, questions, and follow-ups in the manifest.
- **NVIDIA hosted Riva ASR** — `nvidia-riva` calls hosted NVIDIA Riva gRPC endpoints with provider-native word timestamps, optional server-side diarization, and manifest metadata.
- **Deepgram Nova batch ASR** — `deepgram` calls Deepgram's `/listen` API, defaults to `nova-3`, requests utterances and smart formatting, supports provider-native diarization through `--diarize`, and can submit either direct audio bytes or an S3/R2 pre-signed URL with `--deepgram-use-presigned-url`.
- **Deepgram audio intelligence** — `--deepgram-intelligence` captures Deepgram summary, topics, intents, entity detection, and sentiment in `provider_metadata.data.intelligence`; `--deepgram-keyterm` passes Nova-3 keyterm prompts for domain terminology.
- **3 model architectures via sherpa-onnx** — Whisper, Moonshine, and SenseVoice are auto-detected from the model directory contents. Just point `--model` at any supported model directory.
- **Model aliases** — `-m base`, `-m tiny`, etc. resolve from `MODEL_CACHE_DIR` for both `local` and `sherpa-onnx` providers. The sherpa-onnx resolver also supports glob matching (e.g., `-m moonshine-base`, `-m sense-voice`).
- **Language hinting** — Pass `--language` to force local and API transcription language.
- **FFmpeg audio normalization** — Optional `--normalize` to apply loudnorm before transcription.
- **VAD-based segmentation** — Speech-aware segmentation via Silero VAD (sherpa-onnx). Detects speech boundaries with padding and gap merging to avoid mid-word cuts. Use `--vad-model .cache/silero_vad.onnx`.
- **Silence-based segmentation** — Fallback segmentation via FFmpeg `silencedetect` for API providers or when VAD model is not available.
- **sherpa-onnx auto-segmentation** — Whisper ONNX models only support ≤30s per call; segmentation is enabled automatically.
- **sherpa-onnx is optional** — Enable it explicitly with `cargo build --features sherpa-onnx` when you need ONNX providers or Sherpa-backed diarization.
- **Auto-split for API limits** — Files exceeding the conservative remote-provider threshold are automatically segmented when using OpenAI, Azure, Qwen file transcription, NVIDIA Riva, or sherpa-onnx. Gemini stays whole-file by default to preserve model-level speaker continuity and falls back to segmented mode only after long whole-file failures.
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
GEMINI_API_KEY=your_gemini_key_here
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta
NVIDIA_API_KEY=your_nvidia_key_here
NVIDIA_RIVA_FUNCTION_ID=your_hosted_riva_function_id
NVIDIA_RIVA_SERVER=grpc.nvcf.nvidia.com:443
DEEPGRAM_API_KEY=your_deepgram_key_here
DEEPGRAM_API_BASE_URL=https://api.deepgram.com/v1
DEEPGRAM_INTELLIGENCE=false
DEEPGRAM_KEYTERM=Ofev,Esbriet,IPF
DEEPGRAM_USE_PRESIGNED_URL=false
AZURE_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
AZURE_DEPLOYMENT_NAME=whisper
AZURE_API_VERSION=2024-06-01
DASHSCOPE_API_KEY=sk-your_dashscope_key_here
DASHSCOPE_ASR_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
S3_BUCKET=your-staging-bucket
S3_REGION=auto
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
S3_ACCESS_KEY_ID=your_s3_access_key
S3_SECRET_ACCESS_KEY=your_s3_secret_key
# Optional; when unset, URL-staging providers choose their own prefix.
S3_PREFIX=transcribeit/qwen-filetrans
S3_PRESIGN_EXPIRES_SECS=3600
S3_FORCE_PATH_STYLE=false
TRANSCRIBEIT_MAX_RETRIES=5
TRANSCRIBEIT_REQUEST_TIMEOUT_SECS=120
TRANSCRIBEIT_RETRY_WAIT_BASE_SECS=10
TRANSCRIBEIT_RETRY_WAIT_MAX_SECS=120
VAD_MODEL=.cache/silero_vad.onnx
DIARIZE_SEGMENTATION_MODEL=.cache/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
DIARIZE_EMBEDDING_MODEL=.cache/wespeaker_en_voxceleb_CAM++.onnx
```

## Binary distribution

Pre-built binaries can be deployed without Rust or build tools. The binary needs FFmpeg on PATH and the sherpa-onnx shared libraries alongside it:

```
transcribeit              # binary
lib/                      # sherpa-onnx shared libraries
  libsherpa-onnx-c-api.dylib
  libonnxruntime.dylib
```

On first run, use `transcribeit setup` to download models and additional components. The binary looks for shared libraries in `lib/` relative to itself — no environment variables needed at runtime.

To build a distributable binary:

```bash
cargo build --release --features sherpa-onnx
# Copy binary + libs
cp target/release/transcribeit dist/
cp vendor/sherpa-onnx-*/lib/lib*.dylib dist/lib/
```

To build without sherpa-onnx (no shared library dependency):

```bash
cargo build --release
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
- [Provider behavior](docs/provider-behavior.md) — Provider-specific API shape, upload behavior, and authentication
- [Troubleshooting](docs/troubleshooting.md) — Common setup/runtime issues and fixes
- [Performance benchmarks](docs/performance-benchmarks.md) — Measurement plan, reference results, and templates
