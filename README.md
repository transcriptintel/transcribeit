# transcribeit

A Rust CLI for speech-to-text transcription. Supports local inference via [whisper.cpp](https://github.com/ggerganov/whisper.cpp), remote transcription via OpenAI-compatible APIs, Azure OpenAI, Qwen ASR file transcription, Gemini multimodal transcription, NVIDIA hosted Riva ASR, and Deepgram.

Accepts any audio or video format — FFmpeg handles conversion automatically.
Sherpa-ONNX and its local ONNX/VAD/diarization stack were retired on 2026-08-06;
see the [retirement note](docs/retired/sherpa-onnx.md) for historical results and
migration guidance.

## Prerequisites

- Rust 1.96+ (edition 2024)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- C/C++ toolchain and CMake (for building whisper.cpp)
- S3-compatible storage credentials when using `qwen-filetrans`, Gemini signed-URL mode, or Deepgram signed-URL mode; Cloudflare R2 is supported through `S3_ENDPOINT_URL`
- NVIDIA API key and hosted Riva function id when using `nvidia-riva`
- Deepgram API key when using `deepgram`

## Quick start

```bash
# Build the default binary
cargo build --release

# Download a GGML model for --provider local
transcribeit download-model -s base

# List downloaded GGML models
transcribeit list-models

# Transcribe with local whisper.cpp (model alias resolves from MODEL_CACHE_DIR)
transcribeit run -i recording.mp3 -m base

# Or pass an explicit model path
transcribeit run -i recording.mp3 -m .cache/ggml-base.bin

# Process a directory (default output format is vtt)
transcribeit run -i samples/ -m base -o ./output

# Process a glob (brace expansion is not supported)
transcribeit run --input "samples/**/*.mp4" -p azure -o ./output

# Choose output format: text, vtt (default), or srt
transcribeit run -i meeting.mp4 -m base -f srt -o ./output

# Transcribe via OpenAI API
transcribeit run -p openai -i recording.mp3

# Use OpenAI's recommended completed-file model for text-only transcription
transcribeit run -p openai --remote-model gpt-transcribe \
  --language en -i recording.mp3 -f text -o ./output

# Use a separately managed local llama.cpp Qwen3-ASR server through the
# OpenAI-compatible endpoint (text only; no timestamps or provider metadata)
transcribeit run -p openai --api-key local \
  --base-url http://127.0.0.1:18080 \
  --remote-model ggml-org/Qwen3-ASR-0.6B-GGUF \
  -i recording.mp3 -f text -o ./output

# llama.cpp and its Qwen model/projector files remain operator-managed.
# TranscribeIt only sends the OpenAI-compatible transcription request.

# Transcribe via OpenAI hosted diarization
transcribeit run -p openai --diarize -i meeting.mp3 -f srt -o ./output

# Transcribe via Azure OpenAI
transcribeit run -p azure -i recording.mp3 \
  --azure-deployment my-whisper -b https://myresource.openai.azure.com

# Transcribe whole files with Qwen ASR via S3/R2 pre-signed URLs
transcribeit run -p qwen-filetrans -i recording.mp3 -f vtt -o ./output

# Transcribe whole files with Gemini Files API + streamed generateContent
transcribeit run -p gemini --remote-model gemini-3.6-flash \
  -i recording.mp3 -f vtt -o ./output

# Reuse Gemini Files API uploads for repeated runs within the 48h Files API window
transcribeit run -p gemini --gemini-file-cache \
  -i recording.mp3 -f vtt -o ./output

# Use Gemini explicit cachedContent for deterministic token-cache reuse
transcribeit run -p gemini --gemini-explicit-cache --gemini-cache-ttl-secs 3600 \
  -i recording.mp3 -f vtt -o ./output

# Use S3/R2 pre-signed URL input for a one-off Gemini run
transcribeit run -p gemini --gemini-use-presigned-url \
  -i recording.mp3 -f vtt -o ./output

# Keep a staged S3/R2 object for debugging (cleanup is otherwise the default)
transcribeit run -p qwen-filetrans --keep-staged-resources \
  -i recording.mp3 -f vtt -o ./output

# Transcribe with Gemini and add a structured summary to the manifest
transcribeit run -p gemini --analysis summary \
  -i interview.mp4 -f vtt -o ./output

# Transcribe with NVIDIA hosted Riva ASR over gRPC
transcribeit run -p nvidia-riva -i recording.wav \
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

```

## Features

- **Any local input format** — MP3, MP4, WAV, FLAC, OGG, etc. FFmpeg converts to mono 16kHz WAV automatically. Nested network/data protocols are blocked for local inputs.
- **7 providers** — Local whisper.cpp, OpenAI API, Azure OpenAI, Qwen file transcription, Gemini, NVIDIA Riva, and Deepgram. Extensible via the `Transcriber` trait.
- **Qwen ASR whole-file transcription** — `qwen-filetrans` stages audio in S3-compatible storage, passes a pre-signed URL to DashScope, polls the async task, and maps Qwen timestamps into the transcript model.
- **External llama.cpp compatibility** — A user-managed Qwen3-ASR `llama-server` can be reached through `-p openai --base-url`; TranscribeIt does not install, launch, monitor, update, stop, or clean that server or its GGUF artifacts, and the tested endpoint returns text without timestamps or speakers.
- **Resumable benchmark harness** — Bun-native YAML matrices select pinned corpus fixtures, declare local/hosted and warm/cold state, persist attempts atomically, preserve failures, remove transcript-bearing outputs by default, and require an explicit opt-in for hosted execution.
- **Stable manifest schema** — Manifests use `transcribeit.manifest.v2` with canonical millisecond timestamps, provider-neutral capabilities/quality fields, and provider-specific metadata under `provider_metadata.data`.
- **Cache telemetry** — Manifests normalize provider token-cache signals under `cache`, including Gemini `cachedContentTokenCount` and OpenAI/Azure-style `cached_tokens` when returned.
- **Qwen provider metadata** — Manifests include Qwen task timing/usage, audio info, per-segment language/emotion, and word-level timestamps. Temporary pre-signed URLs are not persisted.
- **Qwen model guardrails** — Accidental short-audio `qwen3-asr-flash` model selection is rejected before conversion and S3 upload; use `qwen3-asr-flash-filetrans` for this provider.
- **Gemini whole-file transcription** — `gemini` uploads prepared audio through Gemini Files API, streams `generateContent` response chunks with structured JSON output, and maps segment timestamps, speaker labels, language, and emotion when returned.
- **Gemini file reuse** — `--gemini-file-cache` keeps a locked, atomically replaced local index keyed by SHA-256 of the prepared 16 kHz mono MP3 bytes. Corrupt indexes are quarantined and rebuilt before remote reuse.
- **Gemini signed URL input** — `--gemini-use-presigned-url` stages prepared MP3 audio in S3/R2 and sends the signed URL as Gemini `file_uri` for one-off inputs up to 100 MB. Files API cache and explicit cached content remain Files API-only.
- **Gemini explicit cache** — `--gemini-explicit-cache` creates and reuses Gemini `cachedContent` objects with a configurable TTL, producing deterministic `cachedContentTokenCount` telemetry when Gemini accepts the cache.
- **Gemini summary analysis** — `--analysis summary` runs only after transcript/output persistence. Success updates the manifest; failure remains a command error and is recorded separately under `analysis_error` without losing the transcript.
- **Temporary resource cleanup** — S3/R2 staged objects for Qwen, Gemini signed URL mode, and Deepgram signed URL mode are deleted after every provider outcome by default. `--keep-staged-resources` is the explicit debugging opt-out.
- **NVIDIA hosted Riva ASR** — `nvidia-riva` calls hosted NVIDIA Riva gRPC endpoints with provider-native word timestamps and splits mixed alternatives at contiguous speaker changes so server diarization labels are retained.
- **Deepgram Nova batch ASR** — `deepgram` calls Deepgram's `/listen` API, defaults to `nova-3`, requests utterances and smart formatting, supports provider-native diarization through `--diarize`, and can submit either direct audio bytes or an S3/R2 pre-signed URL with `--deepgram-use-presigned-url`.
- **Deepgram audio intelligence** — `--deepgram-intelligence` captures Deepgram summary, topics, intents, entity detection, and sentiment in `provider_metadata.data.intelligence`; `--deepgram-keyterm` passes Nova-3 keyterm prompts for domain terminology.
- **Model aliases** — `-m base`, `-m tiny`, etc. resolve whisper.cpp GGML files from `MODEL_CACHE_DIR` for the local provider.
- **Language hinting** — Pass `--language` to force local and API transcription language.
- **FFmpeg audio normalization** — Optional `--normalize` to apply loudnorm before transcription.
- **Silence-based segmentation** — FFmpeg `silencedetect` provides bounded segmentation for local and hosted providers.
- **Provider-specific splitting** — OpenAI, Azure, and NVIDIA Riva auto-split only when the actual prepared upload exceeds 25 MiB. Qwen FileTrans, Gemini, and Deepgram remain whole-file unless `--segment` is explicit.
- **Verified model installation** — Managed whisper.cpp GGML files are pinned by revision, size, and SHA-256.
- **Progress spinner** — Shows live terminal feedback during transcription (single file and segmented mode).
- **Parallel API segment transcription** — Multiple segment requests can be processed concurrently with `--segment-concurrency`.
- **VTT output** (default) — WebVTT subtitle files with validated monotonic, positive-duration timestamps.
- **SRT output** — SubRip subtitle files with the same timing validation.
- **Text output** — Writes plain text transcript to stdout by default and `<input>.txt` when `--output-dir` is specified.
- **Private outputs** — Transcript, subtitle, cache-index, and manifest files are created with owner-only permissions on Unix; manifests are atomically replaced.
- **Bounded responses** — Hosted HTTP results/errors and Gemini SSE events have hard size limits; SSE UTF-8 is decoded only after complete event framing.
- **JSON manifest** — Processing metadata, segment details, statistics, and all-segment timing reliability.
- **Model caching** — Loaded whisper models are cached in memory for batch processing.
- **Model management** — Download and list verified whisper.cpp GGML models.

## Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
MODEL_CACHE_DIR=.cache
OPENAI_API_KEY=sk-your_key_here
GEMINI_API_KEY=your_gemini_key_here
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_USE_PRESIGNED_URL=false
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
TRANSCRIBEIT_KEEP_STAGED_RESOURCES=false
TRANSCRIBEIT_MAX_RETRIES=5
TRANSCRIBEIT_REQUEST_TIMEOUT_SECS=120
TRANSCRIBEIT_RETRY_WAIT_BASE_SECS=10
TRANSCRIBEIT_RETRY_WAIT_MAX_SECS=120
```

Keep `.env` private because it contains provider and storage credentials:

```bash
chmod 600 .env
```

Provider-specific environment keys never fall back to `OPENAI_API_KEY`. The generic
`--api-key` option remains an explicit compatibility override, but environment
variables or a private `--api-key-file` are preferred because command-line values
can be exposed through process listings and shell history. Batch inputs require
`--output-dir` and fail before transcription when two inputs would produce the same
output stem.

## Binary distribution

Pre-built binaries can be deployed without Rust or build tools. Every binary
needs FFmpeg on PATH. Use `transcribeit setup` to install the default verified
GGML model; `--output-dir` controls the model directory and the setup summary
prints its absolute reusable path.

To build a distributable binary:

```bash
cargo build --release
mkdir -p dist
cp target/release/transcribeit dist/
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
- [Release process](docs/releasing.md) — Versioning, validation, tagging, release archives, and verification
- [Retired Sherpa-ONNX integration](docs/retired/sherpa-onnx.md) — Historical scope, evidence, and migration guidance
- [Representative corpus](benchmarks/corpus/README.md) — Versioned fixtures, rights, materialization, and scoring contract
- [Latest representative provider record](benchmarks/results/2026-08-06-ti-007-representative-corpus.sanitized.json) — TI-007 metrics, failures, and cleanup evidence without transcript content
- [Engineering issues](docs/issues/README.md) — Tracked `TI-NNN` findings, priorities, and acceptance criteria
