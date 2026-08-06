# transcribeit

A Rust CLI for speech-to-text transcription. Supports local inference via [whisper.cpp](https://github.com/ggerganov/whisper.cpp), local inference via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), remote transcription via OpenAI-compatible APIs, Azure OpenAI, Qwen ASR file transcription, Gemini multimodal transcription, NVIDIA hosted Riva ASR, and Deepgram.

Accepts any audio or video format — FFmpeg handles conversion automatically.

## Prerequisites

- Rust 1.96+ (edition 2024)
- [FFmpeg](https://ffmpeg.org/) installed and on PATH
- C/C++ toolchain and CMake (for building whisper.cpp)
- sherpa-onnx shared libraries (if using the `sherpa-onnx` provider) — set `SHERPA_ONNX_LIB_DIR` in `.env` to the directory containing them
- S3-compatible storage credentials when using `qwen-filetrans`, Gemini signed-URL mode, or Deepgram signed-URL mode; Cloudflare R2 is supported through `S3_ENDPOINT_URL`
- NVIDIA API key and hosted Riva function id when using `nvidia-riva`
- Deepgram API key when using `deepgram`

## Quick start

```bash
# Build the default binary
cargo build --release

# Bootstrap verified native libraries, then build with sherpa-onnx
./scripts/bootstrap-sherpa.sh
# Copy the printed SHERPA_ONNX_LIB_DIR into .env or export it in this shell.
cargo build --release --features sherpa-onnx

# Download a GGML model (default format, for --provider local)
transcribeit download-model -s base

# Download an ONNX model (for --provider sherpa-onnx)
transcribeit download-model -s base -f onnx

# Install the verified Qwen3-ASR 0.6B int8 ONNX model
transcribeit setup --component qwen3-asr

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

# Transcribe locally with Qwen3-ASR (auto-segments at <=20s boundaries)
transcribeit run -p sherpa-onnx -i recording.mp3 -m qwen3-asr -f text -o ./output

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
  --remote-model ggml-org/Qwen3-ASR-1.7B-GGUF \
  -i recording.mp3 -f text -o ./output

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

# VAD-based segmentation (speech-aware, avoids mid-word cuts)
transcribeit run -p sherpa-onnx -m base -i recording.mp3 --vad-model .cache/silero_vad.onnx

# Speaker diarization (local Sherpa post-processing, fixed speaker count required)
transcribeit run -i interview.mp3 -m base --diarize --speakers 2 \
  --diarize-segmentation-model .cache/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --diarize-embedding-model .cache/wespeaker_en_voxceleb_CAM++.onnx
```

## Features

- **Any local input format** — MP3, MP4, WAV, FLAC, OGG, etc. FFmpeg converts to mono 16kHz WAV automatically. Nested network/data protocols are blocked for local inputs.
- **8 providers** — Local whisper.cpp, sherpa-onnx, OpenAI API, Azure OpenAI, Qwen file transcription, Gemini, NVIDIA Riva, and Deepgram. Extensible via the `Transcriber` trait.
- **Qwen ASR whole-file transcription** — `qwen-filetrans` stages audio in S3-compatible storage, passes a pre-signed URL to DashScope, polls the async task, and maps Qwen timestamps into the transcript model.
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
- **4 model architectures via sherpa-onnx** — Whisper, Qwen3-ASR, Moonshine, and SenseVoice are auto-detected from the model directory contents. Just point `--model` at any supported model directory.
- **Model aliases** — `-m base`, `-m tiny`, etc. resolve from `MODEL_CACHE_DIR` for both `local` and `sherpa-onnx` providers. `-m qwen3-asr` resolves the managed 0.6B int8 model, and glob matching supports partial names such as `moonshine-base` and `sense-voice`.
- **Language hinting** — Pass `--language` to force local and API transcription language.
- **FFmpeg audio normalization** — Optional `--normalize` to apply loudnorm before transcription.
- **VAD-based segmentation** — Speech-aware segmentation via Silero VAD (sherpa-onnx). Detects speech boundaries with padding and gap merging to avoid mid-word cuts. Use `--vad-model .cache/silero_vad.onnx`.
- **Silence-based segmentation** — Fallback segmentation via FFmpeg `silencedetect` for API providers or when VAD model is not available.
- **sherpa-onnx auto-segmentation** — Segmentation is enabled automatically and capped at 30 seconds for existing architectures or 20 seconds for Qwen3-ASR, based on the long-form evaluation.
- **sherpa-onnx is optional** — Enable it explicitly with `cargo build --features sherpa-onnx` when you need ONNX providers or Sherpa-backed diarization.
- **Provider-specific splitting** — OpenAI, Azure, and NVIDIA Riva auto-split only when the actual prepared upload exceeds 25 MiB. Qwen FileTrans, Gemini, and Deepgram remain whole-file unless `--segment` is explicit. Sherpa-ONNX always segments, with a model-safe 30-second maximum or 20 seconds for Qwen3-ASR.
- **Verified model installation** — Managed models and Sherpa native libraries are pinned by revision, size, and SHA-256. Archives are verified before atomic extraction, and installed directories carry a tree-integrity marker.
- **Progress spinner** — Shows live terminal feedback during transcription (single file and segmented mode).
- **Parallel API segment transcription** — Multiple segment requests can be processed concurrently with `--segment-concurrency`.
- **VTT output** (default) — WebVTT subtitle files with validated monotonic, positive-duration timestamps.
- **SRT output** — SubRip subtitle files with the same timing validation.
- **Text output** — Writes plain text transcript to stdout by default and `<input>.txt` when `--output-dir` is specified.
- **Private outputs** — Transcript, subtitle, cache-index, and manifest files are created with owner-only permissions on Unix; manifests are atomically replaced.
- **Bounded responses** — Hosted HTTP results/errors and Gemini SSE events have hard size limits; SSE UTF-8 is decoded only after complete event framing.
- **JSON manifest** — Processing metadata, segment details, statistics, and all-segment timing reliability.
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
VAD_MODEL=.cache/silero_vad.onnx
DIARIZE_SEGMENTATION_MODEL=.cache/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
DIARIZE_EMBEDDING_MODEL=.cache/wespeaker_en_voxceleb_CAM++.onnx
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

Pre-built binaries can be deployed without Rust or build tools. Every binary needs
FFmpeg on PATH. A binary built with `--features sherpa-onnx` also needs the
sherpa-onnx shared libraries alongside it:

```
transcribeit              # binary
lib/                      # sherpa-onnx shared libraries
  libsherpa-onnx-c-api.dylib
  libonnxruntime.dylib
```

Use `transcribeit setup` to download models and additional components. The
`--output-dir` option applies to Sherpa libraries as well as models and the setup
summary prints absolute, reusable environment paths. For a source checkout,
`./scripts/bootstrap-sherpa.sh [INSTALL_ROOT]` installs the verified native archive
before the first all-feature build and prints `SHERPA_ONNX_LIB_DIR`.
A Sherpa-enabled binary looks for shared libraries in `lib/` relative to itself—no
environment variables are needed at runtime when that layout is used.

To build a distributable binary:

```bash
cargo build --release --features sherpa-onnx
# Copy binary + libs
mkdir -p dist/lib
cp target/release/transcribeit dist/
cp -R "$SHERPA_ONNX_LIB_DIR"/. dist/lib/
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
- [Representative corpus](benchmarks/corpus/README.md) — Versioned fixtures, rights, materialization, and scoring contract
- [Latest representative provider record](benchmarks/results/2026-08-06-ti-007-representative-corpus.sanitized.json) — TI-007 metrics, failures, and cleanup evidence without transcript content
- [Engineering issues](docs/issues/README.md) — Tracked `TI-NNN` findings, priorities, and acceptance criteria
