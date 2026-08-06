# Architecture

## Overview

transcribeit uses a trait-based architecture with a pipeline-driven processing flow. Any audio or video file goes through: conversion → optional segmentation → transcription → output.

```
src/
├── main.rs                # Secret-safe environment loading and CLI entry point
├── command_dispatch.rs    # Setup, download, list, and run command routing
├── run_command.rs         # Batch run policy and pipeline configuration
├── provider_factory/      # Focused provider and S3 uploader construction
├── cli.rs                 # Clap command and validated option definitions
├── credentials.rs         # Provider-specific credential resolution
├── input.rs               # File, directory, and glob input resolution
├── batch.rs               # Batch output preflight and collision detection
├── transcriber.rs         # Transcriber trait, Segment, Transcript types
├── pipeline.rs            # Processing orchestration
├── pipeline_merge.rs      # Segmented transcript/timestamp/metadata merging
├── pipeline_output/       # Output coordination, manifest, capability, quality, cache
├── artifacts.rs           # Verified and atomic managed-artifact installation
├── setup.rs               # Model, VAD, diarization, and native-library setup
├── analysis/              # Provider-neutral post-transcription analysis
├── audio/
│   ├── extract.rs         # FFmpeg audio conversion
│   ├── segment.rs         # Silence detection and audio splitting
│   ├── vad.rs             # VAD-based speech segmentation (Silero VAD via sherpa-onnx)
│   └── wav.rs             # WAV reading and encoding (shared)
├── diarize/
│   ├── mod.rs             # Speaker diarization engine and speaker assignment
│   └── ffi.rs             # Raw C FFI bindings for sherpa-onnx speaker diarization
├── output/
│   ├── vtt.rs             # WebVTT subtitle writer (supports <v Speaker N> tags)
│   ├── srt.rs             # SRT subtitle writer (supports [Speaker N] labels)
│   └── manifest.rs        # JSON manifest writer (includes speaker labels)
├── storage/
│   └── s3.rs              # S3/R2 upload, presigning, and cleanup
└── engines/
    ├── whisper_local.rs   # Local whisper.cpp via whisper-rs
    ├── sherpa_onnx.rs     # Local Whisper, Qwen3-ASR, Moonshine, and SenseVoice
    ├── openai_api.rs      # OpenAI-compatible REST API
    ├── azure_openai.rs    # Azure OpenAI REST API
    ├── gemini.rs          # Gemini provider and submodules
    ├── nvidia_riva.rs     # NVIDIA hosted Riva gRPC ASR
    ├── deepgram.rs        # Deepgram Nova batch ASR + audio intelligence
    ├── qwen_filetrans.rs  # Qwen async file transcription provider
    ├── qwen_filetrans/    # Qwen request/response types and model limits
    ├── rate_limit.rs      # Typed retry policies and backoff
    └── model_cache.rs     # In-memory whisper model cache
```

## Core trait

All engines implement the async `Transcriber` trait, which provides three methods with a layered default implementation:

```rust
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Transcribe from decoded f32 PCM samples. All engines must implement this.
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript>;

    /// Transcribe from a file path. Default reads the file and delegates to transcribe_wav().
    /// API engines override this to upload the file directly (avoiding decode→re-encode).
    async fn transcribe_path(&self, wav_path: &Path) -> Result<Transcript>;

    /// Transcribe from in-memory WAV bytes. Default decodes to f32 and delegates to transcribe().
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript>;
}
```

- **Local engine** (`whisper_local`) uses `transcribe()` — it needs decoded samples for whisper.cpp.
- **Sherpa-ONNX engine** (`sherpa_onnx`) uses `transcribe()` — it needs decoded samples for the ONNX runtime.
- **OpenAI/Azure API engines** override `transcribe_path()` to upload files directly via multipart, and `transcribe_wav()` to upload in-memory bytes — avoiding the decode→re-encode round-trip.
- **Qwen file transcription** overrides `transcribe_path()` to upload prepared audio to S3-compatible storage, generate a pre-signed URL, and submit that URL to DashScope.
- **Gemini** overrides `transcribe_path()` to upload prepared audio through Gemini Files API and call streamed `streamGenerateContent` with structured JSON output. In signed URL mode, it stages the prepared MP3 in S3-compatible storage and sends the pre-signed URL as Gemini `file_uri` instead.
- **NVIDIA Riva** overrides `transcribe_path()` and `transcribe_wav()` to send WAV bytes to a hosted Riva gRPC endpoint with provider-native timestamps.
- **Deepgram** overrides `transcribe_path()` and `transcribe_wav()` to post WAV bytes to Deepgram's `/listen` endpoint with utterances, word timestamps, optional diarization, and optional audio intelligence flags. In URL mode, it stages the prepared WAV in S3-compatible storage and sends Deepgram a pre-signed URL JSON request instead.

## Processing pipeline

The `pipeline.rs` module orchestrates the full flow:

```
Input file (any format)
  │
  ├─ Create one canonical 16 kHz mono WAV when conversion or normalization is needed
  ├─ Duration, segmentation, VAD, and local diarization all use the canonical WAV
  ├─ Encode one provider upload artifact from that WAV when MP3 is required
  ├─ Hosted Riva and Deepgram use the canonical WAV for recognition
  │
  ├─ get_duration() via ffprobe
  │   └─ Local input protocols are restricted to `file,pipe`; nested network/data URLs are rejected
  │
  ├─ Should segment?
  │   ├─ --segment flag explicitly set
  │   ├─ Auto: OpenAI/Azure/NVIDIA Riva + actual prepared upload > 25 MiB
  │   └─ Auto: sherpa-onnx provider (always segments; max 30s per chunk)
  │
  ├─ If segmenting:
  │   ├─ VAD path (when --vad-model is set and sherpa-onnx feature is enabled):
  │   │   ├─ read_wav_bytes() → f32 PCM samples
  │   │   ├─ vad_segment(): detect speech → pad 250ms → merge gaps <200ms → split long chunks at low-energy points
  │   │   ├─ Extract chunk samples directly from memory
  │   │   └─ Transcribe each chunk, offset segment and word timestamps
  │   ├─ FFmpeg fallback (no VAD model, or sherpa-onnx feature disabled):
  │   │   ├─ detect_silence() via FFmpeg silencedetect filter
  │   │   ├─ compute_segments() at silence midpoints
  │   │   ├─ split_audio() into temp WAV files
  │   │   └─ Transcribe each segment, offset segment/word timestamps, preserve per-chunk metadata
  │
  ├─ If not segmenting:
  │   ├─ Local: read_wav() → transcribe() directly
  │   └─ API: transcribe_path() with prepared file or staged pre-signed URL
  │
  ├─ Speaker diarization? (when --diarize or --speakers N is set)
  │   ├─ read audio samples for diarization
  │   ├─ Diarizer.diarize() → speaker-labeled time spans
  │   └─ assign_speakers() overlays speaker labels onto transcript segments
  │
  ├─ Persist transcription output before optional analysis:
      ├─ Text to stdout or `<input_stem>.txt`
      ├─ VTT to file or stdout (with `<v Speaker N>` tags when diarized)
      ├─ SRT to file or stdout (with `[Speaker N]` labels when diarized)
      └─ JSON manifest to output directory (`transcribeit.manifest.v2`)
  │
  └─ Optional analysis:
      ├─ Success atomically updates the manifest with `analysis`
      └─ Failure preserves the transcript and records `analysis_error`
```

Temporary files use the `tempfile` crate and are cleaned up automatically on drop.

## Manifest contract

When `--output-dir` is set, the JSON manifest is the stable machine-readable contract for downstream applications. The current schema is `transcribeit.manifest.v2`.

- `transcript.text` and `transcript.segments` are the preferred consumer-facing transcript fields.
- Segment and word timestamps include canonical integer millisecond fields (`start_ms`, `end_ms`) plus second fields for readability.
- `capabilities` describes which optional fields are present, such as word timestamps, speaker labels, segment language, and emotion.
- `quality` describes how reliable timing/speaker metadata is, including `timing_source`, `timing_reliable`, and `timestamps_clamped`.
- `cache` describes normalized provider token-cache telemetry for transcription and optional analysis passes.
- `provider_metadata` is a stable envelope: `{ "provider": "...", "schema_version": "...", "data": { ... } }`.
- Provider-specific payloads live only under `provider_metadata.data`; temporary URLs and secrets must not be persisted.
- Segmented runs preserve exact provider metadata per chunk under `provider_metadata.data.chunks`, paired with each chunk index and absolute offset.
- Post-transcription analysis lives under the optional top-level `analysis` object. It is provider-neutral and separate from `provider_metadata` because downstream consumers should not need provider-specific parsing for summaries.
- `analysis_error` records an optional post-transcription analysis failure after the transcript and initial manifest have already been persisted.
- On Unix, transcript/subtitle/manifest files use owner-only permissions. Subtitle writers reject negative, zero-duration, reversed, and non-monotonic cue timing; `quality.timing_reliable` requires every segment to satisfy the timing invariants.
- The top-level `segments` array remains as a compatibility mirror for older consumers.

## Cache telemetry

Provider token-cache signals are normalized into `cache`:

- Gemini maps `usageMetadata.cachedContentTokenCount` and `usageMetadata.cacheTokensDetails`.
- OpenAI-compatible and Azure providers map `usage.prompt_tokens_details.cached_tokens` or `usage.input_tokens_details.cached_tokens` when a transcription endpoint returns `usage`.
- Qwen file transcription, NVIDIA Riva, local Whisper, and Sherpa-ONNX currently report `mode: "none"` because they do not expose token-cache telemetry through their transcription paths.

This is observability plus provider integration. The Gemini file cache reuses Files API uploads, and `--gemini-explicit-cache` creates/reuses Gemini `cachedContent` objects so provider token-cache hits can be deterministic when Gemini accepts the cache.

## Engines

### Local (`whisper_local.rs`)

Wraps [whisper-rs](https://github.com/tazz4843/whisper-rs), which binds to whisper.cpp. Inference runs on `tokio::task::spawn_blocking` to avoid blocking the async runtime.

Uses `ModelCache` to avoid reloading the same model across multiple transcription calls (important for segmented processing).

### OpenAI API (`openai_api.rs`)

Sends audio to any OpenAI-compatible `/v1/audio/transcriptions` endpoint via multipart upload with either WAV or MP3 input. The `base_url` is configurable, so it works with:

- OpenAI (`https://api.openai.com`)
- Self-hosted services (LocalAI, vLLM, etc.)
- Other compatible APIs

### Azure OpenAI (`azure_openai.rs`)

Same multipart upload pattern as OpenAI, but with Azure-specific URL format and `api-key` header authentication instead of Bearer token:

```
{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}
```

Caches whether the endpoint supports `verbose_json` via an `AtomicU8` flag to skip the fallback on subsequent segment calls within the same run.

### Qwen File Transcription (`qwen_filetrans.rs`)

Uses Alibaba DashScope `qwen3-asr-flash-filetrans` for whole-file asynchronous transcription. The provider:

- validates model selection before conversion/upload
- converts input audio/video to 16 kHz mono MP3
- uploads the prepared file to S3-compatible storage
- generates a pre-signed GET URL
- submits a DashScope async transcription task
- polls until completion
- downloads the result JSON
- maps sentence timestamps, word timestamps, language, and emotion into the normalized transcript/manifest model

The S3 staging implementation lives in `storage::s3` and works with AWS S3-compatible providers such as Cloudflare R2. Temporary pre-signed URLs are not persisted in manifests; only `file_url_present` is recorded.

Short synchronous Qwen models such as `qwen3-asr-flash` use a different API path and have strict 10 MB / 300 second limits. If one is selected with `-p qwen-filetrans`, the CLI fails before conversion or S3 upload.

### Gemini (`gemini.rs`)

Uses Gemini Files API and streamed `streamGenerateContent` for whole-file multimodal transcription. The provider:

- converts input audio/video to 16 kHz mono MP3
- uploads the prepared file with a resumable Files API upload
- waits for the file to become `ACTIVE`
- requests structured JSON with `text`, segment timestamps, speaker, language, and emotion fields
- joins streamed response text chunks and maps valid segments into the normalized transcript/manifest model
- falls back to generated transcript text when structured JSON is missing or invalid
- deletes the temporary Gemini file after the transcription request by default
- optionally reuses Gemini Files API uploads with `--gemini-file-cache`, using a local index keyed by SHA-256 of the exact prepared upload bytes
- optionally creates and reuses Gemini explicit `cachedContent` objects with `--gemini-explicit-cache`
- optionally bypasses Gemini Files API upload with `--gemini-use-presigned-url`, staging the prepared MP3 in S3/R2 and passing the signed URL as `file_uri`

Gemini is not a dedicated ASR endpoint. Timestamp, speaker, language, and emotion values come from the model's structured output, so benchmark quality before relying on them for subtitle workflows. The default path keeps Gemini whole-file for speaker continuity. Explicit segmentation is available, but failed whole-file requests are not automatically resubmitted as independent chunk jobs.

Gemini signed URL mode is for one-off prepared inputs up to 100 MB. It is rejected for Gemini 2.0 family models and cannot be combined with Gemini Files API cache or explicit cached content.

The Gemini cache index is protected by an inter-process lock and replaced atomically
from the same directory. A malformed index is preserved with a `.corrupt-*` suffix
and a new private index is created. HTTP result/error bodies and streamed SSE data
are bounded; SSE bytes are decoded as UTF-8 only after a complete event delimiter.

S3/R2 URL-staging resources are deleted after success or failure by default; `--keep-staged-resources` is the explicit debugging opt-out. Cleanup metadata records every attempt. Cleanup failure remains a warning after provider success and is attached to the provider error when both operations fail.

### NVIDIA Riva (`nvidia_riva.rs`)

Uses hosted NVIDIA Riva ASR over gRPC through generated protobuf bindings in `proto/riva/proto/`. The provider:

- connects to `grpc.nvcf.nvidia.com:443` by default
- sends `function-id` and Bearer authorization metadata
- submits `RecognizeRequest` with WAV bytes, language, sample rate, channel count, automatic punctuation, and word timestamp settings
- enables Riva diarization when `--diarize` is provided, using `--speakers N` as an optional maximum speaker hint
- maps Riva alternatives and word offsets into normalized segments and words, splitting at each contiguous speaker-tag change so mixed alternatives retain labels
- records request ids, audio info, feature flags, response counts, elapsed time, and confidence under `provider_metadata.data`

The provider is implemented entirely in Rust with `tonic`/`prost`. It does not download local NVIDIA NIM containers or require Python clients.

### Deepgram (`deepgram.rs`)

Uses Deepgram's pre-recorded `/listen` REST API for batch transcription. The provider:

- defaults to `nova-3`, with `nova-3-medical` available through `--remote-model` when enabled for the account
- requests `smart_format=true` and `utterances=true`
- enables provider-native diarization with `diarize_model=latest` when `--diarize` or `--speakers` is set
- can send either direct audio bytes or a staged pre-signed S3/R2 URL with `--deepgram-use-presigned-url`
- accepts Nova-3 keyterm prompts through `--deepgram-keyterm`
- can enable Deepgram audio intelligence through `--deepgram-intelligence` or individual flags for summary, topics, intents, entities, and sentiment
- maps Deepgram utterances and word timestamps into normalized segments and words
- preserves returned intelligence blocks under `provider_metadata.data.intelligence`
- clamps provider timestamps to `metadata.duration` when necessary and records that under `provider_metadata.data.response.timestamps_clamped`

Deepgram's intelligence JSON is intentionally kept as provider metadata because it is richer than the normalized transcript schema and because downstream Transcript Intelligence consumers may want to inspect provider-native topics, intents, sentiments, entities, and token usage. URL-mode metadata records only that a file URL was used; temporary pre-signed URLs are not persisted.

## Analysis (`analysis.rs`)

Post-transcription analysis is separate from transcription. The completed transcript,
selected text/subtitle output, and initial manifest are persisted first. The first
supported analysis is `--analysis summary`, which currently uses Gemini to run a
second structured JSON call over the transcript text. Success updates `analysis`;
failure returns an error while updating `analysis_error` and leaving transcription
artifacts intact:

- `analysis.summary.short`
- `analysis.summary.detailed`
- `analysis.summary.key_points`
- `analysis.summary.topics`
- `analysis.summary.action_items`
- `analysis.summary.questions`
- `analysis.summary.follow_ups`

The separation keeps transcript generation focused on ASR and allows future providers to implement the same `TranscriptAnalyzer` shape without changing transcript output formats.

### Sherpa-ONNX (`sherpa_onnx.rs`)

Local inference using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with automatic model architecture detection. Uses a **dedicated worker thread pattern**: the `OfflineRecognizer` is created on a plain `std::thread` (not on the Tokio runtime) and stays there for its entire lifetime. Transcription requests are sent to the thread via an `mpsc` channel and results come back through `tokio::sync::oneshot` channels. This design avoids:

- Blocking the async runtime during inference.
- Thread-safety issues with the C FFI recognizer, which is neither `Send` nor `Sync`.

Model initialization also happens on the worker thread, with errors propagated back through a sync channel so callers get a clear error if the model directory is invalid.

#### Auto-detected model architectures

The engine auto-detects the model architecture by inspecting the files present in the model directory:

| Architecture | Required files | Config used |
|---|---|---|
| **Whisper** | `encoder.onnx` + `decoder.onnx` | `OfflineWhisperModelConfig` |
| **Qwen3-ASR** | `conv_frontend.onnx` + `encoder.onnx` + `decoder.onnx` + `tokenizer/` | `OfflineQwen3ASRModelConfig` |
| **Moonshine** | `preprocess.onnx` + `encode.onnx` + `uncached_decode.onnx` + `cached_decode.onnx` | `OfflineMoonshineModelConfig` |
| **SenseVoice** | `model.onnx` (single file) | `OfflineSenseVoiceModelConfig` |

Whisper, Moonshine, and SenseVoice also require a `tokens.txt` (or `*-tokens.txt`) file. Qwen3-ASR instead requires its tokenizer directory. The engine prefers `int8` quantized ONNX files when available (e.g., `encoder.int8.onnx`) for lower memory usage, falling back to full-precision variants.

The model resolver supports glob-based directory matching, so you can use partial names like `-m moonshine-base` or `-m sense-voice` to find models in the cache directory.

**SenseVoice limitation:** SenseVoice models can detect emotions and audio events (laughter, applause, music), but these tags are stripped by the sherpa-onnx C API and are not available in the transcription output.

The pipeline automatically enables segmentation and caps `--max-segment-secs` at 30 when using this provider. Qwen3-ASR is capped at 20 seconds because the representative-corpus run failed on a 29.9-second chunk and completed at 20 seconds. Sherpa's Qwen result currently exposes text only, so the normalized transcript deliberately has no reliable native timing, language metadata, or speaker labels. The Qwen model must auto-detect language; a CLI language hint is rejected rather than silently ignored.

#### C++ stderr suppression

During `recognizer.decode()`, the sherpa-onnx C++ library prints warnings to stderr. The engine temporarily redirects stderr to `/dev/null` via `libc::dup`/`dup2` during decode calls and restores it immediately after, keeping the terminal output clean.

### Rate limiting (`rate_limit.rs`)

Shared retry logic classifies requests by replay safety. On 429 responses:
1. Parses `Retry-After` header
2. Falls back to parsing "retry after N seconds" from error body
3. Defaults to configurable base wait time
4. Retries up to configurable max attempts

Idempotent polling and reads may also retry transport and 5xx failures. Non-idempotent transcription/task-submission POSTs do not retry those ambiguous failures, avoiding duplicate billable work. All settings (timeout, retries, wait times) are configurable via CLI flags and env vars.

### Shared WAV encoding

OpenAI/Azure engines can send file uploads directly and choose the correct container format for compatibility (WAV for local transcribe path, MP3 for API provider uploads). Qwen file transcription stages MP3 in S3-compatible storage and sends DashScope a pre-signed URL. Gemini uploads MP3 through Gemini Files API by default, or stages MP3 in S3-compatible storage and sends a pre-signed URL when signed URL mode is enabled. NVIDIA Riva sends WAV bytes through gRPC. Deepgram posts WAV bytes to `/listen` by default, or stages WAV in S3-compatible storage and sends a pre-signed URL when URL mode is enabled. The `audio::wav::encode_wav()` helper is still used by local engines and non-file upload paths.

## Model cache (`model_cache.rs`)

`ModelCache` wraps a `Mutex<HashMap<String, Arc<WhisperContext>>>`. On first use, a model is loaded and stored; subsequent calls return a cloned `Arc`. This matters because:

- whisper.cpp model loading takes seconds
- Segmented processing calls `transcribe()` multiple times
- The cache is thread-safe via `Mutex` (lock contention is negligible since cache hits are fast)

Managed GGML, ONNX, VAD, diarization, and native-library downloads are pinned to exact upstream artifacts and verified by byte size and SHA-256. Archive extraction happens in a temporary staging directory before an atomic rename. Extracted directories carry a tree-digest marker and are reverified before reuse; explicit user-provided model paths remain trusted overrides.

## Build requirements

The `sherpa-onnx` Cargo feature is opt-in. It requires the sherpa-onnx shared libraries at both compile time and runtime. The `build.rs` script loads a `.env` file and reads `SHERPA_ONNX_LIB_DIR` to configure the linker search path and embed an `rpath` so the binary can find the dylibs at runtime.

Set `SHERPA_ONNX_LIB_DIR` in your `.env` file or environment before building:

```bash
# .env
SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
```

To build with sherpa-onnx enabled:

```bash
./scripts/bootstrap-sherpa.sh
cargo build --release --features sherpa-onnx
```

The setup `--output-dir` applies to native libraries as well as models and prints
absolute environment paths. CI has a dedicated all-feature job that bootstraps the
pinned Sherpa archive, runs FFmpeg-dependent tests explicitly, and executes tests
and strict Clippy with all features.

The default build omits the sherpa-onnx provider and eliminates the need for `SHERPA_ONNX_LIB_DIR`.

## VAD-based segmentation (`audio/vad.rs`)

When `--vad-model` is set and the `sherpa-onnx` feature is enabled, the pipeline uses Silero VAD (via sherpa-onnx) for speech-aware segmentation instead of FFmpeg's `silencedetect` filter. This avoids the main problem with silence-based splitting: mid-word cuts.

The VAD pipeline (`vad_segment()`) has four stages:

1. **Detect speech** -- Silero VAD processes 512-sample frames (~32ms at 16kHz) to find speech boundaries with sample-level precision.
2. **Pad 250ms** -- Each speech chunk is extended by 250ms on both sides to protect word boundaries at the edges.
3. **Merge gaps <200ms** -- Adjacent chunks separated by less than 200ms are merged to avoid splitting within short pauses.
4. **Split long chunks** -- Chunks exceeding `--max-segment-secs` are split at the lowest-energy point within a 1-second search window around the target cut point.

The VAD approach works directly on in-memory PCM samples, so there is no need for intermediate temp files during segmentation. Each chunk is transcribed via `engine.transcribe()` with sample slices, and timestamps are offset by the chunk start time.

When `--vad-model` is not set, segmentation falls back to FFmpeg `silencedetect` (the original behavior).

## Speaker diarization (`diarize/`)

Speaker diarization identifies which speaker is talking at each point in the audio. It requires the `sherpa-onnx` feature and two ONNX models:

- **Segmentation model** (`--diarize-segmentation-model`): a pyannote segmentation ONNX model that detects speaker change points.
- **Embedding model** (`--diarize-embedding-model`): a speaker embedding ONNX model that clusters voice characteristics.

The `Diarizer` follows the same dedicated worker thread pattern as `SherpaOnnxEngine`: the C FFI types are not `Send`/`Sync`, so they live on a plain `std::thread` and communicate via channels. Diarization requests are sent through `mpsc` and results come back through `tokio::sync::oneshot`.

After transcription completes, `assign_speakers()` overlays speaker labels onto transcript segments by finding the diarization segment with the maximum time overlap for each transcript segment. Speaker labels appear as:

- **VTT**: `<v Speaker 0>text</v>`
- **SRT**: `[Speaker 0] text`
- **Manifest JSON**: `"speaker": "Speaker 0"` field on each segment

## Adding a new engine

1. Create `src/engines/your_engine.rs`
2. Implement `Transcriber` for your struct
3. Add `pub mod your_engine;` to `src/engines/mod.rs`
4. Add a new `Provider` variant in `cli.rs`
5. Add a focused factory under `provider_factory/` and route it from `provider_factory/mod.rs`
6. Update provider behavior, CLI, architecture, and benchmark documentation

## Engineering issue registry

Tracked engineering findings and planned implementation work use stable
[`TI-NNN` pages](issues/README.md). Frontmatter is the source of truth for status,
priority, area, title, and resolution date. Every issue records explicit acceptance
criteria; resolved issues also preserve the implementation boundary and exact
validation evidence. Use `bun run scripts/issues_registry.ts generate` after page
metadata changes and `bun run scripts/issues_registry.ts check` before handoff.
