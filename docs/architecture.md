# Architecture

## Overview

transcribeit uses a trait-based architecture with a pipeline-driven processing flow. Any audio or video file goes through: conversion → optional segmentation → transcription → output.

```
src/
├── main.rs                # CLI parsing, engine construction, input resolution
├── transcriber.rs         # Transcriber trait, Segment, Transcript types
├── pipeline.rs            # Processing orchestration
├── audio/
│   ├── extract.rs         # FFmpeg audio conversion
│   ├── segment.rs         # Silence detection and audio splitting
│   └── wav.rs             # WAV reading and encoding (shared)
├── output/
│   ├── vtt.rs             # WebVTT subtitle writer
│   ├── srt.rs             # SRT subtitle writer
│   └── manifest.rs        # JSON manifest writer
└── engines/
    ├── whisper_local.rs   # Local whisper.cpp via whisper-rs
    ├── openai_api.rs      # OpenAI-compatible REST API
    ├── azure_openai.rs    # Azure OpenAI REST API
    ├── rate_limit.rs      # Retry logic and 429 handling
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

- **Local engine** uses `transcribe()` — it needs decoded samples for whisper.cpp.
- **API engines** override `transcribe_path()` to upload files directly via multipart, and `transcribe_wav()` to upload in-memory bytes — avoiding the decode→re-encode round-trip.

## Processing pipeline

The `pipeline.rs` module orchestrates the full flow:

```
Input file (any format)
  │
  ├─ needs_conversion()? ──→ extract_to_wav(normalize) for local provider
  ├─ upload_as_mp3(normalize) for API provider (16kHz mono MP3)
  │
  ├─ get_duration() via ffprobe
  │
  ├─ Should segment?
  │   ├─ --segment flag explicitly set
  │   └─ Auto: remote provider + estimated size > 25MB
  │
  ├─ If segmenting:
  │   ├─ detect_silence() via FFmpeg silencedetect filter
  │   ├─ compute_segments() at silence midpoints
  │   ├─ split_audio() into temp WAV files
  │   └─ Transcribe each segment, offset timestamps (concurrently for API providers)
  │
  ├─ If not segmenting:
  │   ├─ Local: read_wav() → transcribe() directly
  │   └─ API: transcribe_path() with prepared file
  │
  ├─ normalize_audio? ──→ optional loudnorm filter in ffmpeg conversion pipeline
  └─ Output:
      ├─ Text to stdout or `<input_stem>.txt`
      ├─ VTT to file or stdout
      ├─ SRT to file or stdout
      └─ JSON manifest to output directory
```

Temporary files use the `tempfile` crate and are cleaned up automatically on drop.

## Engines

### Local (`whisper_local.rs`)

Wraps [whisper-rs](https://github.com/tazz4843/whisper-rs), which binds to whisper.cpp. Inference runs on `tokio::task::spawn_blocking` to avoid blocking the async runtime.

Uses `ModelCache` to avoid reloading the same model across multiple transcription calls (important for segmented processing).

### OpenAI API (`openai_api.rs`)

Sends audio to any OpenAI-compatible `/v1/audio/transcriptions` endpoint via multipart upload with either WAV or MP3 input. The `base_url` is configurable, so it works with:

- OpenAI (`https://api.openai.com`)
- Self-hosted services (LocalAI, vLLM, etc.)
- Other compatible APIs (Qwen AST, etc.)

### Azure OpenAI (`azure_openai.rs`)

Same multipart upload pattern as OpenAI, but with Azure-specific URL format and `api-key` header authentication instead of Bearer token:

```
{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}
```

Caches whether the endpoint supports `verbose_json` via an `AtomicU8` flag to skip the fallback on subsequent segment calls within the same run.

### Rate limiting (`rate_limit.rs`)

Shared retry logic for both API engines. On 429 responses:
1. Parses `Retry-After` header
2. Falls back to parsing "retry after N seconds" from error body
3. Defaults to configurable base wait time
4. Retries up to configurable max attempts

All settings (timeout, retries, wait times) are configurable via CLI flags and env vars.

### Shared WAV encoding

Both API engines can send file uploads directly and choose the correct container format for compatibility (WAV for local transcribe path, MP3 for API provider uploads). The `audio::wav::encode_wav()` helper is still used by local engines and non-file upload paths.

## Model cache (`model_cache.rs`)

`ModelCache` wraps a `Mutex<HashMap<String, Arc<WhisperContext>>>`. On first use, a model is loaded and stored; subsequent calls return a cloned `Arc`. This matters because:

- whisper.cpp model loading takes seconds
- Segmented processing calls `transcribe()` multiple times
- The cache is thread-safe via `Mutex` (lock contention is negligible since cache hits are fast)

## Adding a new engine

1. Create `src/engines/your_engine.rs`
2. Implement `Transcriber` for your struct
3. Add `pub mod your_engine;` to `src/engines/mod.rs`
4. Add a new `Provider` variant in `main.rs`
5. Add a match arm in the `Command::Run` handler to construct your engine
