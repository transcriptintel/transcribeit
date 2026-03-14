# Architecture

## Overview

transcribeit uses a trait-based architecture with a pipeline-driven processing flow. Any audio or video file goes through: conversion → optional segmentation → transcription → output.

```
src/
├── main.rs                # CLI parsing, engine construction
├── transcriber.rs         # Transcriber trait, Segment, Transcript types
├── pipeline.rs            # Processing orchestration
├── audio/
│   ├── extract.rs         # FFmpeg audio conversion
│   ├── segment.rs         # Silence detection and audio splitting
│   └── wav.rs             # WAV reading and encoding (shared)
├── output/
│   ├── vtt.rs             # WebVTT subtitle writer
│   └── manifest.rs        # JSON manifest writer
└── engines/
    ├── whisper_local.rs   # Local whisper.cpp via whisper-rs
    ├── openai_api.rs      # OpenAI-compatible REST API
    ├── azure_openai.rs    # Azure OpenAI REST API
    └── model_cache.rs     # In-memory whisper model cache
```

## Core trait

All engines implement the async `Transcriber` trait:

```rust
#[async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript>;
}
```

The trait accepts normalized `f32` PCM samples (mono, 16kHz) and returns a `Transcript` containing timed `Segment`s. Audio decoding and format conversion happen upstream in the pipeline — engines only deal with raw samples.

## Processing pipeline

The `pipeline.rs` module orchestrates the full flow:

```
Input file (any format)
  │
  ├─ needs_conversion()? ──→ extract_to_wav() via FFmpeg
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
  │   └─ Transcribe each segment, offset timestamps
  │
  ├─ If not segmenting:
  │   └─ read_wav() → transcribe() directly
  │
  └─ Output:
      ├─ Text to stdout (default)
      ├─ VTT to file or stdout
      └─ JSON manifest to output directory
```

Temporary files use the `tempfile` crate and are cleaned up automatically on drop.

## Engines

### Local (`whisper_local.rs`)

Wraps [whisper-rs](https://github.com/tazz4843/whisper-rs), which binds to whisper.cpp. Inference runs on `tokio::task::spawn_blocking` to avoid blocking the async runtime.

Uses `ModelCache` to avoid reloading the same model across multiple transcription calls (important for segmented processing).

### OpenAI API (`openai_api.rs`)

Sends audio to any OpenAI-compatible `/v1/audio/transcriptions` endpoint via multipart upload. The `base_url` is configurable, so it works with:

- OpenAI (`https://api.openai.com`)
- Self-hosted services (LocalAI, vLLM, etc.)
- Other compatible APIs (Qwen AST, etc.)

### Azure OpenAI (`azure_openai.rs`)

Same multipart upload pattern as OpenAI, but with Azure-specific URL format and `api-key` header authentication instead of Bearer token:

```
{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}
```

### Shared WAV encoding

Both API engines need to send audio as a WAV file. The `audio::wav::encode_wav()` function re-encodes `f32` samples to 16-bit WAV in memory — shared across all API engines to avoid duplication.

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
