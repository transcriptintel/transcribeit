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
│   ├── vad.rs             # VAD-based speech segmentation (Silero VAD via sherpa-onnx)
│   └── wav.rs             # WAV reading and encoding (shared)
├── diarize/
│   ├── mod.rs             # Speaker diarization engine and speaker assignment
│   └── ffi.rs             # Raw C FFI bindings for sherpa-onnx speaker diarization
├── output/
│   ├── vtt.rs             # WebVTT subtitle writer (supports <v Speaker N> tags)
│   ├── srt.rs             # SRT subtitle writer (supports [Speaker N] labels)
│   └── manifest.rs        # JSON manifest writer (includes speaker labels)
└── engines/
    ├── whisper_local.rs   # Local whisper.cpp via whisper-rs
    ├── sherpa_onnx.rs     # Local sherpa-onnx engine (auto-detects Whisper, Moonshine, SenseVoice)
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

- **Local engine** (`whisper_local`) uses `transcribe()` — it needs decoded samples for whisper.cpp.
- **Sherpa-ONNX engine** (`sherpa_onnx`) uses `transcribe()` — it needs decoded samples for the ONNX runtime.
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
  │   ├─ Auto: remote provider + estimated size > 25MB
  │   └─ Auto: sherpa-onnx provider (always segments; max 30s per chunk)
  │
  ├─ If segmenting:
  │   ├─ VAD path (when --vad-model is set and sherpa-onnx feature is enabled):
  │   │   ├─ read_wav_bytes() → f32 PCM samples
  │   │   ├─ vad_segment(): detect speech → pad 250ms → merge gaps <200ms → split long chunks at low-energy points
  │   │   ├─ Extract chunk samples directly from memory
  │   │   └─ Transcribe each chunk via transcribe(), offset timestamps
  │   ├─ FFmpeg fallback (no VAD model, or sherpa-onnx feature disabled):
  │   │   ├─ detect_silence() via FFmpeg silencedetect filter
  │   │   ├─ compute_segments() at silence midpoints
  │   │   ├─ split_audio() into temp WAV files
  │   │   └─ Transcribe each segment, offset timestamps (concurrently for API providers)
  │
  ├─ If not segmenting:
  │   ├─ Local: read_wav() → transcribe() directly
  │   └─ API: transcribe_path() with prepared file
  │
  ├─ normalize_audio? ──→ optional loudnorm filter in ffmpeg conversion pipeline
  ├─ Speaker diarization? (when --speakers N is set)
  │   ├─ read audio samples for diarization
  │   ├─ Diarizer.diarize() → speaker-labeled time spans
  │   └─ assign_speakers() overlays speaker labels onto transcript segments
  │
  └─ Output:
      ├─ Text to stdout or `<input_stem>.txt`
      ├─ VTT to file or stdout (with `<v Speaker N>` tags when diarized)
      ├─ SRT to file or stdout (with `[Speaker N]` labels when diarized)
      └─ JSON manifest to output directory (includes speaker field per segment)
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
| **Moonshine** | `preprocess.onnx` + `encode.onnx` + `uncached_decode.onnx` + `cached_decode.onnx` | `OfflineMoonshineModelConfig` |
| **SenseVoice** | `model.onnx` (single file) | `OfflineSenseVoiceModelConfig` |

All architectures also require a `tokens.txt` (or `*-tokens.txt`) file in the model directory. The engine prefers `int8` quantized ONNX files when available (e.g., `encoder.int8.onnx`) for lower memory usage, falling back to full-precision variants.

The model resolver supports glob-based directory matching, so you can use partial names like `-m moonshine-base` or `-m sense-voice` to find models in the cache directory.

**SenseVoice limitation:** SenseVoice models can detect emotions and audio events (laughter, applause, music), but these tags are stripped by the sherpa-onnx C API and are not available in the transcription output.

Whisper ONNX models only support audio chunks of 30 seconds or less, so the pipeline automatically enables segmentation and caps `--max-segment-secs` at 30 when using this provider.

#### C++ stderr suppression

During `recognizer.decode()`, the sherpa-onnx C++ library prints warnings to stderr. The engine temporarily redirects stderr to `/dev/null` via `libc::dup`/`dup2` during decode calls and restores it immediately after, keeping the terminal output clean.

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

## Build requirements

The `sherpa-onnx` Cargo feature is **enabled by default**. It requires the sherpa-onnx shared libraries at both compile time and runtime. The `build.rs` script loads a `.env` file and reads `SHERPA_ONNX_LIB_DIR` to configure the linker search path and embed an `rpath` so the binary can find the dylibs at runtime.

Set `SHERPA_ONNX_LIB_DIR` in your `.env` file or environment before building:

```bash
# .env
SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
```

To build without the sherpa-onnx dependency entirely:

```bash
cargo build --release --no-default-features
```

This removes the sherpa-onnx provider and eliminates the need for `SHERPA_ONNX_LIB_DIR`.

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
4. Add a new `Provider` variant in `main.rs`
5. Add a match arm in the `Command::Run` handler to construct your engine
