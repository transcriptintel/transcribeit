# CLI Reference

## Commands

### `download-model`

Download a Whisper model in GGML or ONNX format.

```bash
transcribeit download-model [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --model-size` | Model size | `base` |
| `-f, --format` | Model format: `ggml` or `onnx` | `ggml` |
| `-o, --output-dir` | Override download directory | `MODEL_CACHE_DIR` |
| `-t, --hf-token` | Hugging Face token (GGML only) | `HF_TOKEN` env var |

Available model sizes: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v3`, `large-v3-turbo`.

GGML models are downloaded from Hugging Face (`ggerganov/whisper.cpp`). ONNX models are downloaded from the sherpa-onnx GitHub releases as `.tar.bz2` archives and extracted automatically. Note: `large-v3` is not available in ONNX format.

### `list-models`

List downloaded models with file sizes. Shows both `[ggml]` and `[onnx]` models. GGML models appear as `.bin` files with sizes; ONNX models appear as directories with a trailing `/`.

```bash
transcribeit list-models [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --dir` | Override models directory | `MODEL_CACHE_DIR` |

### `run`

Transcribe audio/video files.

```bash
transcribeit run [OPTIONS] --input <FILE_OR_PATH_OR_GLOB>
```

#### Input options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input path, directory, or glob pattern for audio/video files | required |
| `-p, --provider` | `local`, `sherpa-onnx`, `openai`, `azure`, `qwen-filetrans`, `gemini`, or `nvidia-riva` | `local` |

#### Local provider options (`-p local`)

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to GGML model file or cache alias (`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v3`, `large-v3-turbo`) | required |

Model aliases auto-resolve from the `MODEL_CACHE_DIR` cache directory (default `.cache`).

#### Sherpa-ONNX provider options (`-p sherpa-onnx`)

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to ONNX model directory or partial name (e.g. `tiny`, `base.en`, `moonshine-base`, `sense-voice`) | required |

The engine auto-detects the model architecture from files in the directory:

- **Whisper** -- `encoder.onnx` + `decoder.onnx` (or int8 variants) + `tokens.txt`
- **Moonshine** -- `preprocess.onnx` + `encode.onnx` + `uncached_decode.onnx` + `cached_decode.onnx` + `tokens.txt`
- **SenseVoice** -- `model.onnx` + `tokens.txt`

When an alias like `base.en` is given, the cache is searched for a directory named `sherpa-onnx-whisper-base.en` under `MODEL_CACHE_DIR`. The resolver also supports glob matching, so partial names like `-m moonshine-base` or `-m sense-voice` will match any directory in the cache containing that string.

Sherpa-ONNX automatically enables segmentation and caps segment length at 30 seconds due to the Whisper ONNX model limitation.

#### OpenAI provider options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --base-url` | API base URL | `https://api.openai.com` |
| `-a, --api-key` | API key | `OPENAI_API_KEY` env var |
| `--remote-model` | Model name | `whisper-1` |

Supported hosted OpenAI transcription models include `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`, and `gpt-4o-transcribe-diarize`.

`whisper-1` returns timestamped segments through the default `verbose_json` request path. `gpt-4o-mini-transcribe` and `gpt-4o-transcribe` return plain transcript text through the current CLI. When `--diarize` is set and no `--remote-model` is provided, the CLI selects `gpt-4o-transcribe-diarize`. When `gpt-4o-transcribe-diarize` is selected, the provider requests `diarized_json` with `chunking_strategy=auto` and maps speaker labels into VTT/SRT/manifest output.

#### Azure provider options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --base-url` | Azure endpoint URL | `AZURE_OPENAI_ENDPOINT` env var |
| `-a, --api-key` | Azure API key fallback env var (`OPENAI_API_KEY`) | |
| `--azure-api-key` | Azure API key | `AZURE_API_KEY` env var |
| `--azure-deployment` | Deployment name | `AZURE_DEPLOYMENT_NAME` env var, or `whisper` |
| `--azure-api-version` | API version | `AZURE_API_VERSION` env var, or `2024-06-01` |

#### Qwen file transcription provider options (`-p qwen-filetrans`)

| Option | Description | Default |
|--------|-------------|---------|
| `--dashscope-api-key` | DashScope API key | `DASHSCOPE_API_KEY` env var |
| `--qwen-api-base-url` | DashScope async ASR API base URL | `DASHSCOPE_ASR_BASE_URL` env var, or `https://dashscope-intl.aliyuncs.com/api/v1` |
| `--remote-model` | Qwen file transcription model | `qwen3-asr-flash-filetrans` |
| `--s3-bucket` | S3 bucket for temporary audio uploads | `S3_BUCKET` env var |
| `--s3-region` | S3 region | `S3_REGION` or `AWS_REGION` env var |
| `--s3-endpoint-url` | S3-compatible endpoint URL | `S3_ENDPOINT_URL` env var |
| `--s3-access-key-id` | S3 access key ID | `S3_ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID` env var |
| `--s3-secret-access-key` | S3 secret access key | `S3_SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY` env var |
| `--s3-session-token` | S3 session token | `S3_SESSION_TOKEN` or `AWS_SESSION_TOKEN` env var |
| `--s3-prefix` | S3 object prefix | `transcribeit/qwen-filetrans` |
| `--s3-presign-expires-secs` | Pre-signed URL expiry in seconds | `3600` |
| `--s3-force-path-style` | Force path-style URLs for S3-compatible storage | disabled |

Qwen file transcription uploads the prepared audio to S3-compatible storage and passes a pre-signed GET URL to DashScope. The provider is intended for whole-file transcription; avoid `--segment` unless you explicitly want multiple independent remote jobs.

When available, Qwen manifests include `provider_metadata.provider = "qwen-filetrans"` and Qwen task timing/usage, audio info, and transcript counts under `provider_metadata.data`. Word-level timestamps remain on each normalized segment. Temporary pre-signed URLs are not persisted.

If a short-audio `qwen3-asr-flash` model is selected with `-p qwen-filetrans`, the CLI validates the file size and duration before upload and fails without staging the file to S3. Use `qwen3-asr-flash-filetrans` for this provider.

#### Gemini provider options (`-p gemini`)

| Option | Description | Default |
|--------|-------------|---------|
| `--gemini-api-key` | Gemini API key | `GEMINI_API_KEY` env var |
| `--gemini-api-base-url` | Gemini API base URL | `GEMINI_API_BASE_URL` env var, or `https://generativelanguage.googleapis.com/v1beta` |
| `--remote-model` | Gemini model name | `gemini-3.5-flash` |

The Gemini provider uses the Gemini Files API plus streamed `streamGenerateContent` with structured JSON output. It converts input audio/video to 16 kHz mono MP3 before upload, then asks Gemini for a transcript object with `text`, `segments`, timestamps, speaker, language, and emotion fields.

Current model candidates verified through the Gemini models API include `gemini-3.5-flash`, `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, and `gemini-2.5-flash`. Prefer stable `gemini-3.5-flash` for the default path and benchmark preview models before adopting them in production workflows.

Gemini timestamps and speaker labels are generated structured output rather than a dedicated ASR response schema. The parser is defensive: invalid JSON, missing fields, empty segments, unknown future response fields, and streamed response shape changes fall back to transcript text instead of failing the run.

#### NVIDIA Riva provider options (`-p nvidia-riva`)

| Option | Description | Default |
|--------|-------------|---------|
| `--nvidia-api-key` | NVIDIA API key | `NVIDIA_API_KEY` env var |
| `--nvidia-riva-function-id` | Hosted Riva function id | `NVIDIA_RIVA_FUNCTION_ID` env var |
| `--nvidia-riva-server` | Riva gRPC server | `NVIDIA_RIVA_SERVER` env var, or `grpc.nvcf.nvidia.com:443` |
| `--remote-model` | Optional Riva model name sent in `RecognitionConfig.model` | none |

The NVIDIA Riva provider uses gRPC and sends `function-id` plus Bearer authorization metadata. It converts input audio/video to 16 kHz mono WAV and requests provider-native word timestamps and automatic punctuation. If `--diarize` is set, it enables Riva speaker diarization with a default maximum of 4 speakers. Use `--speakers N` to override that maximum.

Manifests include `provider_metadata.provider = "nvidia-riva"`, audio details, request ids, feature flags, detected languages, response counts, elapsed time, and mean confidence when returned.

#### Output options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Directory for text/VTT/SRT output and manifest files | none (stdout) |
| `-f, --output-format` | `text`, `vtt`, or `srt` | `vtt` |
| `--language` | Language hint (e.g. `en`, `es`, `auto`) | `auto` |
| `--normalize` | Normalize audio with ffmpeg `loudnorm` before transcription | disabled |

#### API resilience options

These options apply to OpenAI, Azure, Qwen file transcription, Gemini, and NVIDIA Riva providers where supported:

| Option | Description | Default |
|--------|-------------|---------|
| `--max-retries` | Maximum request retries on 429 responses for REST API providers | `5` |
| `--request-timeout-secs` | Timeout in seconds for each API request | `120` |
| `--retry-wait-base-secs` | Initial wait time used when rate-limited | `10` |
| `--retry-wait-max-secs` | Maximum wait time when parsing retry delay | `120` |

#### Segmentation options

| Option | Description | Default |
|--------|-------------|---------|
| `--segment` | Enable silence-based segmentation | disabled |
| `--silence-threshold` | Silence threshold in dB (negative) | `-40` |
| `--min-silence-duration` | Minimum silence duration in seconds | `0.8` |
| `--max-segment-secs` | Maximum segment length in seconds | `600` |
| `--segment-concurrency` | Max parallel segment requests (API providers only) | `2` |
| `--vad-model` | Path to Silero VAD ONNX model (`silero_vad.onnx`) for speech-aware segmentation | `VAD_MODEL` env var |

When using `openai`, `azure`, `qwen-filetrans`, or `nvidia-riva` providers, files exceeding the conservative 25MB auto-split threshold are automatically segmented even without `--segment`. This keeps long remote requests smaller and more reliable. Gemini stays whole-file by default to preserve model-level speaker continuity; use `--segment` only when you want independent chunk requests, or let the provider fall back to segmentation if a long whole-file request fails. When using `sherpa-onnx`, segmentation is always enabled with a maximum segment length of 30 seconds.

When `--vad-model` is set and segmentation is needed, VAD-based segmentation is used instead of FFmpeg `silencedetect`. VAD detects actual speech boundaries using Silero VAD, avoiding mid-word cuts. It pads chunks by 250ms, merges gaps shorter than 200ms, and splits long chunks at low-energy points. This requires the `sherpa-onnx` feature to be enabled. When `--vad-model` is not set, the original FFmpeg silence-based segmentation is used as a fallback.

#### Speaker diarization options

| Option | Description | Default |
|--------|-------------|---------|
| `--diarize` | Enable speaker diarization | disabled |
| `--speakers` | Speaker count or provider-specific maximum speaker hint | none |
| `--diarize-segmentation-model` | Path to pyannote segmentation ONNX model | `DIARIZE_SEGMENTATION_MODEL` env var |
| `--diarize-embedding-model` | Path to speaker embedding ONNX model | `DIARIZE_EMBEDDING_MODEL` env var |

For local post-processing diarization, use `--diarize --speakers N`. Both `--diarize-segmentation-model` and `--diarize-embedding-model` are required because the current Sherpa diarizer needs a fixed speaker count. Speaker labels appear in VTT output as `<v Speaker 0>`, in SRT output as `[Speaker 0]`, and in manifest JSON as a `"speaker"` field on each segment. Requires the `sherpa-onnx` feature.

For OpenAI, `--diarize` uses provider-native diarization through `gpt-4o-transcribe-diarize` unless a different `--remote-model` is explicitly selected.

For NVIDIA Riva, use `--diarize` when the exact speaker count is unknown. The provider uses `--speakers N` as a maximum speaker hint; if omitted, the CLI sends a default maximum of 4 speakers.

For Gemini, speaker labels are model-generated structured output and may be present even without local diarization. For Qwen file transcription, Azure, local Whisper, and non-diarizing OpenAI models, `--diarize` requires the local Sherpa diarizer.

## Output behavior

During transcription, the CLI shows an animated spinner in the terminal so you can see progress while waiting for Whisper/API calls to complete.

### `text` output format

- If `--output-dir` is set, output is written to `<input_stem>.txt`.
- If `--output-dir` is not set, output is printed to stdout.

When `--input` resolves to multiple files (directory or glob), all files are processed sequentially with the same provider/model. For API providers, model/auth setup is reused for efficiency.

### `vtt` output format

- If `--output-dir` is set, output is written to `<input_stem>.vtt`.
- If `--output-dir` is not set, output is printed to stdout.

### `srt` output format

- If `--output-dir` is set, output is written to `<input_stem>.srt`.
- If `--output-dir` is not set, output is printed to stdout.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHERPA_ONNX_LIB_DIR` | Path to sherpa-onnx shared libraries (required when building with `--features sherpa-onnx`) | none |
| `MODEL_CACHE_DIR` | Directory for downloaded models | `.cache` |
| `HF_TOKEN` | Hugging Face API token (optional) | none |
| `OPENAI_API_KEY` | OpenAI API key | none |
| `GEMINI_API_KEY` | Gemini API key | none |
| `GEMINI_API_BASE_URL` | Gemini API base URL | `https://generativelanguage.googleapis.com/v1beta` |
| `NVIDIA_API_KEY` | NVIDIA hosted Riva API key | none |
| `NVIDIA_RIVA_FUNCTION_ID` | NVIDIA hosted Riva function id | none |
| `NVIDIA_RIVA_SERVER` | NVIDIA Riva gRPC server | `grpc.nvcf.nvidia.com:443` |
| `AZURE_API_KEY` | Azure API key fallback for Azure provider if `--azure-api-key` is unset | none |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | none |
| `AZURE_DEPLOYMENT_NAME` | Azure deployment name | `whisper` |
| `AZURE_API_VERSION` | Azure API version | `2024-06-01` |
| `DASHSCOPE_API_KEY` | DashScope API key for Qwen providers | none |
| `DASHSCOPE_ASR_BASE_URL` | DashScope async ASR base URL for Qwen file transcription | `https://dashscope-intl.aliyuncs.com/api/v1` |
| `S3_BUCKET` | S3 bucket for Qwen file transcription staging | none |
| `S3_REGION` / `AWS_REGION` | S3 region for Qwen file transcription staging | none |
| `S3_ENDPOINT_URL` | S3-compatible endpoint URL | none |
| `S3_ACCESS_KEY_ID` / `AWS_ACCESS_KEY_ID` | S3 access key ID | none |
| `S3_SECRET_ACCESS_KEY` / `AWS_SECRET_ACCESS_KEY` | S3 secret access key | none |
| `S3_SESSION_TOKEN` / `AWS_SESSION_TOKEN` | S3 session token | none |
| `S3_PREFIX` | S3 object prefix for Qwen staging uploads | `transcribeit/qwen-filetrans` |
| `S3_PRESIGN_EXPIRES_SECS` | S3 pre-signed URL expiry in seconds | `3600` |
| `S3_FORCE_PATH_STYLE` | Force path-style URLs for S3-compatible storage | `false` |
| `VAD_MODEL` | Path to Silero VAD ONNX model for speech-aware segmentation | none |
| `DIARIZE_SEGMENTATION_MODEL` | Path to pyannote segmentation ONNX model for speaker diarization | none |
| `DIARIZE_EMBEDDING_MODEL` | Path to speaker embedding ONNX model for speaker diarization | none |
| `TRANSCRIBEIT_MAX_RETRIES` | Maximum 429 retries | `5` |
| `TRANSCRIBEIT_REQUEST_TIMEOUT_SECS` | API request timeout in seconds | `120` |
| `TRANSCRIBEIT_RETRY_WAIT_BASE_SECS` | Base retry wait time in seconds | `10` |
| `TRANSCRIBEIT_RETRY_WAIT_MAX_SECS` | Maximum retry wait time in seconds | `120` |

All variables can be set in a `.env` file in the project root.

## Examples

```bash
# Download GGML models
transcribeit download-model -s base
transcribeit download-model -s small.en

# Download ONNX models (for sherpa-onnx provider)
transcribeit download-model -f onnx -s base.en
transcribeit download-model -f onnx -s tiny

# List all downloaded models (shows [ggml] and [onnx] tags)
transcribeit list-models

# Process a single file with local whisper.cpp (using cache alias)
transcribeit run -i recording.mp3 -m base
# Process a single file (explicit path)
transcribeit run -i recording.mp3 -m .cache/ggml-base.bin
transcribeit run -i meeting.mp4 -m .cache/ggml-small.en.bin

# Process with sherpa-onnx Whisper (auto-segments at 30s)
transcribeit run -p sherpa-onnx -i recording.mp3 -m base.en
transcribeit run -p sherpa-onnx -i lecture.mp4 -m tiny -f vtt -o ./output

# Process with sherpa-onnx Moonshine (auto-detected from model files)
transcribeit run -p sherpa-onnx -i recording.mp3 -m moonshine-base

# Process with sherpa-onnx SenseVoice (auto-detected from model files)
transcribeit run -p sherpa-onnx -i recording.mp3 -m sense-voice

# Process a directory
transcribeit run --input samples/ --output-dir ./output

# Process a glob
transcribeit run --input \"samples/**/*.mp4\" -p azure --output-dir ./output

# VTT subtitles with segmentation (vtt is the default format)
transcribeit run -i lecture.mp4 -m .cache/ggml-base.bin --segment -o ./output

# Plain text output
transcribeit run -i lecture.mp4 -m base -f text

# SRT subtitles
transcribeit run -i lecture.mp4 -m base -f srt -o ./output

# Tune segmentation for noisy audio
transcribeit run -i noisy.wav -m .cache/ggml-base.bin \
  --segment --silence-threshold -30 --min-silence-duration 0.5

# VAD-based segmentation (avoids mid-word cuts)
transcribeit run -p sherpa-onnx -i lecture.mp4 -m base.en \
  --vad-model /path/to/silero_vad.onnx -f vtt -o ./output

# VAD with env var (set VAD_MODEL in .env)
VAD_MODEL=/path/to/silero_vad.onnx transcribeit run -p sherpa-onnx -i recording.mp3 -m base.en

# Speaker diarization (2 speakers)
transcribeit run -p sherpa-onnx -i meeting.mp4 -m base.en \
  --diarize --speakers 2 \
  --diarize-segmentation-model /path/to/segmentation.onnx \
  --diarize-embedding-model /path/to/embedding.onnx \
  -f vtt -o ./output

# VAD + speaker diarization combined
transcribeit run -p sherpa-onnx -i interview.wav -m base.en \
  --vad-model /path/to/silero_vad.onnx \
  --diarize --speakers 2 \
  --diarize-segmentation-model /path/to/segmentation.onnx \
  --diarize-embedding-model /path/to/embedding.onnx \
  -f srt -o ./output

# OpenAI API
OPENAI_API_KEY=sk-... transcribeit run -p openai -i recording.mp3

# OpenAI-compatible self-hosted endpoint
transcribeit run -p openai -b http://localhost:8080 \
  -a dummy --remote-model qwen-asr -i recording.wav

# Azure OpenAI
transcribeit run -p azure -i recording.wav \
  -b https://myresource.openai.azure.com \
  -a $AZURE_API_KEY --azure-deployment my-whisper

# Qwen file transcription with S3 staging
transcribeit run -p qwen-filetrans -i recording.mp3 \
  --dashscope-api-key "$DASHSCOPE_API_KEY" \
  --s3-bucket "$S3_BUCKET" \
  --s3-region "$S3_REGION" \
  --s3-access-key-id "$S3_ACCESS_KEY_ID" \
  --s3-secret-access-key "$S3_SECRET_ACCESS_KEY" \
  -f vtt -o ./output

# NVIDIA hosted Riva ASR
transcribeit run -p nvidia-riva -i recording.wav \
  --nvidia-api-key "$NVIDIA_API_KEY" \
  --nvidia-riva-function-id "$NVIDIA_RIVA_FUNCTION_ID" \
  --language en-US -f vtt -o ./output
```

### Provider behavior

- **Local** (`-p local`) runs whisper.cpp in-process using GGML models.
- **Sherpa-ONNX** (`-p sherpa-onnx`) runs sherpa-onnx in-process. Auto-detects Whisper, Moonshine, and SenseVoice models from directory contents. Always auto-segments at 30s.
- **OpenAI-compatible** (`-p openai`) uses `--remote-model` and calls `POST {base-url}/v1/audio/transcriptions`.
  `gpt-4o-transcribe-diarize` is handled specially: the request includes `response_format=diarized_json` and `chunking_strategy=auto`, and response segments are parsed defensively so unknown or missing fields do not fail the run.
- **Azure** (`-p azure`) uses `--azure-deployment` and calls:
  `POST {base-url}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}`.
- **Qwen file transcription** (`-p qwen-filetrans`) uploads audio to S3-compatible storage, passes a pre-signed URL to DashScope, and polls the async transcription task.
- **Gemini** (`-p gemini`) uploads audio through Gemini Files API, calls streamed `streamGenerateContent`, and parses structured transcript JSON defensively.
- **NVIDIA Riva** (`-p nvidia-riva`) sends WAV audio to hosted Riva gRPC, requesting native word timestamps and optional server-side diarization.

For the full matrix and upload/auth notes, see: [Provider behavior](provider-behavior.md).  
For benchmark guidance and result templates, see: [Performance benchmarks](performance-benchmarks.md).

## Output files

When `--output-dir` is specified, the following files are created:

- `<input_stem>.txt` — Transcript text file (if `--output-format text`)
- `<input_stem>.vtt` — WebVTT subtitle file (if `--output-format vtt`)
- `<input_stem>.srt` — SRT subtitle file (if `--output-format srt`)
- `<input_stem>.manifest.json` — Processing manifest with metadata

### Manifest format

Manifests use `schema_version: "transcribeit.manifest.v2"`. New consumers should prefer `transcript.text`, `transcript.segments`, `capabilities`, `quality`, and the `provider_metadata` envelope. The top-level `segments` array remains for compatibility with earlier consumers.

```json
{
  "schema_version": "transcribeit.manifest.v2",
  "input": {
    "file": "meeting.mp4",
    "duration_secs": 3600.0,
    "duration_ms": 3600000
  },
  "config": {
    "provider": "local",
    "model": ".cache/ggml-base.bin",
    "segmentation_enabled": true,
    "silence_threshold_db": -40.0,
    "min_silence_duration_secs": 0.8,
    "output_format": "vtt",
    "language": "en",
    "normalized_audio": true
  },
  "capabilities": {
    "segments": true,
    "word_timestamps": true,
    "speaker_labels": true,
    "language_per_segment": true,
    "emotion_per_segment": true,
    "native_timestamps": true
  },
  "quality": {
    "timing_source": "provider_native",
    "timing_reliable": true,
    "timestamps_clamped": false,
    "speaker_source": "provider_native",
    "warnings": []
  },
  "transcript": {
    "text": "Hello, welcome to the meeting.",
    "segments": [
      {
        "id": "seg_000001",
        "index": 0,
        "start_secs": 0.0,
        "end_secs": 5.25,
        "start_ms": 0,
        "end_ms": 5250,
        "text": "Hello, welcome to the meeting.",
        "speaker": "Speaker 0",
        "language": "en",
        "emotion": "neutral",
        "words": [
          {
            "id": "seg_000001_word_000001",
            "index": 0,
            "start_secs": 0.0,
            "end_secs": 0.4,
            "start_ms": 0,
            "end_ms": 400,
            "text": "Hello",
            "punctuation": ","
          }
        ]
      }
    ]
  },
  "segments": [
    {
      "id": "seg_000001",
      "index": 0,
      "start_secs": 0.0,
      "end_secs": 5.25,
      "start_ms": 0,
      "end_ms": 5250,
      "text": "Hello, welcome to the meeting.",
      "speaker": "Speaker 0",
      "language": "en",
      "emotion": "neutral",
      "words": []
    }
  ],
  "stats": {
    "total_duration_secs": 3600.0,
    "total_duration_ms": 3600000,
    "total_segments": 42,
    "total_characters": 15000,
    "processing_time_secs": 120.5,
    "processing_time_ms": 120500
  },
  "provider_metadata": {
    "provider": "qwen-filetrans",
    "schema_version": "qwen-filetrans.metadata.v1",
    "data": {
      "model": "qwen3-asr-flash-filetrans",
      "task": {
        "task_status": "SUCCEEDED",
        "usage": {
          "seconds": 3600
        }
      },
      "result": {
        "audio_info": {
          "format": "mp3",
          "sample_rate": 16000
        },
        "file_url_present": true,
        "transcript_count": 1,
        "sentence_count": 42,
        "word_count": 8000,
        "words_enabled": true
      }
    }
  }
}
```
