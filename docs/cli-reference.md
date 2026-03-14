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
| `-p, --provider` | `local`, `sherpa-onnx`, `openai`, or `azure` | `local` |

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

#### Azure provider options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --base-url` | Azure endpoint URL | `AZURE_OPENAI_ENDPOINT` env var |
| `-a, --api-key` | Azure API key fallback env var (`OPENAI_API_KEY`) | |
| `--azure-api-key` | Azure API key | `AZURE_API_KEY` env var |
| `--azure-deployment` | Deployment name | `AZURE_DEPLOYMENT_NAME` env var, or `whisper` |
| `--azure-api-version` | API version | `AZURE_API_VERSION` env var, or `2024-06-01` |

#### Output options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Directory for text/VTT/SRT output and manifest files | none (stdout) |
| `-f, --output-format` | `text`, `vtt`, or `srt` | `vtt` |
| `--language` | Language hint (e.g. `en`, `es`, `auto`) | `auto` |
| `--normalize` | Normalize audio with ffmpeg `loudnorm` before transcription | disabled |

#### API resilience options

These options apply to OpenAI/Azure providers:

| Option | Description | Default |
|--------|-------------|---------|
| `--max-retries` | Maximum request retries on 429 responses | `5` |
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

When using `openai` or `azure` providers, files exceeding 25MB are automatically segmented even without `--segment`. When using `sherpa-onnx`, segmentation is always enabled with a maximum segment length of 30 seconds.

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
| `SHERPA_ONNX_LIB_DIR` | Path to sherpa-onnx shared libraries (required for build) | none |
| `MODEL_CACHE_DIR` | Directory for downloaded models | `.cache` |
| `HF_TOKEN` | Hugging Face API token (optional) | none |
| `OPENAI_API_KEY` | OpenAI API key | none |
| `AZURE_API_KEY` | Azure API key fallback for Azure provider if `--azure-api-key` is unset | none |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | none |
| `AZURE_DEPLOYMENT_NAME` | Azure deployment name | `whisper` |
| `AZURE_API_VERSION` | Azure API version | `2024-06-01` |
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

# OpenAI API
OPENAI_API_KEY=sk-... transcribeit run -p openai -i recording.mp3

# OpenAI-compatible self-hosted endpoint
transcribeit run -p openai -b http://localhost:8080 \
  -a dummy --remote-model qwen-asr -i recording.wav

# Azure OpenAI
transcribeit run -p azure -i recording.wav \
  -b https://myresource.openai.azure.com \
  -a $AZURE_API_KEY --azure-deployment my-whisper
```

### Provider behavior

- **Local** (`-p local`) runs whisper.cpp in-process using GGML models.
- **Sherpa-ONNX** (`-p sherpa-onnx`) runs sherpa-onnx in-process. Auto-detects Whisper, Moonshine, and SenseVoice models from directory contents. Always auto-segments at 30s.
- **OpenAI-compatible** (`-p openai`) uses `--remote-model` and calls `POST {base-url}/v1/audio/transcriptions`.
- **Azure** (`-p azure`) uses `--azure-deployment` and calls:
  `POST {base-url}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}`.

For the full matrix and upload/auth notes, see: [Provider behavior](provider-behavior.md).  
For benchmark guidance and result templates, see: [Performance benchmarks](performance-benchmarks.md).

## Output files

When `--output-dir` is specified, the following files are created:

- `<input_stem>.txt` — Transcript text file (if `--output-format text`)
- `<input_stem>.vtt` — WebVTT subtitle file (if `--output-format vtt`)
- `<input_stem>.srt` — SRT subtitle file (if `--output-format srt`)
- `<input_stem>.manifest.json` — Processing manifest with metadata

### Manifest format

```json
{
  "input": {
    "file": "meeting.mp4",
    "duration_secs": 3600.0
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
  "segments": [
    {
      "index": 0,
      "start_secs": 0.0,
      "end_secs": 5.25,
      "text": "Hello, welcome to the meeting."
    }
  ],
  "stats": {
    "total_duration_secs": 3600.0,
    "total_segments": 42,
    "total_characters": 15000,
    "processing_time_secs": 120.5
  }
}
```
