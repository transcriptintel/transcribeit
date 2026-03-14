# CLI Reference

## Commands

### `download-model`

Download a Whisper GGML model from Hugging Face.

```bash
transcribeit download-model [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --model-size` | Model size | `base` |
| `-o, --output-dir` | Override download directory | `MODEL_CACHE_DIR` |
| `-t, --hf-token` | Hugging Face token | `HF_TOKEN` env var |

Available model sizes: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v3`, `large-v3-turbo`.

### `list-models`

List downloaded models with file sizes.

```bash
transcribeit list-models [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --dir` | Override models directory | `MODEL_CACHE_DIR` |

### `run`

Transcribe an audio or video file.

```bash
transcribeit run [OPTIONS] --input <FILE>
```

#### Input options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Path to audio/video file (any format) | required |
| `-p, --provider` | `local`, `openai`, or `azure` | `local` |

#### Local provider options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to GGML model file | required |

#### OpenAI provider options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --base-url` | API base URL | `https://api.openai.com` |
| `-a, --api-key` | API key | `OPENAI_API_KEY` env var |
| `--remote-model` | Model name | `whisper-1` |

#### Azure provider options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --base-url` | Azure endpoint URL | `https://api.openai.com` |
| `-a, --api-key` | Azure API key | `OPENAI_API_KEY` env var |
| `--azure-deployment` | Deployment name | `whisper` |
| `--azure-api-version` | API version | `2024-06-01` |

#### Output options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Directory for VTT and manifest files | none (stdout) |
| `--output-format` | `text` or `vtt` | `text` |

#### Segmentation options

| Option | Description | Default |
|--------|-------------|---------|
| `--segment` | Enable silence-based segmentation | disabled |
| `--silence-threshold` | Silence threshold in dB (negative) | `-40` |
| `--min-silence-duration` | Minimum silence duration in seconds | `0.8` |
| `--max-segment-secs` | Maximum segment length in seconds | `600` |

When using `openai` or `azure` providers, files exceeding 25MB are automatically segmented even without `--segment`.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_CACHE_DIR` | Directory for downloaded models | `.cache` |
| `HF_TOKEN` | Hugging Face API token | none |
| `OPENAI_API_KEY` | OpenAI / Azure API key | none |

All variables can be set in a `.env` file in the project root.

## Examples

```bash
# Download models
transcribeit download-model -s base
transcribeit download-model -s small.en

# Basic local transcription (any format)
transcribeit run -i recording.mp3 -m .cache/ggml-base.bin
transcribeit run -i meeting.mp4 -m .cache/ggml-small.en.bin

# VTT subtitles with segmentation
transcribeit run -i lecture.mp4 -m .cache/ggml-base.bin \
  --output-format vtt --segment -o ./output

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

## Output files

When `--output-dir` is specified, the following files are created:

- `<input_stem>.vtt` — WebVTT subtitle file (if `--output-format vtt`)
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
    "output_format": "vtt"
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
