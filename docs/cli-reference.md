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
| `-p, --provider` | `local`, `sherpa-onnx`, `openai`, `azure`, `qwen-filetrans`, `gemini`, `nvidia-riva`, or `deepgram` | `local` |

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
| `--gemini-file-cache` | Reuse Gemini Files API uploads keyed by SHA-256 of prepared upload bytes | disabled |
| `--gemini-use-presigned-url` | Stage prepared audio in S3-compatible storage and pass the pre-signed URL as Gemini `file_uri` | disabled |
| `--gemini-file-cache-index` | Local Gemini file cache index path | `GEMINI_FILE_CACHE_INDEX` env var, or `.cache/transcribeit/gemini-files.json` |
| `--gemini-autoclean` | Deprecated alias for `--autoclean` for Gemini temporary uploads | disabled |
| `--gemini-explicit-cache` | Create and reuse Gemini explicit `cachedContent` objects for prepared audio | disabled |
| `--gemini-cache-ttl-secs` | TTL in seconds for Gemini explicit `cachedContent` objects | `3600` |

The Gemini provider uses the Gemini Files API plus streamed `streamGenerateContent` with structured JSON output. It converts input audio/video to 16 kHz mono MP3 before upload, then asks Gemini for a transcript object with `text`, `segments`, timestamps, speaker, language, and emotion fields.

With `--gemini-use-presigned-url`, the CLI uploads the prepared MP3 to S3-compatible storage, generates a pre-signed GET URL, and sends that URL as Gemini `file_uri` instead of using the Gemini Files API. This mode is useful for one-off inputs up to 100 MB when S3/R2 staging is already configured. It is rejected for Gemini 2.0 family models and cannot be combined with `--gemini-file-cache` or `--gemini-explicit-cache`; use the Files API path when Gemini file reuse or explicit cached content is required.

By default, Gemini Files API uploads are deleted after each run. With `--gemini-file-cache`, the CLI stores a local JSON index for the uploaded Gemini file reference and keeps the remote file for reuse within the Gemini Files API retention window. The cache key is the SHA-256 hash of the exact prepared 16 kHz mono MP3 bytes, not the input path. Before reuse, the CLI calls `files.get` and only reuses files that still exist and are `ACTIVE`. Use `--autoclean` to force deletion after a run while keeping the same command shape for experiments; `--gemini-autoclean` remains as a deprecated Gemini-only alias.

With `--gemini-explicit-cache`, the CLI also creates or reuses a Gemini `cachedContent` object for the prepared audio and passes its name as `cachedContent` in the streamed generation request. This automatically enables the local Gemini file cache index because the cached-content handle must be persisted between runs. Explicit cached content has its own TTL and provider billing behavior; it is separate from the 48-hour Files API upload retention window.

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

#### Deepgram provider options (`-p deepgram`)

| Option | Description | Default |
|--------|-------------|---------|
| `--deepgram-api-key` | Deepgram API key | `DEEPGRAM_API_KEY` env var |
| `--deepgram-api-base-url` | Deepgram API base URL | `DEEPGRAM_API_BASE_URL` env var, or `https://api.deepgram.com/v1` |
| `--remote-model` | Deepgram model name | `nova-3` |
| `--deepgram-intelligence` | Enable summary, topics, intents, entity detection, and sentiment | disabled |
| `--deepgram-summarize` | Enable `summarize=v2` | disabled |
| `--deepgram-topics` | Enable topic detection | disabled |
| `--deepgram-intents` | Enable intent recognition | disabled |
| `--deepgram-detect-entities` | Enable entity detection | disabled |
| `--deepgram-sentiment` | Enable sentiment analysis | disabled |
| `--deepgram-keyterm` | Nova-3 keyterm prompt; repeat or comma-separate terms | none |
| `--deepgram-search` | Search term or phrase; repeat or comma-separate terms | none |
| `--deepgram-redact` | Redaction target such as `pii`, `phi`, `pci`, `numbers`, or entity class | none |
| `--deepgram-replace` | Find/replace rule in `FIND:REPLACE` format | none |
| `--deepgram-filler-words` | Enable filler word transcription | disabled |
| `--deepgram-numerals` | Enable numerals formatting | disabled |
| `--deepgram-use-presigned-url` | Stage prepared audio in S3-compatible storage and send Deepgram a JSON URL request | disabled |

The Deepgram provider calls `POST {deepgram-api-base-url}/listen` with `smart_format=true` and `utterances=true`. Input audio/video is converted to 16 kHz mono WAV when needed. The default model is `nova-3`; `nova-3-general` and `nova-3-medical` can be passed through `--remote-model` when available for the account/region.

By default, Deepgram receives direct audio bytes. With `--deepgram-use-presigned-url`, the CLI uploads the prepared audio to S3-compatible storage, generates a pre-signed GET URL, and sends Deepgram `{"url":"..."}`. This uses the same `S3_*` settings as Qwen file transcription; if `S3_PREFIX` is unset, the Deepgram default prefix is `transcribeit/deepgram`. The pre-signed URL itself is not written to manifests.

If `--diarize` is set, the request uses `diarize_model=latest`, which enables Deepgram's current provider-native batch diarizer. `--speakers N` is accepted as a request to enable diarization, but Deepgram does not accept an exact speaker-count hint through this path.

`--deepgram-intelligence` is the convenience switch for Transcript Intelligence workflows. It enables Deepgram summarization, topic detection, intent recognition, entity detection, and sentiment analysis in the same transcription request. Returned intelligence blocks are stored under `provider_metadata.data.intelligence`; they are useful downstream metadata, but should still be treated as model output rather than validated facts.

For Nova-3 and Nova-3 Medical, use `--deepgram-keyterm` for important domain terms and brands. In local benchmarking, keyterms such as `Ofev`, `Esbriet`, `IPF`, and `Producta` materially improved medical brand recognition and speaker consistency.

Manifests include `provider_metadata.provider = "deepgram"`, Deepgram request metadata, channel/utterance/alternative counts, model info, intelligence token usage metadata, mean confidence, extracted intelligence blocks, and `timestamps_clamped` when provider timestamps exceed the reported media duration and are clamped for output safety.

#### Output options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Directory for text/VTT/SRT output and manifest files | none (stdout) |
| `-f, --output-format` | `text`, `vtt`, or `srt` | `vtt` |
| `--language` | Language hint (e.g. `en`, `es`, `auto`) | `auto` |
| `--normalize` | Normalize audio with ffmpeg `loudnorm` before transcription | disabled |
| `--autoclean` | Best-effort cleanup of temporary provider resources created during the run | disabled |
| `--analysis summary` | Add post-transcription summary analysis to the manifest | disabled |

`--analysis summary` currently requires `--provider gemini` and `--output-dir`. It runs after transcription, uses the transcript text as input, and writes a provider-neutral `analysis` object into the manifest without changing the VTT/SRT/text transcript output.

#### API resilience options

These options apply to OpenAI, Azure, Qwen file transcription, Gemini, NVIDIA Riva, and Deepgram providers where supported:

| Option | Description | Default |
|--------|-------------|---------|
| `--max-retries` | Maximum retries for retryable API failures | `5` |
| `--request-timeout-secs` | Timeout in seconds for each API request | `120` |
| `--retry-wait-base-secs` | Initial wait time used for rate limits, transport failures, and retryable server errors | `10` |
| `--retry-wait-max-secs` | Maximum wait time when parsing retry delay | `120` |

REST providers retry HTTP 429, HTTP 5xx, and transport send/stream failures when the provider implementation can safely retry. NVIDIA Riva uses the shared timeout setting for gRPC requests.

#### Segmentation options

| Option | Description | Default |
|--------|-------------|---------|
| `--segment` | Enable silence-based segmentation | disabled |
| `--silence-threshold` | Silence threshold in dB (negative) | `-40` |
| `--min-silence-duration` | Minimum silence duration in seconds | `0.8` |
| `--max-segment-secs` | Maximum segment length in seconds | `600` |
| `--segment-concurrency` | Max parallel segment requests (API providers only) | `2` |
| `--vad-model` | Path to Silero VAD ONNX model (`silero_vad.onnx`) for speech-aware segmentation | `VAD_MODEL` env var |

When using `openai`, `azure`, `qwen-filetrans`, or `nvidia-riva` providers, files exceeding the conservative 25MB auto-split threshold are automatically segmented even without `--segment`. This keeps long remote requests smaller and more reliable. Gemini and Deepgram stay whole-file by default to preserve provider/model-level speaker continuity; use `--segment` only when you want independent chunk requests, or let Gemini fall back to segmented transcription if a long whole-file request fails. When using `sherpa-onnx`, segmentation is always enabled with a maximum segment length of 30 seconds.

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

For Deepgram, `--diarize` enables provider-native diarization with `diarize_model=latest`. `--speakers N` is treated as a request to enable diarization, but no fixed speaker count is sent.

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
| `GEMINI_FILE_CACHE` | Enable Gemini Files API upload reuse | disabled |
| `GEMINI_USE_PRESIGNED_URL` | Stage Gemini input in S3-compatible storage and submit a pre-signed URL | disabled |
| `GEMINI_FILE_CACHE_INDEX` | Local Gemini file cache index path | `.cache/transcribeit/gemini-files.json` |
| `GEMINI_AUTOCLEAN` | Deprecated Gemini cleanup alias; prefer `TRANSCRIBEIT_AUTOCLEAN` / `--autoclean` | disabled |
| `GEMINI_EXPLICIT_CACHE` | Enable Gemini explicit `cachedContent` reuse | disabled |
| `GEMINI_CACHE_TTL_SECS` | Gemini explicit `cachedContent` TTL in seconds | `3600` |
| `NVIDIA_API_KEY` | NVIDIA hosted Riva API key | none |
| `NVIDIA_RIVA_FUNCTION_ID` | NVIDIA hosted Riva function id | none |
| `NVIDIA_RIVA_SERVER` | NVIDIA Riva gRPC server | `grpc.nvcf.nvidia.com:443` |
| `DEEPGRAM_API_KEY` | Deepgram API key | none |
| `DEEPGRAM_API_BASE_URL` | Deepgram API base URL | `https://api.deepgram.com/v1` |
| `DEEPGRAM_INTELLIGENCE` | Enable Deepgram summary/topics/intents/entities/sentiment | disabled |
| `DEEPGRAM_SUMMARIZE` | Enable Deepgram summarization | disabled |
| `DEEPGRAM_TOPICS` | Enable Deepgram topic detection | disabled |
| `DEEPGRAM_INTENTS` | Enable Deepgram intent recognition | disabled |
| `DEEPGRAM_DETECT_ENTITIES` | Enable Deepgram entity detection | disabled |
| `DEEPGRAM_SENTIMENT` | Enable Deepgram sentiment analysis | disabled |
| `DEEPGRAM_KEYTERM` | Comma-separated Deepgram keyterms | none |
| `DEEPGRAM_SEARCH` | Comma-separated Deepgram search terms | none |
| `DEEPGRAM_REDACT` | Comma-separated Deepgram redaction targets | none |
| `DEEPGRAM_REPLACE` | Comma-separated Deepgram find/replace rules | none |
| `DEEPGRAM_FILLER_WORDS` | Enable Deepgram filler words | disabled |
| `DEEPGRAM_NUMERALS` | Enable Deepgram numerals | disabled |
| `DEEPGRAM_USE_PRESIGNED_URL` | Stage Deepgram input in S3-compatible storage and submit a pre-signed URL | disabled |
| `AZURE_API_KEY` | Azure API key fallback for Azure provider if `--azure-api-key` is unset | none |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | none |
| `AZURE_DEPLOYMENT_NAME` | Azure deployment name | `whisper` |
| `AZURE_API_VERSION` | Azure API version | `2024-06-01` |
| `DASHSCOPE_API_KEY` | DashScope API key for Qwen providers | none |
| `DASHSCOPE_ASR_BASE_URL` | DashScope async ASR base URL for Qwen file transcription | `https://dashscope-intl.aliyuncs.com/api/v1` |
| `S3_BUCKET` | S3 bucket for remote-provider URL staging | none |
| `S3_REGION` / `AWS_REGION` | S3 region for remote-provider URL staging | none |
| `S3_ENDPOINT_URL` | S3-compatible endpoint URL | none |
| `S3_ACCESS_KEY_ID` / `AWS_ACCESS_KEY_ID` | S3 access key ID | none |
| `S3_SECRET_ACCESS_KEY` / `AWS_SECRET_ACCESS_KEY` | S3 secret access key | none |
| `S3_SESSION_TOKEN` / `AWS_SESSION_TOKEN` | S3 session token | none |
| `S3_PREFIX` | S3 object prefix for remote-provider uploads | Provider-specific if unset: `transcribeit/qwen-filetrans` for Qwen, `transcribeit/gemini` for Gemini URL mode, `transcribeit/deepgram` for Deepgram URL mode |
| `S3_PRESIGN_EXPIRES_SECS` | S3 pre-signed URL expiry in seconds | `3600` |
| `S3_FORCE_PATH_STYLE` | Force path-style URLs for S3-compatible storage | `false` |
| `VAD_MODEL` | Path to Silero VAD ONNX model for speech-aware segmentation | none |
| `DIARIZE_SEGMENTATION_MODEL` | Path to pyannote segmentation ONNX model for speaker diarization | none |
| `DIARIZE_EMBEDDING_MODEL` | Path to speaker embedding ONNX model for speaker diarization | none |
| `TRANSCRIBEIT_MAX_RETRIES` | Maximum 429 retries | `5` |
| `TRANSCRIBEIT_REQUEST_TIMEOUT_SECS` | API request timeout in seconds | `120` |
| `TRANSCRIBEIT_RETRY_WAIT_BASE_SECS` | Base retry wait time in seconds | `10` |
| `TRANSCRIBEIT_RETRY_WAIT_MAX_SECS` | Maximum retry wait time in seconds | `120` |
| `TRANSCRIBEIT_AUTOCLEAN` | Enable best-effort cleanup of temporary provider resources created during the run | disabled |

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

# Gemini with transcript summary analysis in the manifest
transcribeit run -p gemini --analysis summary \
  --remote-model gemini-3.5-flash \
  -i interview.mp4 -f vtt -o ./output

# Gemini with Files API upload reuse
transcribeit run -p gemini --gemini-file-cache \
  --remote-model gemini-3.5-flash \
  -i interview.mp4 -f vtt -o ./output

# Gemini with explicit cachedContent reuse
transcribeit run -p gemini --gemini-explicit-cache \
  --gemini-cache-ttl-secs 3600 \
  --remote-model gemini-3.5-flash \
  -i interview.mp4 -f vtt -o ./output

# Gemini using S3/R2 pre-signed URL input instead of Gemini Files API upload
transcribeit run -p gemini --gemini-use-presigned-url \
  --remote-model gemini-3.5-flash \
  -i interview.mp4 -f vtt -o ./output

# NVIDIA hosted Riva ASR
transcribeit run -p nvidia-riva -i recording.wav \
  --nvidia-api-key "$NVIDIA_API_KEY" \
  --nvidia-riva-function-id "$NVIDIA_RIVA_FUNCTION_ID" \
  --language en-US -f vtt -o ./output

# Deepgram Nova-3 ASR with provider-native diarization
transcribeit run -p deepgram --remote-model nova-3 --diarize \
  --language en -i recording.wav -f vtt -o ./output

# Deepgram Nova-3 using S3/R2 pre-signed URL input instead of direct bytes
transcribeit run -p deepgram --remote-model nova-3 --deepgram-use-presigned-url \
  --language en -i recording.wav -f vtt -o ./output

# Deepgram Nova-3 Medical with intelligence metadata and domain keyterms
transcribeit run -p deepgram --remote-model nova-3-medical \
  --diarize --deepgram-intelligence \
  --deepgram-keyterm Ofev --deepgram-keyterm Esbriet --deepgram-keyterm IPF \
  -i interview.wav -f vtt -o ./output
```

### Provider behavior

- **Local** (`-p local`) runs whisper.cpp in-process using GGML models.
- **Sherpa-ONNX** (`-p sherpa-onnx`) runs sherpa-onnx in-process. Auto-detects Whisper, Moonshine, and SenseVoice models from directory contents. Always auto-segments at 30s.
- **OpenAI-compatible** (`-p openai`) uses `--remote-model` and calls `POST {base-url}/v1/audio/transcriptions`.
  `gpt-4o-transcribe-diarize` is handled specially: the request includes `response_format=diarized_json` and `chunking_strategy=auto`, and response segments are parsed defensively so unknown or missing fields do not fail the run.
- **Azure** (`-p azure`) uses `--azure-deployment` and calls:
  `POST {base-url}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}`.
- **Qwen file transcription** (`-p qwen-filetrans`) uploads audio to S3-compatible storage, passes a pre-signed URL to DashScope, and polls the async transcription task.
- **Gemini** (`-p gemini`) uploads audio through Gemini Files API by default, calls streamed `streamGenerateContent`, and parses structured transcript JSON defensively. With `--gemini-use-presigned-url`, it stages the prepared MP3 in S3/R2 and sends the signed URL as `file_uri`; Files API cache and explicit cached content are unavailable in that mode.
- **Gemini file cache** (`--gemini-file-cache`) reuses verified Gemini Files API uploads by prepared-byte SHA-256 hash; this avoids repeated uploads and may improve implicit cache locality, but provider cache hits still depend on Gemini returning `cachedContentTokenCount`.
- **Gemini explicit cache** (`--gemini-explicit-cache`) creates/reuses Gemini `cachedContent` objects and sends `cachedContent` in the streamed generation request. Manifests report `cache.transcription.mode = "explicit"` when this path is used.
- **Analysis** (`--analysis summary`) runs a second Gemini structured JSON pass over the transcript and stores summary results in the manifest.
- **NVIDIA Riva** (`-p nvidia-riva`) sends WAV audio to hosted Riva gRPC, requesting native word timestamps and optional server-side diarization.
- **Deepgram** (`-p deepgram`) posts audio to Deepgram `/listen`, requesting smart formatting, utterances, word timestamps, and optional provider-native diarization. With `--deepgram-use-presigned-url`, it stages the prepared audio in S3/R2 and submits a JSON URL request.

For the full matrix and upload/auth notes, see: [Provider behavior](provider-behavior.md).  
For benchmark guidance and result templates, see: [Performance benchmarks](performance-benchmarks.md).

## Output files

When `--output-dir` is specified, the following files are created:

- `<input_stem>.txt` — Transcript text file (if `--output-format text`)
- `<input_stem>.vtt` — WebVTT subtitle file (if `--output-format vtt`)
- `<input_stem>.srt` — SRT subtitle file (if `--output-format srt`)
- `<input_stem>.manifest.json` — Processing manifest with metadata

### Manifest format

Manifests use `schema_version: "transcribeit.manifest.v2"`. New consumers should prefer `transcript.text`, `transcript.segments`, `capabilities`, `quality`, `cache`, optional `analysis`, and the `provider_metadata` envelope. The top-level `segments` array remains for compatibility with earlier consumers.

The `cache` object normalizes provider token-cache telemetry:

- `cache.transcription.mode` is `explicit`, `implicit`, `none`, or `unknown`.
- `cache.transcription.hit` is true when the provider reports cached input tokens.
- `cache.transcription.input_tokens` and `cache.transcription.cached_tokens` are normalized when available.
- `cache.transcription.cached_fraction` is `cached_tokens / input_tokens` when both are known and comparable. Some explicit provider cache counters can exceed request prompt counters, in which case the fraction is `null`.
- `cache.analysis` is present when a post-transcription analysis pass ran.

When `--analysis summary` is used, manifests include:

- `analysis.provider`
- `analysis.model`
- `analysis.schema_version`
- `analysis.summary.short`
- `analysis.summary.detailed`
- `analysis.summary.key_points`
- `analysis.summary.topics`
- `analysis.summary.action_items`
- `analysis.summary.questions`
- `analysis.summary.follow_ups`
- `analysis.provider_metadata.response.usage_metadata`

Example analysis object:

```json
{
  "analysis": {
    "provider": "gemini",
    "model": "gemini-3.5-flash",
    "schema_version": "transcribeit.analysis.v1",
    "summary": {
      "short": "Concise summary.",
      "detailed": "Detailed summary.",
      "key_points": [],
      "topics": [],
      "action_items": [],
      "questions": [],
      "follow_ups": []
    },
    "provider_metadata": {
      "response": {
        "streaming": true,
        "generated_json_valid": true,
        "usage_metadata": {}
      }
    }
  }
}
```

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
  "cache": {
    "transcription": {
      "provider": "qwen-filetrans",
      "mode": "none",
      "hit": false,
      "input_tokens": null,
      "cached_tokens": null,
      "cached_fraction": null,
      "source": "transcription"
    }
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
