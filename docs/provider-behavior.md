# Provider behavior

This project supports six providers. They share the same input/output surface, but engine type, API shape, and credentials differ.

## Local (`-p local`)

- Input audio/video is converted with FFmpeg to 16 kHz mono WAV.
- Model loading uses `--model` resolved from:
  - explicit filesystem path (for example `.cache/ggml-base.bin`)
  - or cache alias (`base`, `base.en`, `small`, `small.en`, etc.) resolved under `MODEL_CACHE_DIR`.
- Transcription happens in-process through `whisper.cpp` (`whisper-rs`).
- Outputs are produced locally and no external API key is required.

## Sherpa-ONNX (`-p sherpa-onnx`)

- Input audio/video is converted with FFmpeg to 16 kHz mono WAV.
- The engine **auto-detects model architecture** from the files in the model directory:
  - **Whisper** -- `encoder.onnx` + `decoder.onnx` + `tokens.txt`
  - **Moonshine** -- `preprocess.onnx` + `encode.onnx` + `uncached_decode.onnx` + `cached_decode.onnx` + `tokens.txt`
  - **SenseVoice** -- `model.onnx` + `tokens.txt`
- Model loading uses `--model` resolved from:
  - explicit filesystem path to a model directory
  - cache alias (`tiny`, `base.en`, `small`, etc.) resolved under `MODEL_CACHE_DIR` as `sherpa-onnx-whisper-<alias>/`
  - or glob-based partial name matching (e.g., `-m moonshine-base`, `-m sense-voice`) against directories in `MODEL_CACHE_DIR`.
- The engine prefers `int8` quantized ONNX files when available for lower memory usage.
- Transcription runs in-process on a dedicated worker thread using the sherpa-onnx C library via FFI.
- C++ stderr warnings from the sherpa-onnx library are suppressed during inference to keep terminal output clean.
- Whisper ONNX models only support audio of 30 seconds or less per call. The pipeline automatically enables segmentation and caps `--max-segment-secs` at 30, regardless of user-supplied values.
- **VAD-based segmentation:** When `--vad-model` is set (or `VAD_MODEL` env var), Silero VAD is used for speech-aware segmentation instead of FFmpeg `silencedetect`. This detects actual speech boundaries and avoids mid-word cuts. The VAD pipeline pads chunks by 250ms, merges gaps shorter than 200ms, and splits long chunks at low-energy points. This is the recommended segmentation method for sherpa-onnx. When no VAD model is provided, the pipeline falls back to FFmpeg `silencedetect`.
- **Speaker diarization:** When `--speakers N` is set along with `--diarize-segmentation-model` and `--diarize-embedding-model`, speaker labels are assigned to each transcript segment after transcription. Labels appear in VTT (`<v Speaker 0>`), SRT (`[Speaker 0]`), and manifest JSON output.
- **SenseVoice limitation:** emotion and audio event detection tags are stripped by the sherpa-onnx C API and are not available in the output.
- Segment concurrency is always 1 (sequential processing).
- No external API key is required.
- The `sherpa-onnx` feature is enabled by default. Build without it using `cargo build --no-default-features`.
- Requires `SHERPA_ONNX_LIB_DIR` to be set at build time (see [Architecture](architecture.md#build-requirements)).

## OpenAI-compatible (`-p openai`)

- Authentication: `--api-key` (or `OPENAI_API_KEY`).
- Base URL: `--base-url` (defaults to `https://api.openai.com`).
- Model/engine: `--remote-model` (default `whisper-1`).
- Endpoint used: `POST {base-url}/v1/audio/transcriptions`.
- Files are uploaded as 16 kHz mono MP3 by default for compatibility.
- Response handling:
  - `whisper-1` is requested as `verbose_json` first, then retried without `response_format` if the endpoint rejects it.
  - `gpt-4o-mini-transcribe` and `gpt-4o-transcribe` usually return top-level `text`, which becomes one untimed segment in the current CLI.
  - `gpt-4o-transcribe-diarize` is requested as `diarized_json` with `chunking_strategy=auto`; speaker labels and segment timestamps are mapped into the transcript model.
  - JSON responses are parsed defensively. Unknown fields are ignored, missing segment timestamps default to `0`, and invalid/empty segments fall back to top-level `text` when available.
- Supports API resilience options:
  - `--max-retries`
  - `--request-timeout-secs`
  - `--retry-wait-base-secs`
  - `--retry-wait-max-secs`

## Azure (`-p azure`)

- Authentication:
  - `--azure-api-key` (or `AZURE_API_KEY`)
  - `--api-key`/`OPENAI_API_KEY` is used as a fallback.
- Base URL: `--base-url` (or `AZURE_OPENAI_ENDPOINT`).
- Deployment:
  - `--azure-deployment` identifies the deployment name.
  - API version from `--azure-api-version` (default `2024-06-01`).
- Endpoint used:
  - `{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version={version}`
- Uses model name from deployment; `--remote-model` is ignored for Azure.
- Uploads the prepared 16 kHz mono MP3 when possible.

## Qwen file transcription (`-p qwen-filetrans`)

- Uses Alibaba DashScope `qwen3-asr-flash-filetrans` by default.
- Endpoint used:
  - `POST {qwen-api-base-url}/services/audio/asr/transcription`
  - `GET {qwen-api-base-url}/tasks/{task_id}`
- Base URL defaults to `https://dashscope-intl.aliyuncs.com/api/v1` and can be overridden with `--qwen-api-base-url` or `DASHSCOPE_ASR_BASE_URL`.
- Authentication:
  - `--dashscope-api-key` or `DASHSCOPE_API_KEY`
  - `--api-key`/`OPENAI_API_KEY` is used as a fallback.
- The provider stages audio in S3-compatible object storage because Qwen file transcription only accepts publicly reachable `input.file_url` values.
- Required storage configuration:
  - `S3_BUCKET`
  - `S3_REGION` or `AWS_REGION`
  - `S3_ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID`
  - `S3_SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY`
- Optional storage configuration:
  - `S3_ENDPOINT_URL` for S3-compatible providers
  - `S3_SESSION_TOKEN` or `AWS_SESSION_TOKEN`
  - `S3_PREFIX` (defaults to `transcribeit/qwen-filetrans`)
  - `S3_PRESIGN_EXPIRES_SECS` (defaults to `3600`, minimum `300`)
  - `S3_FORCE_PATH_STYLE=true` for providers that require path-style URLs
- Input audio/video is converted with FFmpeg to 16 kHz mono MP3 before upload.
- The engine uploads the prepared file, generates a pre-signed GET URL, submits the Qwen async task, polls until completion, downloads the transcription JSON, and maps Qwen sentence timestamps into the project transcript model.
- Manifests include Qwen provider metadata when available:
  - `provider_metadata.provider = "qwen-filetrans"`
  - `provider_metadata.schema_version = "qwen-filetrans.metadata.v1"`
  - `provider_metadata.data.task` with task ID, request ID, timing, status, and usage
  - `provider_metadata.data.result` with audio info and transcript/sentence/word counts
  - per-segment `language`, `emotion`, and `words` with word-level timestamps
- Temporary pre-signed URLs are not persisted in the manifest; only `file_url_present` is recorded.
- Qwen file transcription is intended for whole-file processing. Do not enable segmentation unless you explicitly want multiple independent remote tasks.
- If a short-audio `qwen3-asr-flash` model is accidentally selected with `-p qwen-filetrans`, the CLI validates the local file before upload and fails without staging it to S3. Short flash models have a 10 MB and 300 second limit and use a different API path.

## Gemini (`-p gemini`)

- Uses Gemini Files API plus `generateContent`.
- Authentication: `--gemini-api-key` or `GEMINI_API_KEY`.
  - `--api-key`/`OPENAI_API_KEY` is accepted as a fallback for scripting consistency.
- Base URL defaults to `https://generativelanguage.googleapis.com/v1beta` and can be overridden with `--gemini-api-base-url` or `GEMINI_API_BASE_URL`.
- Default model: `gemini-3.5-flash`.
- Useful benchmark candidates include `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, and `gemini-2.5-flash`.
- Endpoint flow:
  - `POST {upload-base-url}/files` to start a resumable file upload.
  - Upload bytes to the returned `x-goog-upload-url`.
  - Poll `GET {base-url}/files/{id}` until the file is `ACTIVE`.
  - `POST {base-url}/models/{model}:generateContent`.
  - `DELETE {base-url}/files/{id}` after transcription.
- Input audio/video is converted with FFmpeg to 16 kHz mono MP3 before upload.
- The request uses Gemini structured JSON output and asks for:
  - full transcript text
  - chronological segments
  - optional segment timestamps
  - optional speaker, language, and emotion fields
- Gemini timestamps, speakers, and emotions are generated model output, not a dedicated ASR response schema. The parser accepts future response fields, skips empty segments, and falls back to top-level generated text if structured JSON is missing or invalid.
- Manifests include Gemini provider metadata when available:
  - `provider_metadata.provider = "gemini"`
  - `provider_metadata.schema_version = "gemini.metadata.v1"`
  - `provider_metadata.data.model`
  - `provider_metadata.data.upload_method`
  - `provider_metadata.data.response.usage_metadata`
  - `provider_metadata.data.response.finish_reasons`
  - `provider_metadata.data.file.deleted`

## Why providers differ

### Local vs Sherpa-ONNX

Both are local engines that run without network access. They differ in the model format and inference backend:

- **Local** uses GGML models via `whisper.cpp` (`whisper-rs` binding). Supports all Whisper model sizes. Uses FFmpeg `silencedetect` for segmentation.
- **Sherpa-ONNX** uses ONNX models via the `sherpa-onnx` C library. Supports three model architectures (Whisper, Moonshine, SenseVoice) with automatic detection. Whisper ONNX supports all sizes except `large-v3`. Requires auto-segmentation at 30s due to Whisper ONNX limitations. Supports VAD-based segmentation via `--vad-model` for cleaner speech boundaries (recommended). Also supports speaker diarization via `--speakers`. The `sherpa-onnx` feature is optional (enabled by default); build without it using `cargo build --no-default-features`.

### Segmentation: VAD vs FFmpeg silencedetect

| | VAD (Silero) | FFmpeg silencedetect |
|---|---|---|
| **Availability** | Requires `sherpa-onnx` feature + `--vad-model` | Always available |
| **Boundary quality** | Speech-aware; avoids mid-word cuts | Silence-based; may cut mid-word |
| **Approach** | Detects speech regions, pads, merges, splits at low-energy | Detects silence gaps, splits at midpoints |
| **Config flags** | `--vad-model`, `--max-segment-secs` | `--silence-threshold`, `--min-silence-duration`, `--max-segment-secs` |
| **Best for** | Local sherpa-onnx transcription | API providers, or when no VAD model is available |

### OpenAI vs Azure

- OpenAI-style providers use a model parameter (`--remote-model`) in the transcriptions request body.
- Azure uses a deployment resource in the request path and requires deployment-specific routing.
- That is why Azure command lines require `--azure-deployment` while OpenAI requires `--remote-model`.
