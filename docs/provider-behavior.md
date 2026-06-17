# Provider behavior

This project supports eight providers. They share the same input/output surface, but engine type, API shape, and credentials differ.

## Remote URL input support

| Provider | Pre-signed URL support in this CLI | Notes |
|----------|------------------------------------|-------|
| `qwen-filetrans` | Required | Stages prepared MP3 audio in S3/R2 and passes the pre-signed URL to DashScope async ASR. |
| `deepgram` | Optional | Direct byte upload is the default. `--deepgram-use-presigned-url` stages prepared WAV audio in S3/R2 and sends Deepgram `{"url":"..."}`. |
| `gemini` | Optional | Files API upload/cache remains the default. `--gemini-use-presigned-url` stages prepared MP3 audio in S3/R2 and sends the signed URL as `file_uri`; this path is limited to 100 MB and excludes Gemini 2.0 family models. |
| `openai` | No | Uses multipart transcription upload. |
| `azure` | No | Uses Azure OpenAI multipart transcription upload. |
| `nvidia-riva` | No | Uses hosted Riva gRPC audio streaming/buffers. |

For URL-staging providers, `--autoclean` / `TRANSCRIBEIT_AUTOCLEAN=true` enables best-effort deletion of the S3/R2 object after the provider has consumed it. Cleanup status is written to provider metadata. Cleanup errors are warnings and do not fail an otherwise successful transcription.

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
- **Speaker diarization:** When `--diarize --speakers N` is set along with `--diarize-segmentation-model` and `--diarize-embedding-model`, speaker labels are assigned to each transcript segment after transcription. Labels appear in VTT (`<v Speaker 0>`), SRT (`[Speaker 0]`), and manifest JSON output.
- **SenseVoice limitation:** emotion and audio event detection tags are stripped by the sherpa-onnx C API and are not available in the output.
- Segment concurrency is always 1 (sequential processing).
- No external API key is required.
- The `sherpa-onnx` feature is opt-in. Build with it using `cargo build --features sherpa-onnx`.
- Requires `SHERPA_ONNX_LIB_DIR` to be set at build time when the feature is enabled (see [Architecture](architecture.md#build-requirements)).

## OpenAI-compatible (`-p openai`)

- Authentication: `--api-key` (or `OPENAI_API_KEY`).
- Base URL: `--base-url` (defaults to `https://api.openai.com`).
- Model/engine: `--remote-model` (default `whisper-1`).
- Endpoint used: `POST {base-url}/v1/audio/transcriptions`.
- Files are uploaded as 16 kHz mono MP3 by default for compatibility.
- Response handling:
  - `whisper-1` is requested as `verbose_json` first, then retried without `response_format` if the endpoint rejects it.
  - `gpt-4o-mini-transcribe` and `gpt-4o-transcribe` usually return top-level `text`, which becomes one untimed segment in the current CLI.
  - `--diarize` defaults the model to `gpt-4o-transcribe-diarize` when `--remote-model` is omitted.
- `gpt-4o-transcribe-diarize` is requested as `diarized_json` with `chunking_strategy=auto`; speaker labels and segment timestamps are mapped into the transcript model.
- When an OpenAI-compatible endpoint returns `usage`, the provider preserves it under `provider_metadata.data.response.usage`.
- OpenAI-compatible cache telemetry is normalized into `cache.transcription` from `usage.prompt_tokens_details.cached_tokens` or `usage.input_tokens_details.cached_tokens` when returned.
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
- When Azure returns `usage`, the provider preserves it under `provider_metadata.data.response.usage`.
- Azure cache telemetry is normalized into `cache.transcription` from `usage.prompt_tokens_details.cached_tokens` or `usage.input_tokens_details.cached_tokens` when returned.

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
- With `--autoclean`, the staged S3/R2 object is deleted after the Qwen result JSON is downloaded.
- Manifests include Qwen provider metadata when available:
  - `provider_metadata.provider = "qwen-filetrans"`
  - `provider_metadata.schema_version = "qwen-filetrans.metadata.v1"`
  - `provider_metadata.data.task` with task ID, request ID, timing, status, and usage
  - `provider_metadata.data.result` with audio info and transcript/sentence/word counts
  - per-segment `language`, `emotion`, and `words` with word-level timestamps
- Temporary pre-signed URLs are not persisted in the manifest; only `file_url_present` is recorded.
- S3/R2 staging cleanup, when attempted, is recorded under `provider_metadata.data.staging.cleanup`.
- Qwen file transcription does not expose token-cache telemetry through this path, so manifests use `cache.transcription.mode = "none"`.
- Qwen file transcription is intended for whole-file processing. Do not enable segmentation unless you explicitly want multiple independent remote tasks.
- If a short-audio `qwen3-asr-flash` model is accidentally selected with `-p qwen-filetrans`, the CLI validates the local file before upload and fails without staging it to S3. Short flash models have a 10 MB and 300 second limit and use a different API path.

## Gemini (`-p gemini`)

- Uses Gemini Files API plus streamed `streamGenerateContent`.
- Authentication: `--gemini-api-key` or `GEMINI_API_KEY`.
  - `--api-key`/`OPENAI_API_KEY` is accepted as a fallback for scripting consistency.
- Base URL defaults to `https://generativelanguage.googleapis.com/v1beta` and can be overridden with `--gemini-api-base-url` or `GEMINI_API_BASE_URL`.
- Default model: `gemini-3.5-flash`.
- Useful benchmark candidates include `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, and `gemini-2.5-flash`.
- By default, Gemini uses the Files API. With `--gemini-use-presigned-url` or `GEMINI_USE_PRESIGNED_URL=true`, the CLI stages the prepared MP3 in S3-compatible storage and sends the pre-signed HTTPS URL directly as `file_uri`.
- Gemini signed URL mode:
  - is intended for prepared files up to 100 MB
  - is rejected for Gemini 2.0 family models
  - cannot be combined with `--gemini-file-cache` or `--gemini-explicit-cache`
  - uses `S3_PREFIX=transcribeit/gemini` when no explicit `S3_PREFIX` is provided
  - records `provider_metadata.data.upload_method = "signed_url"` and `provider_metadata.data.request.file_url_present = true`, but does not persist the pre-signed URL
  - deletes the staged S3/R2 object after streamed generation when `--autoclean` is set
- Endpoint flow:
  - `POST {upload-base-url}/files` to start a resumable file upload.
  - Upload bytes to the returned `x-goog-upload-url`.
  - Poll `GET {base-url}/files/{id}` until the file is `ACTIVE`.
  - `POST {base-url}/models/{model}:streamGenerateContent?alt=sse`.
  - `DELETE {base-url}/files/{id}` after transcription unless `--gemini-file-cache` is enabled without `--autoclean`.
- Input audio/video is converted with FFmpeg to 16 kHz mono MP3 before upload.
- `--gemini-file-cache` stores a local index of Gemini Files API uploads keyed by SHA-256 of the prepared 16 kHz mono MP3 bytes. The CLI verifies an indexed file with `files.get` before reuse and uploads again if the file is missing, expired, failed, or mismatched.
- The default index path is `.cache/transcribeit/gemini-files.json`, or `--gemini-file-cache-index` / `GEMINI_FILE_CACHE_INDEX`.
- Gemini Files API uploads are retained by Gemini for up to 48 hours.
- `--gemini-explicit-cache` creates or reuses Gemini explicit `cachedContent` objects for the prepared audio and sends the cached-content name in the streamed generation request. This automatically enables the local Gemini cache index.
- Explicit cached content has its own TTL, controlled by `--gemini-cache-ttl-secs` / `GEMINI_CACHE_TTL_SECS` and defaulting to `3600` seconds.
- The request uses Gemini structured JSON output and asks for:
  - full transcript text
  - chronological segments
  - optional segment timestamps
  - optional speaker, language, and emotion fields
- Gemini timestamps, speakers, and emotions are generated model output, not a dedicated ASR response schema. The parser accepts future response fields, skips empty segments, joins streamed text chunks, and falls back to top-level generated text if structured JSON is missing or invalid.
- Gemini stays whole-file by default to preserve speaker continuity across the full request. Explicit `--segment` is still supported, and long whole-file failures can fall back to segmented transcription with a manifest warning because speaker identity may not be stable across independent chunks.
- `--diarize` does not change Gemini's API shape today; the provider already asks for optional speaker labels in structured output.
- `--analysis summary` runs a second structured JSON call over the transcript text and stores the result under the manifest `analysis` object. This is intentionally separate from the transcription request so the large transcript JSON response stays focused on transcription.
- Manifests include Gemini provider metadata when available:
  - `provider_metadata.provider = "gemini"`
  - `provider_metadata.schema_version = "gemini.metadata.v1"`
  - `provider_metadata.data.model`
  - `provider_metadata.data.upload_method`
  - `provider_metadata.data.response.usage_metadata`
  - `provider_metadata.data.response.finish_reasons`
  - `provider_metadata.data.file.deleted`
  - `provider_metadata.data.file.cache_enabled`
  - `provider_metadata.data.file.cache_reused`
  - `provider_metadata.data.file.cache_hash`
  - `provider_metadata.data.staging.cleanup` when signed URL S3/R2 staging is used
  - `provider_metadata.data.cached_content` when explicit cached content is used
- Gemini cache telemetry is normalized into `cache.transcription` from `usageMetadata.cachedContentTokenCount` and `usageMetadata.cacheTokensDetails` when returned.

Gemini summary analysis includes:

- `analysis.provider = "gemini"`
- `analysis.schema_version = "transcribeit.analysis.v1"`
- `analysis.summary.short`
- `analysis.summary.detailed`
- `analysis.summary.key_points`
- `analysis.summary.topics`
- `analysis.summary.action_items`
- `analysis.summary.questions`
- `analysis.summary.follow_ups`
- `analysis.provider_metadata.response.usage_metadata`
- Gemini analysis cache telemetry is normalized into `cache.analysis` when `usageMetadata` includes cache fields.

## NVIDIA Riva (`-p nvidia-riva`)

- Uses hosted NVIDIA Riva ASR over gRPC.
- Authentication:
  - `--nvidia-api-key` or `NVIDIA_API_KEY`
  - `--nvidia-riva-function-id` or `NVIDIA_RIVA_FUNCTION_ID`
  - `--api-key`/`OPENAI_API_KEY` is accepted as an API key fallback for scripting consistency.
- Server defaults to `grpc.nvcf.nvidia.com:443` and can be overridden with `--nvidia-riva-server` or `NVIDIA_RIVA_SERVER`.
- `--remote-model` is optional. When provided, it is sent as `RecognitionConfig.model`; otherwise hosted routing is driven by the function id.
- Input audio/video is converted with FFmpeg to 16 kHz mono WAV before the gRPC request.
- The request enables provider-native word timestamps and automatic punctuation. When `--diarize` is set, Riva speaker diarization is requested. `--speakers N` is used as `max_speaker_count`; when omitted, the CLI uses a default maximum of 4 speakers.
- Manifests include NVIDIA provider metadata when available:
  - `provider_metadata.provider = "nvidia-riva"`
  - `provider_metadata.schema_version = "nvidia-riva.metadata.v1"`
  - `provider_metadata.data.request_id` and `provider_metadata.data.provider_request_id`
  - `provider_metadata.data.audio` with sample rate and channel count
  - `provider_metadata.data.features` with timestamp, punctuation, and diarization flags
  - `provider_metadata.data.response` with result/word counts, elapsed time, and mean confidence
- Riva does not expose token-cache telemetry through this path, so manifests use `cache.transcription.mode = "none"`.
- The implementation targets hosted Riva gRPC. Local/self-hosted NIM REST transcription is not wired into this provider yet.

## Deepgram (`-p deepgram`)

- Uses Deepgram batch speech-to-text through `POST {deepgram-api-base-url}/listen`.
- Authentication:
  - `--deepgram-api-key` or `DEEPGRAM_API_KEY`
  - `--api-key`/`OPENAI_API_KEY` is accepted as an API key fallback for scripting consistency.
- Base URL defaults to `https://api.deepgram.com/v1` and can be overridden with `--deepgram-api-base-url` or `DEEPGRAM_API_BASE_URL`.
- Default model: `nova-3`. Use `--remote-model nova-3-medical` for Deepgram's medical-domain Nova-3 model when enabled for the account.
- The request always enables `smart_format=true` and `utterances=true` so the provider returns readable utterance segments plus word-level timestamps.
- Input audio/video is converted with FFmpeg to 16 kHz mono WAV when it is not already compatible with the internal pipeline format.
- By default, the provider uploads audio bytes directly. With `--deepgram-use-presigned-url` or `DEEPGRAM_USE_PRESIGNED_URL=true`, the CLI stages the prepared WAV in S3-compatible storage, generates a pre-signed GET URL, and sends Deepgram a JSON body containing that URL.
- Deepgram URL mode uses the shared S3 settings:
  - `S3_BUCKET`
  - `S3_REGION` or `AWS_REGION`
  - `S3_ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID`
  - `S3_SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY`
  - `S3_ENDPOINT_URL` for S3-compatible providers such as Cloudflare R2
  - `S3_SESSION_TOKEN` or `AWS_SESSION_TOKEN`
  - `S3_PREFIX` (defaults to `transcribeit/deepgram` for this mode when unset)
  - `S3_PRESIGN_EXPIRES_SECS` (defaults to `3600`, minimum `300`)
  - `S3_FORCE_PATH_STYLE=true` for providers that require path-style URLs
- With `--autoclean`, the staged S3/R2 object is deleted after the Deepgram `/listen` response is received.
- When `--diarize` is set, the request uses `diarize_model=latest`. `--speakers N` is treated as a request to enable diarization, but Deepgram does not accept a fixed speaker-count hint through this provider path.
- `--deepgram-intelligence` enables `summarize=v2`, `topics=true`, `intents=true`, `detect_entities=true`, and `sentiment=true`.
- Individual feature flags are also available:
  - `--deepgram-summarize`
  - `--deepgram-topics`
  - `--deepgram-intents`
  - `--deepgram-detect-entities`
  - `--deepgram-sentiment`
  - `--deepgram-filler-words`
  - `--deepgram-numerals`
- Custom vocabulary and downstream-processing flags:
  - `--deepgram-keyterm TERM` for Nova-3 keyterm prompting; repeat or comma-separate terms.
  - `--deepgram-search TERM` for provider search hits.
  - `--deepgram-redact TARGET` for redaction targets such as `pii`, `phi`, `pci`, `numbers`, or entity classes.
  - `--deepgram-replace FIND:REPLACE` for find-and-replace rules.
- Manifests include Deepgram provider metadata when available:
  - `provider_metadata.provider = "deepgram"`
  - `provider_metadata.schema_version = "deepgram.metadata.v1"`
  - `provider_metadata.data.request` with `audio_source` (`direct_upload` or `presigned_url`) and `file_url_present`; the actual pre-signed URL is not persisted.
  - `provider_metadata.data.staging.cleanup` when signed URL S3/R2 staging is used.
  - `provider_metadata.data.metadata` with Deepgram request id, duration, channel count, model ids, `model_info`, and intelligence token usage metadata.
  - `provider_metadata.data.response` with channel, utterance, and alternative counts, mean confidence, and timestamp clamping telemetry.
  - `provider_metadata.data.intelligence` with returned `summary`, `topics`, `intents`, `sentiments`, `entities`, `summaries`, `search`, and warnings.
- If Deepgram returns timestamps beyond its reported media duration, the provider clamps segment and word times to the duration and records `provider_metadata.data.response.timestamps_clamped = true`.
- Deepgram intelligence output is valuable for Transcript Intelligence workflows, but summaries/topics/intents/entities are still model output. Treat them as provider metadata for downstream review rather than as validated facts.
- Deepgram does not expose token-cache telemetry through this path, so manifests use `cache.transcription.mode = "none"`.

## Why providers differ

### Local vs Sherpa-ONNX

Both are local engines that run without network access. They differ in the model format and inference backend:

- **Local** uses GGML models via `whisper.cpp` (`whisper-rs` binding). Supports all Whisper model sizes. Uses FFmpeg `silencedetect` for segmentation.
- **Sherpa-ONNX** uses ONNX models via the `sherpa-onnx` C library. Supports three model architectures (Whisper, Moonshine, SenseVoice) with automatic detection. Whisper ONNX supports all sizes except `large-v3`. Requires auto-segmentation at 30s due to Whisper ONNX limitations. Supports VAD-based segmentation via `--vad-model` for cleaner speech boundaries (recommended). Also supports speaker diarization via `--diarize --speakers N`. The `sherpa-onnx` feature is optional; enable it with `cargo build --features sherpa-onnx`.

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
