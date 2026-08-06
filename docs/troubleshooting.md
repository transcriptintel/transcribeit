# Troubleshooting

## Common issues

### `ffmpeg` is not installed or not on PATH

Symptoms:
- `Error: ffmpeg not found`
- Conversion fails before transcription begins

Fix:
- Install FFmpeg and make sure the `ffmpeg` and `ffprobe` binaries are available on PATH.
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- Verify:
  - `ffmpeg -version`
  - `ffprobe -version`

### A retired Sherpa/ONNX option is rejected

The Sherpa-ONNX provider, ONNX model installer, Silero VAD path, and local
post-processing diarization were retired in TI-010. Their former CLI options and
environment variables are no longer accepted. See the
[retirement note](retired/sherpa-onnx.md) for migration choices and preserved
benchmark evidence.

For diarization, use a provider with native or model-generated speaker labels:
Deepgram, Gemini, NVIDIA Riva, or OpenAI `gpt-4o-transcribe-diarize`.

### Model download fails

Common symptoms:
- `Download failed with status: 404 Not Found`
- `Download failed with status: 403 Forbidden`
- `Failed to create directory: ...` or `Failed to start download`

Fix:
- Verify model size name (`base`, `small.en`, `large-v3`, etc.).
- Ensure network connectivity and DNS resolution.
- For GGML downloads: check `HF_TOKEN` if Hugging Face is rate-limiting your requests.
- Use `transcribeit list-models` to confirm successful downloads in `MODEL_CACHE_DIR`.
- If an integrity check fails, remove only the named managed artifact and run `download-model` or `setup` again. Do not bypass the size/SHA-256 check.

Example:

```bash
transcribeit download-model -s base
transcribeit list-models
```

### Azure authentication errors

Common errors:
- Missing or invalid API key
- Unauthorized errors from Azure endpoint

Fix:
- Provide one of:
  - `AZURE_API_KEY` in the environment or private `.env`, or
  - a private `--api-key-file` override
- `OPENAI_API_KEY` is intentionally not used for Azure or any other non-OpenAI provider.
- Ensure `--base-url` points to your Azure resource endpoint, for example:
  `https://myresource.openai.azure.com`
- Verify deployment:
  - `--azure-deployment` must match your Azure deployment name
  - `--azure-api-version` should be supported (default `2024-06-01`)
- Confirm model/version availability in your Azure deployment settings.

Example:

```bash
transcribeit run -p azure -i recording.wav \
  --base-url https://myresource.openai.azure.com \
  --azure-deployment my-whisper
```

### OpenAI-compatible endpoint rate limiting

Common symptoms:
- intermittent request failures after some files
- backoff retry logs and eventual timeout/retry exhaustion

Fix:
- Tune request resilience flags:
  - `--max-retries`
  - `--retry-wait-base-secs`
  - `--retry-wait-max-secs`
- Use smaller segments with `--segment` or lower `--max-segment-secs` for very long audio.
- Consider reducing parallelism for API providers with `--segment-concurrency`.
- If needed, lower `--request-timeout-secs`.
- Transcription POSTs retry HTTP 429 only. An ambiguous transport or 5xx failure is returned without replay because the provider may already have accepted and billed the request; retry it manually after checking provider-side state.

Example:

```bash
transcribeit run -p openai -i long.wav \
  --segment --max-segment-secs 300 \
  --segment-concurrency 1 \
  --max-retries 8 \
  --retry-wait-base-secs 12 \
  --retry-wait-max-secs 180
```

### Gemini summary analysis errors

Symptoms:
- `--analysis requires --output-dir so results can be written to the manifest`
- `--analysis is currently supported only with --provider gemini`

Fix:
- Use `--analysis summary` only with Gemini for now.
- Always provide `-o` / `--output-dir`; analysis is written into `<input_stem>.manifest.json`.

### Gemini signed URL mode fails before transcription

Symptoms:
- `--gemini-use-presigned-url` fails before or during the Gemini request
- error mentions S3 credentials, Gemini 2.0, file cache, explicit cache, or a 100 MB prepared input limit

Fix:
- Confirm `S3_BUCKET`, `S3_REGION`, `S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY` are set, plus `S3_ENDPOINT_URL` when using Cloudflare R2.
- Use a supported non-Gemini-2.0 model such as `gemini-3.6-flash` or `gemini-2.5-flash`.
- Do not combine `--gemini-use-presigned-url` with `--gemini-file-cache` or `--gemini-explicit-cache`; signed URL mode does not create reusable Gemini Files API handles.
- Keep the prepared 16 kHz mono MP3 under 100 MB, or use the default Gemini Files API path for larger or reusable files.

Example:

```bash
transcribeit run -p gemini --analysis summary \
  --remote-model gemini-3.6-flash \
  -i interview.mp4 -f vtt -o ./output
```

### Cache telemetry looks empty

Symptoms:
- `cache.transcription.hit` is `false`
- `cache.transcription.cached_tokens` is `null`
- `cache.transcription.mode` is `none`

Explanation:
- `cache` is telemetry for most providers. Gemini also supports explicit cached-content integration through `--gemini-explicit-cache`.
- Gemini and OpenAI/Azure cache hits depend on provider-side behavior and prompt length. Short audio/transcript prompts often do not produce cache hits.
- Qwen file transcription, NVIDIA Riva, and local Whisper do not expose token-cache telemetry through the current transcription paths, so their manifest cache mode is `none`.

### Gemini file cache reuses uploads but token cache still misses

Symptoms:
- `provider_metadata.data.file.cache_enabled` is `true`
- `provider_metadata.data.file.cache_reused` is `true`
- `cache.transcription.hit` is still `false`

Explanation:
- `--gemini-file-cache` reuses the Gemini Files API upload by prepared-byte SHA-256 hash. It prevents repeated upload and keeps the same Gemini `file_uri` while the Files API object exists.
- This is not the same as Gemini explicit cached content. Gemini implicit token caching can still miss even when the same file is reused.
- A missing `usage_metadata.cachedContentTokenCount` means Gemini did not report a token-cache hit for that request.

Fix:
- For upload reuse, keep `--gemini-file-cache` enabled and avoid `--gemini-autoclean` or deprecated `--autoclean`.
- For deterministic token-cache reuse, run with `--gemini-explicit-cache`. This creates or reuses a Gemini `cachedContent` object and should produce `cache.transcription.mode = "explicit"` plus `cachedContentTokenCount` when Gemini accepts the cache.
- Explicit cached content has TTL and billing behavior. Use `--gemini-cache-ttl-secs` to control how long the cache is retained by Gemini.

### Qwen file transcription rejects async calls

Symptoms:
- `Qwen ASR task query returned 403 Forbidden`
- message includes `current user api does not support asynchronous calls`

Fix:
- Use the async ASR base URL for file transcription:

```bash
DASHSCOPE_ASR_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
```

- Keep `DASHSCOPE_BASE_URL` for OpenAI-compatible chat/short ASR calls if needed. `qwen-filetrans` reads `DASHSCOPE_ASR_BASE_URL` to avoid accidentally using a compatible-mode workspace endpoint that does not support async task polling.

### Qwen file transcription cannot access audio

Symptoms:
- DashScope task fails after submit
- provider result indicates the audio URL could not be downloaded

Fix:
- Confirm S3/R2 credentials can upload objects to `S3_BUCKET`.
- Confirm the generated pre-signed GET URL is valid for the duration of the DashScope job.
- For Cloudflare R2, set:

```bash
S3_REGION=auto
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
S3_BUCKET=<bucket>
S3_ACCESS_KEY_ID=<key>
S3_SECRET_ACCESS_KEY=<secret>
```

### Wrong Qwen model selected

Symptoms:
- `qwen3-asr-flash... is a short-audio Qwen3-ASR-Flash model and is not supported by --provider qwen-filetrans`

Fix:
- Use the default `qwen3-asr-flash-filetrans` model for `-p qwen-filetrans`.
- Short `qwen3-asr-flash` models are limited to 10 MB and 300 seconds and use a different synchronous API path. The CLI rejects this mismatch before conversion and S3 upload.

### Audio format / preprocessing issues

Common symptoms:
- Wrong transcription quality for some files
- Unexpected long processing time

Fix:
- Use `--normalize` to reduce volume inconsistency from recorded content.
- Ensure input is not corrupted and ffmpeg conversion succeeds.
- For OpenAI/Azure providers, MP3 conversion is used internally; local provider uses WAV input internally. Qwen file transcription stages a prepared MP3 in S3-compatible storage and passes a pre-signed URL to DashScope. Gemini uses Gemini Files API by default, but can optionally stage prepared MP3 in S3/R2 and submit a signed URL with `--gemini-use-presigned-url`. Deepgram and NVIDIA Riva use WAV input internally; Deepgram can optionally stage that prepared WAV in S3/R2 and submit a pre-signed URL with `--deepgram-use-presigned-url`.

### Empty or tiny transcript outputs

Common causes:
- Language mismatch (auto-detection failed on very short clips)
- Excessive background noise
- Previously, a `whisper-rs` bug with `set_detect_language(true)` caused 0 segments when `--language` was not specified. This has been fixed; if you encounter this on an older build, rebuild with the latest code.

Fix:
- Provide `--language` hint (for example `--language en`).
- Use `--segment` and tune silence thresholds:
  - raise (less negative) `--silence-threshold` for more aggressive splits
  - lower `--min-silence-duration` for noisy recordings
- Try the same file with a different model (for example `base.en`, `small`, `small.en`).
