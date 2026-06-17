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

### `SHERPA_ONNX_LIB_DIR` not set / dylib not found

Symptoms:
- Build fails with linker errors referencing `sherpa-onnx` symbols
- Runtime error: `dyld: Library not loaded` (macOS) or `error while loading shared libraries` (Linux)

Fix:
- Set `SHERPA_ONNX_LIB_DIR` to the directory containing the sherpa-onnx shared libraries. This can be placed in a `.env` file in the project root or exported in your shell.

```bash
# In .env file
SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib

# Or export directly
export SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
cargo build --release
```

- The `build.rs` script reads this variable, adds it to the linker search path, and embeds an `rpath` so the binary can find the dylibs at runtime.
- If you installed sherpa-onnx via a package manager, the lib directory is typically something like `/usr/local/lib` or `/opt/homebrew/lib`.
- Verify the directory contains files like `libsherpa-onnx-core.dylib` (macOS) or `libsherpa-onnx-core.so` (Linux).

### ONNX model directory invalid

Symptoms:
- `ONNX model not found for '<name>'`
- `encoder.onnx (or encoder.int8.onnx) not found in ...`
- `Could not detect model architecture in ...`
- `tokens.txt not found in ...`

Fix:
- The sherpa-onnx engine auto-detects the model architecture. Ensure the model directory contains the correct files for one of:
  - **Whisper:** `encoder.onnx` + `decoder.onnx` (or int8 variants) + `tokens.txt`
  - **Moonshine:** `preprocess.onnx` + `encode.onnx` + `uncached_decode.onnx` + `cached_decode.onnx` + `tokens.txt`
  - **SenseVoice:** `model.onnx` + `tokens.txt`
- Download Whisper ONNX models with: `transcribeit download-model -f onnx -s <size>`
- For Moonshine and SenseVoice models, download from the [sherpa-onnx model releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) and extract into `MODEL_CACHE_DIR`.
- Verify with: `transcribeit list-models` (ONNX models appear with an `[onnx]` tag)
- The model resolver supports partial name matching (e.g., `-m moonshine-base`, `-m sense-voice`).

### VAD model not found or fails to load

Symptoms:
- `Failed to create VAD (check vad_model_path)`
- `No such file or directory` when using `--vad-model`

Fix:
- Verify that the path provided to `--vad-model` (or the `VAD_MODEL` env var) points to a valid `silero_vad.onnx` file.
- Download the Silero VAD model from the [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases). Look for `silero_vad.onnx` in the VAD model archives.
- Ensure the `sherpa-onnx` feature is enabled (it is by default). VAD-based segmentation is not available without it.
- The VAD model path can be set in your `.env` file:

```bash
# .env
VAD_MODEL=/path/to/silero_vad.onnx
```

If you do not have a VAD model, omit `--vad-model` and the pipeline will fall back to FFmpeg `silencedetect` for segmentation.

### Speaker diarization model issues

Symptoms:
- `Failed to create speaker diarization engine`
- `--speakers is required for local diarization because the current Sherpa diarizer requires a fixed cluster count`
- `--diarize-segmentation-model is required when --diarize is set`
- `--diarize-embedding-model is required when --diarize is set`
- `--diarize for provider 'qwen-filetrans' requires local Sherpa diarization`

Fix:
- When using local post-processing diarization, pass `--diarize --speakers N`; both `--diarize-segmentation-model` and `--diarize-embedding-model` are required.
- When using NVIDIA Riva, `--diarize` can be used without `--speakers`; the CLI sends a default max-speaker hint of 4.
- When using OpenAI, `--diarize` selects `gpt-4o-transcribe-diarize` by default. If you explicitly choose another OpenAI model, local Sherpa diarization is required.
- Qwen file transcription and Azure do not currently provide native diarization through this CLI; use local Sherpa diarization for those providers.
- Ensure both model paths point to valid ONNX files:
  - **Segmentation model:** a pyannote speaker segmentation ONNX model.
  - **Embedding model:** a speaker embedding extraction ONNX model.
- Download compatible models from the [sherpa-onnx speaker diarization releases](https://github.com/k2-fsa/sherpa-onnx/releases).
- The model paths can be set via environment variables in your `.env` file:

```bash
# .env
DIARIZE_SEGMENTATION_MODEL=/path/to/segmentation.onnx
DIARIZE_EMBEDDING_MODEL=/path/to/embedding.onnx
```

- Requires the `sherpa-onnx` feature to be enabled.
- The `--speakers` value must be greater than 0.

### Building without sherpa-onnx

If you do not need the sherpa-onnx provider, use the default build. It does not require the shared libraries:

```bash
cargo build --release
```

To enable the sherpa-onnx provider, install the shared libraries and build with:

```bash
cargo build --release --features sherpa-onnx
```

### Model download fails

Common symptoms:
- `Download failed with status: 404 Not Found`
- `Download failed with status: 403 Forbidden`
- `Failed to create directory: ...` or `Failed to start download`

Fix:
- Verify model size name (`base`, `small.en`, `large-v3`, etc.).
- Ensure network connectivity and DNS resolution.
- For GGML downloads: check `HF_TOKEN` if Hugging Face is rate-limiting your requests.
- For ONNX downloads: note that `large-v3` is not available in ONNX format.
- Use `transcribeit list-models` to confirm successful downloads in `MODEL_CACHE_DIR`.

Example:

```bash
transcribeit download-model -s base
transcribeit download-model -f onnx -s base.en
transcribeit list-models
```

### Azure authentication errors

Common errors:
- Missing or invalid API key
- Unauthorized errors from Azure endpoint

Fix:
- Provide one of:
  - `--azure-api-key <key>` and `AZURE_API_KEY=<key>`, or
  - `--api-key <key>` / `OPENAI_API_KEY=<key>` as fallback
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
  --azure-api-key "$AZURE_API_KEY" \
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

Example:

```bash
transcribeit run -p gemini --analysis summary \
  --remote-model gemini-3.5-flash \
  -i interview.mp4 -f vtt -o ./output
```

### Cache telemetry looks empty

Symptoms:
- `cache.transcription.hit` is `false`
- `cache.transcription.cached_tokens` is `null`
- `cache.transcription.mode` is `none`

Explanation:
- `cache` is telemetry only; the CLI does not create explicit provider caches yet.
- Gemini and OpenAI/Azure cache hits depend on provider-side behavior and prompt length. Short audio/transcript prompts often do not produce cache hits.
- Qwen file transcription, NVIDIA Riva, local Whisper, and Sherpa-ONNX do not expose token-cache telemetry through the current transcription paths, so their manifest cache mode is `none`.

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
- For OpenAI/Azure providers, MP3 conversion is used internally; local provider uses WAV input internally. Qwen file transcription stages a prepared MP3 in S3-compatible storage and passes a pre-signed URL to DashScope.

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

### SenseVoice emotion/event tags missing

SenseVoice models are capable of detecting emotions and audio events (laughter, applause, music, etc.), but the sherpa-onnx C API strips these tags from the output. Only the transcription text is available. This is a limitation of the sherpa-onnx C-level bindings, not of transcribeit.

Additionally, the SenseVoice 2025 model is a quality regression compared to the 2024 version. Prefer using the 2024 SenseVoice model for best results.

### Binary fails with "Library not loaded: libsherpa-onnx-c-api.dylib"

Symptoms:
- `dyld: Library not loaded: @rpath/libsherpa-onnx-c-api.dylib`
- Binary crashes immediately on startup

Fix: The binary expects sherpa-onnx shared libraries in a `lib/` directory next to itself:

```
transcribeit              # binary
lib/                      # create this directory
  libsherpa-onnx-c-api.dylib
  libonnxruntime.dylib
  libonnxruntime.1.23.2.dylib
```

Copy the dylibs from `vendor/sherpa-onnx-*/lib/` or download them with `transcribeit setup -c sherpa-libs`.

If you see a hardcoded path from another machine (e.g., `/Users/someone/...`), the binary was built with an old `build.rs`. Rebuild with the latest code — the portable `@executable_path/lib` rpath is now used.

To avoid this dependency entirely, use the default build:
```bash
cargo build --release
```
