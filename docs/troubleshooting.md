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

### Building without sherpa-onnx

If you do not need the sherpa-onnx provider and want to avoid installing the shared libraries:

```bash
cargo build --release --no-default-features
```

This disables the `sherpa-onnx` Cargo feature (which is enabled by default) and removes the dependency on `SHERPA_ONNX_LIB_DIR`.

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

### Audio format / preprocessing issues

Common symptoms:
- Wrong transcription quality for some files
- Unexpected long processing time

Fix:
- Use `--normalize` to reduce volume inconsistency from recorded content.
- Ensure input is not corrupted and ffmpeg conversion succeeds.
- For remote providers, MP3 conversion is used internally; local provider uses WAV input internally.

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
