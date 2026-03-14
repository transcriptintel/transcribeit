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

### Model download fails

Common symptoms:
- `Download failed with status: 404 Not Found`
- `Download failed with status: 403 Forbidden`
- `Failed to create directory: ...` or `Failed to start download`

Fix:
- Verify model size name (`base`, `small.en`, `large-v3`, etc.).
- Ensure network connectivity and DNS resolution.
- Check `HF_TOKEN` if Hugging Face is rate-limiting your requests.
- Use `transcribeit list-models` to confirm successful downloads in `MODEL_CACHE_DIR`.

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

Fix:
- Provide `--language` hint (for example `--language en`).
- Use `--segment` and tune silence thresholds:
  - raise (less negative) `--silence-threshold` for more aggressive splits
  - lower `--min-silence-duration` for noisy recordings
- Try the same file with a different model (for example `base.en`, `small`, `small.en`).
