# Performance benchmarks

This guide keeps benchmark expectations explicit and reproducible so future changes can be measured consistently.

## Measurement environment

Capture these details for every benchmark run:

- Host CPU model and core count
- RAM
- OS and kernel
- `rustc` version
- Provider and exact command used
- Model used (for local/sherpa-onnx) or deployment/model (for APIs)
- Model format (GGML or ONNX) for local providers
- Input file duration and codec/container

## Benchmarks to run

### 1. Local model inference throughput

Run on representative files (e.g. 1, 5, 10 minutes).

```bash
# whisper.cpp (GGML)
time transcribeit run -i <input_file> -m base -f text -o ./output
time transcribeit run -i <input_file> -m small -f text -o ./output
time transcribeit run -i <input_file> -m small.en -f text -o ./output

# sherpa-onnx Whisper (ONNX) — auto-segments at 30s
time transcribeit run -p sherpa-onnx -i <input_file> -m base -f text -o ./output
time transcribeit run -p sherpa-onnx -i <input_file> -m small.en -f text -o ./output

# sherpa-onnx Moonshine
time transcribeit run -p sherpa-onnx -i <input_file> -m moonshine-base -f text -o ./output

# sherpa-onnx SenseVoice
time transcribeit run -p sherpa-onnx -i <input_file> -m sense-voice -f text -o ./output
```

Record:
- wall clock duration
- output length/time ratio (e.g. 600s audio in 180s)
- CPU utilization profile (optional)

### 2. Provider overhead comparison (same input)

```bash
time transcribeit run -p local -i <input_file> -m base -f text -o ./output
time transcribeit run -p sherpa-onnx -i <input_file> -m base -f text -o ./output
time transcribeit run -p openai -i <input_file> -f text -o ./output
time transcribeit run -p azure -i <input_file> -f text -o ./output
```

Record:
- wall clock duration
- retry counts (if any)
- segment count
- average segment latency (from logs)

### 3. Segmentation impact

```bash
# FFmpeg silencedetect segmentation
time transcribeit run -p openai -i <long_file> --segment --segment-concurrency 2 -f text -o ./output
time transcribeit run -p openai -i <long_file> --segment --segment-concurrency 1 --max-segment-secs 300 -f text -o ./output

# sherpa-onnx with FFmpeg silencedetect (default, always segments at 30s max)
time transcribeit run -p sherpa-onnx -i <long_file> -m base -f text -o ./output

# sherpa-onnx with VAD-based segmentation
time transcribeit run -p sherpa-onnx -i <long_file> -m base --vad-model /path/to/silero_vad.onnx -f text -o ./output
```

Record:
- total segment count
- max queue wait
- request-level retry counts
- segmentation method used (VAD vs silencedetect)
- transcript quality at segment boundaries (check for mid-word cuts)

### 4. I/O + conversion overhead

```bash
time transcribeit run -i <video_file> -m base --normalize -f text -o ./output
time transcribeit run -i <audio_file> -m base --normalize -f text -o ./output
```

Record:
- conversion wall time (video vs audio)
- post-conversion processing time ratio

## Suggested result format

```text
Model/Provider: base
Input: meeting_01.wav (300s)
Machine: MacBook Pro M2
Elapsed: 92s
Realtime factor: 3.26x
Segments: 1
Retries: 0
Output size: 4.6 MB
```

Keep rows in a simple table (date + commit hash + environment + results) in your preferred tracker so regressions are easy to catch.

## Reference benchmark results

These results were measured on a 5-minute medical interview recording.

| Engine / Model | Wall clock | Realtime factor | Notes |
|---|---|---|---|
| Local whisper.cpp `base` | 3.6s | 83x RT | Best speed/quality trade-off |
| SenseVoice 2024 | 6.6s | 46x RT | Good quality, 50+ languages |
| Sherpa-ONNX Whisper `base` | 10.9s | 27x RT | |
| Moonshine `base` | 14.1s | 21x RT | |
| Local whisper.cpp `large-v3-turbo` | 33.7s | 8.9x RT | Highest transcription quality |
| Sherpa-ONNX Whisper `turbo` | 47.2s | 6.4x RT | |

**Notes:**
- Local whisper.cpp (GGML) is consistently the fastest engine for a given model size.
- SenseVoice 2024 offers excellent speed with good quality. **Avoid the SenseVoice 2025 model** -- it is a regression in quality.
- Moonshine provides a compact alternative but is slower than Whisper at the same size tier.
- For highest quality where speed is not critical, use `large-v3-turbo` with local whisper.cpp.

### VAD vs FFmpeg silencedetect segmentation

VAD-based segmentation (Silero VAD via `--vad-model`) and FFmpeg `silencedetect` produce different segment boundaries. Key differences to observe when benchmarking:

- **Segment boundary quality:** VAD detects speech regions directly, so segment boundaries align with actual speech. FFmpeg `silencedetect` splits at silence midpoints, which can cut mid-word if silence gaps are short or thresholds are mistuned.
- **Segment count:** VAD typically produces more segments (one per speech region after merging) while `silencedetect` produces fewer, longer segments based on silence gaps.
- **Processing overhead:** VAD runs on the audio samples in-memory (fast, no subprocess). FFmpeg `silencedetect` runs as a subprocess and requires parsing its stderr output.
- **Transcript quality:** VAD-segmented transcripts tend to have fewer artifacts at segment boundaries because chunks start and end at speech boundaries with 250ms padding, rather than at arbitrary silence midpoints.

When comparing, use the same audio file and model to isolate the effect of the segmentation method on overall transcript quality and timing.

## CI/automatable baseline

For now, treat these as manual benchmarks in a fixed environment.
If you want to automate later:
- add a dedicated `criterion` benchmark target,
- pin fixture files,
- and fail CI only on large regressions with generous tolerances.
