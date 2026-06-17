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
time transcribeit run -p qwen-filetrans -i <input_file> -f text -o ./output
time transcribeit run -p gemini -i <input_file> -f text -o ./output
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

### 5. Qwen file transcription

Qwen filetrans is a whole-file async provider, so benchmark it separately from segmented API providers:

```bash
time transcribeit run -p qwen-filetrans -i <long_file> -f text -o ./output
time transcribeit run -p qwen-filetrans -i <long_file> -f vtt -o ./output
```

Record:
- input size and duration
- S3-compatible storage provider and region
- DashScope ASR base URL
- task `usage.seconds` from `provider_metadata.data.task`
- local wall-clock time
- manifest `provider_metadata.data.result.word_count`
- whether word-level timestamps were present

### 6. Gemini hosted transcription

Gemini is a whole-file multimodal provider with streamed response tokens and model-generated structured output, so benchmark transcript quality and timestamp reliability separately from dedicated ASR providers:

```bash
time transcribeit run -p gemini --remote-model gemini-3.5-flash -i <input_file> -f vtt -o ./output
time transcribeit run -p gemini --remote-model gemini-3.1-pro-preview -i <input_file> -f vtt -o ./output
time transcribeit run -p gemini --remote-model gemini-3.5-flash --gemini-use-presigned-url -i <input_file> -f vtt -o ./output
```

Record:
- model name
- upload method from `provider_metadata.data.upload_method`
- wall-clock time
- manifest `quality.timing_reliable`
- manifest `quality.timestamps_clamped`
- manifest `provider_metadata.data.response.usage_metadata`
- manifest `provider_metadata.data.response.streaming`
- manifest `provider_metadata.data.response.chunk_count`
- manifest `cache.transcription`
- whether speaker/language/emotion fields were useful or only generic
- whether `quality.timestamps_clamped` was triggered; clamping means Gemini generated timestamps outside the known source duration

For Gemini summary analysis, also benchmark:

```bash
time transcribeit run -p gemini --analysis summary --remote-model gemini-3.5-flash -i <input_file> -f vtt -o ./output
```

Record:
- manifest `analysis.summary.short`
- manifest `analysis.provider_metadata.response.usage_metadata`
- manifest `cache.analysis`
- whether analysis reused cached prompt tokens
- whether the summary reflects transcript caveats such as missing speaker labels or unreliable timestamps

### 7. NVIDIA hosted Riva

Benchmark hosted Riva separately from REST providers because it uses gRPC and provider-native word timestamps:

```bash
time transcribeit run -p nvidia-riva -i <input_file> -f vtt -o ./output
time transcribeit run -p nvidia-riva -i <input_file> --diarize -f vtt -o ./output
time transcribeit run -p nvidia-riva -i <input_file> --diarize --speakers 2 -f vtt -o ./output
```

Record:
- hosted function id or model name
- wall-clock time
- manifest `provider_metadata.data.response.word_count`
- manifest `provider_metadata.data.response.mean_confidence`
- manifest `quality.timing_reliable`
- whether server-side speaker labels were useful

### 8. Deepgram

Benchmark Deepgram as a whole-file batch provider with both plain Nova-3 and medical/intelligence options:

```bash
time transcribeit run -p deepgram --remote-model nova-3 \
  -i <input_file> --diarize -f vtt -o ./output

time transcribeit run -p deepgram --remote-model nova-3-medical \
  --diarize --deepgram-intelligence \
  --deepgram-keyterm Ofev --deepgram-keyterm Esbriet --deepgram-keyterm IPF \
  -i <input_file> -f vtt -o ./output
```

Record:
- model name and `provider_metadata.data.metadata.model_info`
- wall-clock time and realtime factor
- manifest `provider_metadata.data.response.mean_confidence`
- manifest `provider_metadata.data.intelligence.summary`
- counts for returned topics, intents, sentiments, and entities
- whether keyterm prompting improved domain terms or brand names
- diarization behavior, especially unexpected extra speakers
- whether `quality.timestamps_clamped` was triggered

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

## Current provider assessment

Based on the provider evaluations captured so far, Deepgram is currently the most advanced provider for Transcript Intelligence workflows, especially `nova-3-medical` with domain keyterms. It is the only tested provider that returned high-quality ASR together with provider-native utterances, word timestamps, diarization, summary, topics, intents, sentiment, entity extraction, model metadata, and intelligence token usage in one transcription response.

This does not mean every Deepgram intelligence field should be treated as ground truth. In the 5-minute medical interview sample, `nova-3-medical` returned useful entities, topics, intents, and sentiment, but its summary made a role error. Without keyterms it also misheard `Ofev` as `OFAP`; adding keyterms such as `Ofev`, `Esbriet`, `IPF`, and `Producta` corrected the medical brand terms and improved speaker consistency in the observed run.

Use this working ranking until broader benchmark data says otherwise:

| Rank | Provider / Model | Current assessment |
|---|---|---|
| 1 | Deepgram `nova-3-medical` + keyterms | Best Transcript Intelligence candidate; strongest structured metadata and good ASR when keyterms are supplied. |
| 2 | Qwen `qwen3-asr-flash-filetrans` | Strong pure ASR baseline with word timestamps, but less downstream intelligence metadata. |
| 3 | OpenAI hosted transcription | Strong general ASR, but less structured transcript intelligence in the current CLI path. |
| 4 | Gemini | Useful whole-file multimodal transcription and summary path, but timestamps/speakers are model-generated rather than dedicated ASR metadata. |
| 5 | NVIDIA Riva | Provider-native timestamps/diarization through hosted Riva, but less transcript intelligence returned through the current provider path. |

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
