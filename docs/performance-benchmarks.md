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

## Representative quality corpus

Publishable quality comparisons use
[`transcribeit-representative-v1`](../benchmarks/corpus/README.md), not the
private medical interview retained for historical observations below. The
tracked manifest covers clean US English, deterministic 10 dB noisy US English,
Mandarin-L1 English, and a 40-minute four-person AMI meeting with manual word,
segment, speaker, and overlap annotations.

```bash
bun run scripts/corpus.ts fetch
bun run scripts/corpus.ts verify
```

Media and expanded annotations stay in ignored `samples/corpus/v1/`. The fetcher
verifies pinned source and output hashes and removes temporary downloads by
default. Apply the separate metric definitions in
[`scoring.yaml`](../benchmarks/corpus/v1/scoring.yaml); do not collapse word
accuracy, domain terms, timing, speakers, metadata, latency, and provider
failures into one score.

### Clean regression baseline (2026-08-06)

TI-007 established the first clean-commit baseline for the representative
corpus. The transcript-free record is
[`2026-08-06-ti-007-clean-baseline.sanitized.json`](../benchmarks/results/2026-08-06-ti-007-clean-baseline.sanitized.json).
It contains 110 attempts across 11 provider/model paths: three runs on each
short fixture and one run on the 2,423.68-second AMI meeting. Of those attempts,
109 produced transcripts.

| Provider / model | Short macro WER | AMI WER | AMI wall / RTF | AMI timing or speakers |
|---|---:|---:|---:|---|
| Azure `whisper` deployment | 11.76% | 26.76% | 149.90s / 0.0618 | Start/end MAE 557/691ms; 73.9% reference coverage |
| Deepgram `nova-3` | 17.65% | **18.45%** | **30.17s / 0.0124** | Start/end MAE 185/407ms; DER 25.61%, attributed WER 36.96% |
| Gemini `gemini-3.6-flash` | 11.76% | 19.55% | 490.67s / 0.2024 | No scoreable provider-native timing or speakers |
| NVIDIA hosted Riva | **5.88%** | 28.99% | 104.63s / 0.0432 | Timing unreliable; DER 88.98%, attributed WER 86.69% |
| OpenAI `gpt-transcribe` | 11.76% | 22.50% | 108.50s / 0.0448 | Text only; no scoreable timing or speakers |
| OpenAI `gpt-4o-transcribe-diarize` | 17.65% | N/A | Rejected | AMI exceeded the returned 1,400-second input limit |
| Qwen FileTrans | 11.76% | 21.08% | 108.79s / 0.0449 | Timing unreliable; 79.5% reference coverage |
| Local whisper.cpp `base` | 25.49% | 31.59% | 33.69s / 0.0139 | Start/end MAE 946/1,005ms; 69.4% reference coverage |
| Sherpa-ONNX Whisper `base` + VAD | 21.57% | 32.12% | 121.47s / 0.0501 | No scoreable output timing or speakers |
| Sherpa-ONNX Qwen3-ASR 0.6B int8 + VAD | **5.88%** | 22.71% | 431.10s / 0.1779 | Text only; no scoreable timing or speakers |
| llama.cpp Qwen3-ASR 1.7B | 46.50% | 102.26% | 523.75s / 0.2161 | Long-form output failed the quality check despite request success |

Deepgram had the lowest observed AMI WER and the fastest successful AMI request;
local whisper.cpp was the fastest local AMI path. These are baseline
observations, not a universal provider ranking: the long fixture has one run,
hosted aliases can change, and timing/diarization capabilities differ.

`gpt-4o-transcribe-diarize` completed all nine short runs using
`diarized_json`, but the AMI request failed before inference because of the
returned duration limit. A separate compatibility probe confirmed that
`gpt-transcribe` rejects `response_format=diarized_json` with HTTP 400
`unsupported_value`; use the diarization model for supported speaker-labeled
output within its limit.

The clean run was produced at commit `64afbc29cffc65b654ad7a7eaf37dbcbad279765`
on an Apple M4 Pro host. Model artifacts were hashed before cleanup; four new
evaluation-owned directories were then removed, reclaiming 4,239,458,304 bytes
and verifying that the evaluation root was absent. Gemini and Qwen remote
staging cleanup is recorded per successful request in the sanitized result.

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
time transcribeit run -p openai --remote-model gpt-transcribe -i <input_file> -f text -o ./output
time transcribeit run -p openai --remote-model whisper-1 -i <input_file> -f text -o ./output
time transcribeit run -p azure -i <input_file> -f text -o ./output
time transcribeit run -p qwen-filetrans -i <input_file> -f text -o ./output
time transcribeit run -p gemini -i <input_file> -f text -o ./output
time transcribeit run -p nvidia-riva -i <input_file> -f text -o ./output
time transcribeit run -p deepgram -i <input_file> -f text -o ./output
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

# sherpa-onnx with FFmpeg silencedetect (30s max; Qwen3-ASR uses 20s)
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

#### Local Qwen3-ASR through llama.cpp

This is a compatibility experiment, not a dedicated TranscribeIt provider. Start a
separately managed `llama-server`, then use the existing OpenAI-compatible path:

```bash
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF \
  --port 18080 --no-ui --log-disable -n 4096 --temp 0

time transcribeit run -p openai --api-key local \
  --base-url http://127.0.0.1:18080 \
  --remote-model ggml-org/Qwen3-ASR-1.7B-GGUF \
  -i <input_file> -f text -o ./output
```

The 2026-06-09 smoke test processed the 300.01-second medical interview in about
18 seconds (approximately 0.060 RTF) after the 1.7B server and model were loaded.
It produced readable text and recognized domain terms such as `Ofev` and `Esbriet`,
but returned one plain-text segment without timestamps, word alignment, or provider
metadata. llama.cpp audio support identified itself as experimental. Treat this as
a promising warm-server observation, not a reproducible baseline or a replacement
for DashScope `qwen-filetrans`.

For a repeatable run, additionally record the llama.cpp build, exact GGUF revision
and hash, model load time, warm/cold state, server command, and server logs.

### 6. Gemini hosted transcription

Gemini is a whole-file multimodal provider with streamed response tokens and model-generated structured output, so benchmark transcript quality and timestamp reliability separately from dedicated ASR providers:

```bash
time transcribeit run -p gemini --remote-model gemini-3.6-flash -i <input_file> -f vtt -o ./output
time transcribeit run -p gemini --remote-model gemini-3.1-pro-preview -i <input_file> -f vtt -o ./output
time transcribeit run -p gemini --remote-model gemini-3.6-flash --gemini-use-presigned-url -i <input_file> -f vtt -o ./output
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
time transcribeit run -p gemini --analysis summary --remote-model gemini-3.6-flash -i <input_file> -f vtt -o ./output
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

## OpenAI `gpt-transcribe` update (2026-08-06)

OpenAI now recommends [`gpt-transcribe`](https://developers.openai.com/api/docs/models/gpt-transcribe)
as the starting model for completed recordings. It returns transcript text and
detected languages. Continue using `whisper-1` for word/segment timestamps or
subtitle formats and `gpt-4o-transcribe-diarize` for speaker labels, as described
in the [current transcription guide](https://developers.openai.com/api/docs/guides/transcription).

The five-minute medical-interview fixture was tested in one unhinted run and two
English-hinted runs through the TranscribeIt OpenAI provider, plus one direct API
compatibility probe:

| Variant | Wall time | Processing time | Processing RTF | Output | Domain-term result |
|---|---:|---:|---:|---|---|
| Unhinted | 13.10s | 11.89s | 0.040 | One untimed text segment | Preserved 7/7 audible tracked terms |
| `--language en` (2 runs) | 9.32–10.38s | 8.22–9.32s | 0.027–0.031 | One untimed text segment; detected `en` retained | Preserved the same 7/7 terms |
| `response_format=diarized_json` compatibility probe | 1.53s | API rejected after 0.76s server processing | N/A | HTTP 400; no transcript | Incompatible with `gpt-transcribe` |

All three successful transcription runs used the 300.010688-second, 1,960,635-byte
MP3 fixture with SHA-256
`14ff4dc83b8788bb9052aea72ec3f45f7b112581ba9a13a67baaabc844526e5a` on an
Apple M4 Pro with 24 GiB RAM. Minor wording varied between runs. The two hinted
runs were consistently fast, but matched repetitions of the unhinted and older
model variants are still required before interpreting the language hint as a
model-level latency improvement.
The model returned no timestamps or speakers, which is expected for this model role.

The explicit `diarized_json` probe included `chunking_strategy=auto`, but the API
rejected `response_format` as incompatible with the resolved
`gpt-transcribe-api-ev3` model. This is a compatibility result, not a successful
latency or quality measurement. Use `gpt-4o-transcribe-diarize` when speaker,
start, and end metadata are required.

The adapter sends `--language` as `languages[]` for `gpt-transcribe` and preserves
detected languages in provider metadata. Future capability tests should evaluate
`prompt`, `keywords[]`, multiple expected languages, and streaming separately from
the unhinted quality/latency comparison described in the
[file-transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text).

## TI-007 provider reference rebaseline (2026-08-06)

The configured hosted/local matrix was repeated three times on the same
300.010688-second MP3. The complete sanitized record, including commands,
run-level hashes, capabilities, failures, artifact revisions, and cleanup evidence,
is [`2026-08-06-ti-007-provider-rebaseline.sanitized.json`](../benchmarks/results/2026-08-06-ti-007-provider-rebaseline.sanitized.json).

| Provider / model | Median wall | Median processing RTF | Output used for latency | Reviewed term presence | Capability boundary |
|---|---:|---:|---|---:|---|
| Local whisper.cpp `base` | 3.99s | 0.0132 | VTT | 4/7 | Native segment timing; no speakers or words |
| Deepgram `nova-3-medical` | 7.49s | 0.0242 | VTT | 7/7 | Native words, timing, speakers, and intelligence metadata |
| NVIDIA hosted Riva | 9.66s | 0.0316 | Text | 5/7 | Native words and speakers; one zero-duration segment prevented VTT |
| OpenAI `gpt-transcribe` | 10.45s | 0.0347 | Text | 7/7 | Detected language, but no timestamps or speakers |
| llama.cpp Qwen3-ASR 1.7B | 10.82s | 0.0360 | Text | 7/7 | Warm local server; text only; audio path marked experimental |
| Azure `whisper` deployment | 11.18s | 0.0370 | VTT | 5/7 | Native segment timing; exact underlying deployment revision unavailable |
| Qwen FileTrans | 11.92s | 0.0387 | Text | 7/7 | Native word metadata; zero-duration segments prevented VTT |
| Sherpa-ONNX Whisper `base` + VAD | 13.70s | 0.0446 | Text | 2/7 | 34 untimed model segments; VTT unsupported for this result |
| Gemini `gemini-3.5-flash` | 48.51s | 0.1615 | Text | 7/7 | Model-generated, non-monotonic timing/speakers; VTT unsupported in observed runs |
| OpenAI `gpt-4o-transcribe-diarize` | 110.94s | 0.3695 | VTT | 6/7 | Native segment timing and speakers through `diarized_json` |

Term presence checks only whether seven audible medical/domain phrases appeared
in run 1. It is not WER, DER, semantic accuracy, or a ranking. The worktree was
dirty and the single sensitive fixture does not satisfy TI-006, so none of these
rows is a regression baseline. The 2026-06 table below remains historical context.

Observed output failures were kept as results. `gpt-transcribe` rejected direct
`diarized_json` with HTTP 400, while `gpt-4o-transcribe-diarize` succeeded with
that format. Strict VTT serialization also rejected zero-duration segments from
`gpt-transcribe`, Qwen FileTrans, Gemini, NVIDIA Riva, and Sherpa; successful text
runs were measured separately instead of silently rewriting timestamps.

Downloadable-model evaluations used new, narrow directories and removed them
after hashing. Across Whisper GGML, three Sherpa download cycles (including two
failure paths), and the llama.cpp Qwen cache, 4,278,595,584 bytes were removed and
each exact evaluation path was verified absent. Qwen staged objects and Gemini
Files API uploads were also deleted in all three successful hosted runs.

### Representative-corpus reference (2026-08-06)

The ten-provider matrix was subsequently run against
[`transcribeit-representative-v1`](../benchmarks/corpus/README.md). Each short
fixture has three repetitions; the 2,423.68-second AMI meeting has one. The
sanitized record is
[`2026-08-06-ti-007-representative-corpus.sanitized.json`](../benchmarks/results/2026-08-06-ti-007-representative-corpus.sanitized.json).

| Provider / model | Short macro WER | Noisy WER | AMI WER | AMI wall / RTF | AMI timing or speakers |
|---|---:|---:|---:|---:|---|
| Azure `whisper` deployment | 11.76% | 17.65% | 26.76% | 148.48s / 0.0613 | Start/end MAE 557/691ms; 73.9% reference coverage |
| Deepgram `nova-3` | 17.65% | 29.41% | **18.45%** | 35.03s / 0.0145 | 184/405ms; DER 25.60%, attributed WER 36.96%, 5 speakers vs 4 |
| Gemini `gemini-3.5-flash` | 11.76% | 17.65% | 38.26% | 364.06s / 0.1502 | No scoreable provider-native timing or speakers |
| NVIDIA hosted Riva | **5.88%** | 17.65% | 29.90% | 66.35s / 0.0274 | 228/602ms; DER 60.90%, attributed WER 68.56%, 3 speakers vs 4 |
| OpenAI `gpt-transcribe` | 11.76% | 17.65% | 22.57% | 62.70s / 0.0259 | Text only; no scoreable timing or speakers |
| OpenAI `gpt-4o-transcribe-diarize` | 13.73% | 23.53% | N/A | Rejected | AMI exceeded the returned 1,400-second input limit |
| Qwen FileTrans | 11.76% | 17.65% | 21.19% | 89.14s / 0.0368 | 421/481ms; 79.4% reference coverage |
| Local whisper.cpp `base` | 25.49% | 52.94% | 31.59% | **34.99s / 0.0144** | 946/1,005ms; 69.4% reference coverage |
| Sherpa-ONNX Whisper `base` + VAD | 21.57% | 41.18% | 32.43% | 118.63s / 0.0489 | No scoreable output timing or speakers |
| llama.cpp Qwen3-ASR 1.7B | 46.50% | 41.18% | 102.26% | 553.16s / 0.2282 | Long-form output failed the quality check despite request success |

Short macro WER is the macro-average of each provider's median WER across the
clean FLEURS, 10 dB noisy FLEURS, and Mandarin-L1 English fixtures. The AMI WER
time-orders overlapping reference words; it is not overlap-invariant cpWER.
Timing scores use exact-token alignment, and DER uses 10 ms frames, a 250 ms
collar, overlap scoring, and optimal one-to-one speaker mapping.

This record contains 100 attempts: 99 successful transcriptions and one explicit
duration-limit rejection. A separate compatibility probe again found that
`gpt-transcribe` rejects `response_format=diarized_json`; use
`gpt-4o-transcribe-diarize` for supported diarized output within its input limit.
Azure completed all attempts but recorded three retry events. These results came
from a dirty worktree, and the long fixture has one repetition, so use them as a
reference rather than a regression baseline or definitive provider ranking.

Local-model evaluation directories were newly created and removed after the
run, reclaiming 3,228,049,408 bytes. All ten Gemini and Qwen remote staging
cleanup attempts succeeded. Raw transcripts remain ignored; the tracked record
contains metrics, hashes, capabilities, failure classifications, and cleanup
evidence only.

### Native Qwen3-ASR ONNX evaluation (TI-004, 2026-08-06)

The maintained Sherpa export `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`
was evaluated in-process with `sherpa-onnx` 1.13.4 and ONNX Runtime 1.27.0.
Each short run used a fresh process; the AMI row is one long-form run. This is a
dirty-worktree reference, not a regression baseline. The sanitized record is
[`2026-08-06-ti-004-qwen3-asr-onnx.sanitized.json`](../benchmarks/results/2026-08-06-ti-004-qwen3-asr-onnx.sanitized.json).

| Fixture | Median wall | Median RTF | WER | Median / peak RSS | Term recall |
|---|---:|---:|---:|---:|---:|
| FLEURS clean (12.64s, 3 runs) | 2.98s | 0.235 | 17.65% | 1.74 GB | 100% |
| FLEURS noisy 10 dB (12.64s, 3 runs) | 2.93s | 0.232 | 0.00% | 1.71 GB | 100% |
| SpeechOcean Mandarin-L1 (3s, 3 runs) | 1.88s | 0.627 | 0.00% | 1.55 GB | 100% |
| AMI ES2002c (2,423.68s, 1 run, 20s cap) | 473.42s | 0.195 | 22.71% | 2.87 GB | 100% |

The first AMI attempt used the previous 30-second Sherpa cap and failed after
80.06 seconds on a 29.9-second chunk, peaking at 3.12 GB RSS. Repeating with a
20-second cap completed, so the CLI now applies that cap automatically for
Qwen3-ASR while retaining 30 seconds for existing Sherpa architectures.

Sherpa's current Qwen output contains text only. No native timestamps, word
timing, speaker labels, or detected-language metadata were available to score;
the manifest records zero-duration model segments as unreliable instead of
claiming alignment. The separate official Qwen forced aligner was not folded
into this path: it is another model artifact, is documented for audio up to five
minutes, and is not part of the maintained Sherpa Qwen ONNX package.

## Current remote-provider assessment

For Transcript Intelligence feature breadth, Deepgram remains the most advanced
tested integration, especially `nova-3-medical` with domain keyterms. It is the
only tested provider that returned ASR together with provider-native utterances,
word timestamps, diarization, summary, topics, intents, sentiment, entity
extraction, model metadata, and intelligence token usage in one response. The
representative-corpus reference above separately found the lowest observed AMI
WER and DER for Deepgram `nova-3`, but its short noisy-fixture WER was not the
best. Feature breadth and corpus accuracy are therefore reported separately.

This does not mean every Deepgram intelligence field should be treated as ground truth. In the 5-minute medical interview sample, `nova-3-medical` returned useful entities, topics, intents, and sentiment, but its summary made a role error. Without keyterms it also misheard `Ofev` as `OFAP`; adding keyterms such as `Ofev`, `Esbriet`, `IPF`, and `Producta` corrected the medical brand terms and improved speaker consistency in the observed run.

Use this table as a workflow-feature assessment, not a quality leaderboard. The
local llama.cpp Qwen experiment is excluded because it does not provide the
timestamp and metadata contract used by this comparison and failed the long-form
quality check in the representative-corpus run.

| Rank | Provider / Model | Current assessment |
|---|---|---|
| 1 | Deepgram `nova-3-medical` + keyterms | Best Transcript Intelligence candidate; strongest structured metadata and good ASR when keyterms are supplied. |
| 2 | Qwen `qwen3-asr-flash-filetrans` | Strong pure ASR baseline with word timestamps, but less downstream intelligence metadata. |
| 3 | OpenAI hosted transcription | Strong general ASR, but less structured transcript intelligence in the current CLI path. |
| 4 | Gemini | Useful whole-file multimodal transcription and summary path, but timestamps/speakers are model-generated rather than dedicated ASR metadata. |
| 5 | NVIDIA Riva | Provider-native timestamps/diarization through hosted Riva, but less transcript intelligence returned through the current provider path. |

### Clean 5-minute remote provider comparison (2026-06-17)

Measured on `samples/4289US19IPFSegA17Apr20256.45am_5m.wav` after the Deepgram and Gemini signed-URL provider work. The original run used the then-current generic `--autoclean` option; staged URL resources are now cleaned up by default and `--keep-staged-resources` is the explicit opt-out.

| Provider / model | Processing time | RTF | Segments | Timing | Speakers | Word timestamps | Assessment |
|---|---:|---:|---:|---|---|---|---|
| Deepgram `nova-3-medical` + keyterms | 23.33s | 0.078 | 68 | provider-native, clamped | provider-native | yes | Best overall Transcript Intelligence candidate; preserved key medical terms and returned rich intelligence metadata. Summary still had a role error. |
| Qwen `qwen3-asr-flash-filetrans` | 11.15s | 0.037 | 71 | provider-native, reliable | none | yes | Strong pure ASR baseline; preserved key terms including `Producta`; no speaker labels or intelligence metadata. |
| OpenAI `gpt-4o-transcribe-diarize` | 115.21s | 0.384 | 85 | provider-native, reliable | provider-native | no | Good timing and diarization, but slowest hosted run in this pass. |
| Gemini `gemini-3.5-flash` | 35.29s | 0.118 | 29 | model-generated, clamped | model-generated | no | Useful role labels and multimodal path, but timestamps remain unreliable for subtitle-grade output. |
| NVIDIA Riva hosted function | 5.83s | 0.019 | 38 | provider-native, reliable | provider-native | yes | Fastest run, but weaker domain term recognition and speaker separation on this sample. |

## Reference benchmark results

These results were measured on a five-minute medical interview recording. The
older local rows did not consistently capture the exact producing commit, fixture
hash, host specification, model artifact hash, or warm/cold state, so they are
historical observations rather than a regression baseline.

| Engine / Model | Wall clock | Realtime factor | Notes |
|---|---|---|---|
| Local whisper.cpp `base` | 3.6s | 83x RT | Best speed/quality trade-off |
| SenseVoice 2024 | 6.6s | 46x RT | Good quality, 50+ languages |
| Sherpa-ONNX Whisper `base` | 10.9s | 27x RT | |
| Moonshine `base` | 14.1s | 21x RT | |
| llama.cpp Qwen3-ASR 1.7B | ~18s | ~16.7x RT | Warm server; text only, no timestamps or provider metadata |
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

The tracked workflow and sanitized records live under [`benchmarks/`](../benchmarks/README.md).
Use `scripts/benchmark-metadata.sh` before a run and the tracked jq sanitizer before
promoting local artifacts from the ignored `output/` directory.

For now, treat these as manual benchmarks in a fixed environment. Before a result
is promoted to a reproducible baseline, record:

- the exact TranscribeIt commit and dependency/runtime versions,
- CPU, memory, OS, and acceleration backend,
- the complete command with secrets redacted,
- warm/cold model and connection state,
- input duration, byte size, codec, and SHA-256,
- model/provider version and local artifact SHA-256 when applicable,
- wall time, RTF, retries, segment count, and sanitized raw manifest/output.

If you want to automate later:
- add a dedicated `criterion` benchmark target,
- pin fixture files,
- and fail CI only on large regressions with generous tolerances.
