# Performance benchmarks

This guide keeps benchmark expectations explicit and reproducible so future changes can be measured consistently.

## Measurement environment

Capture these details for every benchmark run:

- Host CPU model and core count
- RAM
- OS and kernel
- `rustc` version
- Provider and exact command used
- Model used (for local) or deployment/model (for APIs)
- Input file duration and codec/container

## Benchmarks to run

### 1. Local model inference throughput

Run on representative files (e.g. 1, 5, 10 minutes).

```bash
time transcribeit run -i <input_file> -m base --output-format text -o ./output
time transcribeit run -i <input_file> -m small --output-format text -o ./output
time transcribeit run -i <input_file> -m small.en --output-format text -o ./output
```

Record:
- wall clock duration
- output length/time ratio (e.g. 600s audio in 180s)
- CPU utilization profile (optional)

### 2. Provider overhead vs local (same input)

```bash
time transcribeit run -p openai -i <input_file> --output-format text -o ./output
time transcribeit run -p azure -i <input_file> --output-format text -o ./output
```

Record:
- wall clock duration
- retry counts (if any)
- segment count
- average segment latency (from logs)

### 3. Segmentation impact

```bash
time transcribeit run -p openai -i <long_file> --segment --segment-concurrency 2 --output-format text -o ./output
time transcribeit run -p openai -i <long_file> --segment --segment-concurrency 1 --max-segment-secs 300 --output-format text -o ./output
```

Record:
- total segment count
- max queue wait
- request-level retry counts

### 4. I/O + conversion overhead

```bash
time transcribeit run -i <video_file> -m base --normalize --output-format text -o ./output
time transcribeit run -i <audio_file> -m base --normalize --output-format text -o ./output
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

## CI/automatable baseline

For now, treat these as manual benchmarks in a fixed environment.
If you want to automate later:
- add a dedicated `criterion` benchmark target,
- pin fixture files,
- and fail CI only on large regressions with generous tolerances.
