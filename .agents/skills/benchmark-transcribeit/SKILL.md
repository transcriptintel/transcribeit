---
name: benchmark-transcribeit
description: Design, run, compare, sanitize, and record reproducible TranscribeIt ASR benchmarks for hosted providers, OpenAI-compatible endpoints, local engines, model revisions, diarization, timestamps, quality, latency, and compatibility. Use when the user asks to benchmark, compare, evaluate, retest, establish a baseline, or verify a new transcription model or response format.
---

# Benchmark TranscribeIt

Use `benchmarks/README.md` as the publication protocol and `docs/performance-benchmarks.md` as the interpretation surface.

## Define the benchmark

1. Record the question being tested: latency, transcript quality, timestamps, speakers, metadata, cost-relevant usage, caching, or request compatibility.
2. Pin the provider, exact model name, endpoint/region when relevant, request options, TranscribeIt commit and dirty state, fixture hash/duration/size, machine, and tool versions.
3. Classify the run as a clean baseline, dirty-worktree reference, compatibility probe, or exploratory observation. Never promote dirty or unrepeatable evidence to a regression baseline.
4. State warm/cold cache and server state, retry count, concurrency, language hint, segmentation, and preprocessing differences.

## Protect credentials and data

- Reuse configured provider keys without per-run confirmation. If a key is missing, ask the user to add it through the ignored environment workflow.
- Pass keys through provider environment variables or private key files. Never print or persist their values.
- Suppress provider response bodies in smoke tasks. Keep transcript text, segments, words, request IDs, signed URLs, local absolute paths, and cache paths out of tracked results.
- Store raw results only in ignored local paths. Run `benchmarks/sanitize-manifest.jq` before publishing a tracked record and inspect the sanitized output manually.

## Run fairly

- Use the same immutable fixture for comparisons and record its SHA-256.
- Separate conversion time, provider processing time, and wall time when the data permits. Report real-time factor with the exact duration basis.
- Repeat latency-sensitive runs. Preserve failures and unsupported request formats as results instead of silently changing the model or request.
- Do not compare text-only output with timestamped or diarized output as if capabilities were equal. Record timing origin and reliability, speaker origin, detected language, and provider metadata separately.
- Treat one fixture and model-generated summaries as observations, not quality ground truth. Use a representative corpus and reviewed references for publishable quality claims.

## Clean up downloaded models

1. Before a download, resolve and record the exact evaluation-owned directory, whether it already existed, its initial byte count, and free disk space. Prefer a newly created, narrowly named directory outside shared model caches.
2. Record the downloaded model revision, file or tree hashes, and byte count before running the benchmark. Do not move an evaluation download into a shared cache merely for convenience.
3. After evidence capture, including on failure or cancellation, remove only the files and directories created for that evaluation. Never delete a pre-existing model, shared cache, workspace root, home directory, broad glob, or unresolved environment-variable target.
4. Verify and record that the evaluation-owned path is absent and how many bytes were reclaimed. If an artifact must remain, record its exact path privately, owner, reason, and expiry instead of reporting cleanup as complete.

## Record and validate

1. Capture machine and tool metadata with `scripts/benchmark-metadata.sh`.
2. Save the exact secret-free command, timestamps, metrics, output artifact hashes, failure classification, and downloadable-model cleanup record when applicable.
3. Sanitize the record and verify that no transcript/request/path material remains.
4. Update `docs/performance-benchmarks.md` with dated interpretation and capability boundaries. Update the relevant tracked `docs/issues/TI-NNN.md` page.
5. Run the benchmark JSON assertions documented in `benchmarks/README.md`, `git diff --check`, and the relevant code/docs validators.

Report successful, failed, rejected, and skipped runs separately. A provider rejection such as an unsupported `response_format` is compatibility evidence, not a successful transcription benchmark.
