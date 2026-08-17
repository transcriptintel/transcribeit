# Changelog

All notable changes to TranscribeIt are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Raised the minimum and pinned build toolchain from Rust 1.96 to Rust 1.97.1,
  including local, CI, and release build surfaces.

## [2.0.0] - 2026-08-07

### Breaking changes

- Retired the released Sherpa-ONNX provider, ONNX model installer, VAD options,
  and Sherpa-backed local diarization. Historical evidence and migration paths
  remain documented under `docs/retired/`.
- Temporary S3/R2 provider inputs are now deleted after every provider outcome
  by default. `--keep-staged-resources` is the explicit debugging opt-out.
- Provider credentials no longer fall back to `OPENAI_API_KEY`; each hosted
  provider uses its own environment variable or an explicit private key file.

### Added

- Hardened OpenAI model-sensitive request formats: `gpt-transcribe` uses the
  default JSON response, while `gpt-4o-transcribe-diarize` uses
  `diarized_json` with automatic chunking and a reviewed fallback.
- Documented and benchmarked a separately managed Qwen3-ASR/llama.cpp
  compatibility path through the OpenAI-compatible transcription endpoint.
- Added pinned representative fixtures, scored hosted/local benchmark records,
  cleanup evidence, and a resumable Bun/YAML benchmark harness.
- Added verified whisper.cpp model installation with pinned revisions, sizes,
  and SHA-256 hashes.
- Added a guarded multi-platform GitHub release workflow producing archives and
  SHA-256 checksums.

### Changed

- Updated the maintained Gemini default to `gemini-3.6-flash` and hardened
  Files API reuse, streaming, cached-content, and staged-resource cleanup.
- Hardened provider retries, response-size limits, prepared-upload segmentation,
  timing/capability evidence, batch collision handling, and private output files.
- Restructured large runtime coordinators into focused provider, command,
  pipeline-output, and setup modules with a repository module-size ratchet.

### Security

- Added credential-specific configuration, owner-readable key-file checks,
  secret-safe generated help, local media protocol restrictions, atomic private
  manifests/cache indexes, RustSec CI, and bounded provider response handling.

[Unreleased]: https://github.com/transcriptintel/transcribeit/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/transcriptintel/transcribeit/compare/v1.6.0...v2.0.0
