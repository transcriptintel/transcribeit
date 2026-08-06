# Retired Sherpa-ONNX integration

TranscribeIt retired its Sherpa-ONNX integration on 2026-08-06 in
[TI-010](../issues/TI-010.md). The provider had no known users, while keeping it
required a native shared-library bootstrap, ONNX model lifecycle code, raw FFI,
platform-specific linking, and a second local segmentation/diarization stack.

## What was removed

- the `sherpa-onnx` provider and Cargo feature;
- ONNX model download, discovery, integrity, and setup commands;
- Whisper, Qwen3-ASR 0.6B, Moonshine, and SenseVoice Sherpa adapters;
- Silero VAD segmentation and its model option;
- local ONNX speaker diarization post-processing;
- Sherpa native-library bootstrap, linker configuration, and all-feature CI job.

Former provider names, setup components, model formats, flags, and environment
variables are intentionally rejected instead of being silently ignored.

## Why it was retired

The clean TI-007 baseline did not justify the maintenance surface:

| Historical path | AMI WER | AMI wall / RTF | Capability boundary |
|---|---:|---:|---|
| local whisper.cpp `base` | 31.59% | 33.69s / 0.0139 | segment timing, no speakers |
| Sherpa Whisper `base` + VAD | 32.12% | 121.47s / 0.0501 | no scoreable timing or speakers |
| Sherpa Qwen3-ASR 0.6B int8 + VAD | 22.71% | 431.10s / 0.1779 | text only |

Sherpa Qwen was more accurate than the local Whisper baseline on the single AMI
run, but it was substantially slower and returned no timing, speaker, or
detected-language metadata. The result remains useful evidence, but it does not
make the integration a good maintained default.

## Migration choices

| Former use | Current choice |
|---|---|
| Local Whisper through Sherpa | Use the built-in local whisper.cpp provider and GGML models. |
| Local Qwen3-ASR | Use Qwen FileTrans, or run a separately managed llama.cpp OpenAI-compatible server with `--provider openai-compatible`. The llama.cpp audio path remains experimental until TI-005 is decided. |
| Local post-processing diarization | Use Deepgram, Gemini, NVIDIA Riva, or OpenAI `gpt-4o-transcribe-diarize`. Unsupported provider/model combinations now fail before transcription. |
| Silero VAD segmentation | Use the maintained FFmpeg `silencedetect` path with `--segment` and its threshold/duration controls. |

## Preserved evidence

The implementation is gone, but dated evidence remains unchanged in:

- [TI-004 native Qwen3-ASR evaluation](../issues/TI-004.md);
- [TI-007 provider rebaseline](../issues/TI-007.md);
- [performance benchmarks](../performance-benchmarks.md);
- the sanitized records under [`benchmarks/results`](../../benchmarks/results/).

Those records describe the code and model revisions tested at the time. They are
historical observations, not current CLI instructions or supported providers.
