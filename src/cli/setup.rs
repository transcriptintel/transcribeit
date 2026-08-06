use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum SetupComponent {
    /// Default STT models (GGML base)
    Models,
    /// Silero VAD model for speech-aware segmentation
    Vad,
    /// Speaker diarization models (segmentation + embedding)
    Diarize,
    /// sherpa-onnx shared libraries for the current platform
    #[value(name = "sherpa-libs")]
    SherpaLibs,
    /// Qwen3-ASR 0.6B int8 ONNX model for local sherpa-onnx inference
    #[value(name = "qwen3-asr")]
    Qwen3Asr,
}
