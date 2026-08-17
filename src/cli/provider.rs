use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum Provider {
    /// Apple on-device SpeechAnalyzer (macOS 26+ with Apple Intelligence)
    #[value(name = "apple-speech")]
    AppleSpeech,
    /// Local whisper.cpp engine
    Local,
    /// OpenAI-compatible API
    Openai,
    /// Azure OpenAI API
    Azure,
    /// Qwen3-ASR-Flash-Filetrans via DashScope and S3 pre-signed URLs
    #[value(name = "qwen-filetrans")]
    QwenFiletrans,
    /// Gemini multimodal transcription through Files API and streamed generateContent
    Gemini,
    /// NVIDIA-hosted Riva ASR over gRPC
    #[value(name = "nvidia-riva")]
    NvidiaRiva,
    /// Deepgram batch transcription API
    Deepgram,
}
