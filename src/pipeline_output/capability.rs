use crate::output::manifest::Capabilities;
use crate::pipeline::PipelineConfig;
use crate::transcriber::Transcript;

pub(super) fn build_capabilities(config: &PipelineConfig, transcript: &Transcript) -> Capabilities {
    Capabilities {
        segments: !transcript.segments.is_empty(),
        word_timestamps: transcript
            .segments
            .iter()
            .any(|segment| !segment.words.is_empty()),
        speaker_labels: transcript
            .segments
            .iter()
            .any(|segment| segment.speaker.is_some()),
        language_per_segment: transcript
            .segments
            .iter()
            .any(|segment| segment.language.is_some()),
        emotion_per_segment: transcript
            .segments
            .iter()
            .any(|segment| segment.emotion.is_some()),
        native_timestamps: native_timestamps(&config.provider_name)
            && transcript
                .segments
                .iter()
                .any(|segment| segment.end_ms > segment.start_ms),
    }
}

fn native_timestamps(provider: &str) -> bool {
    matches!(
        provider,
        "local" | "openai" | "azure" | "qwen-filetrans" | "nvidia-riva" | "deepgram"
    )
}

#[cfg(test)]
mod tests;
