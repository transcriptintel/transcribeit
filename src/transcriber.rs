use anyhow::Result;
use async_trait::async_trait;

/// A segment of transcribed text with timing info.
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// Full transcript result.
pub struct Transcript {
    pub segments: Vec<Segment>,
}

impl Transcript {
    /// Concatenate all segment texts into one string.
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript>;
}
