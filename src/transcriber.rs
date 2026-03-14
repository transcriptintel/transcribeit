use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::Path;

use crate::audio::wav::read_wav_bytes;

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

    async fn transcribe_path(&self, wav_path: &Path) -> Result<Transcript> {
        let wav_bytes = tokio::fs::read(wav_path)
            .await
            .with_context(|| format!("Failed to read WAV file: {}", wav_path.display()))?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript> {
        let audio_samples = read_wav_bytes(&wav_bytes)?;
        self.transcribe(audio_samples).await
    }
}
