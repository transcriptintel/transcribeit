use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::Path;

use crate::audio::wav::read_wav_bytes;

/// A segment of transcribed text with timing info.
#[derive(Default)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub emotion: Option<String>,
    pub words: Vec<Word>,
}

#[derive(Default, Clone)]
pub struct Word {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub punctuation: Option<String>,
}

/// Full transcript result.
pub struct Transcript {
    pub segments: Vec<Segment>,
    pub provider_metadata: Option<serde_json::Value>,
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

    /// Shift every segment and word timestamp by the same absolute offset.
    pub fn shift_by(&mut self, offset_ms: i64) {
        for segment in &mut self.segments {
            segment.start_ms += offset_ms;
            segment.end_ms += offset_ms;
            for word in &mut segment.words {
                word.start_ms += offset_ms;
                word.end_ms += offset_ms;
            }
        }
    }
}

#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Whether the engine prefers the original media file for whole-file transcription.
    ///
    /// The pipeline still prepares canonical WAV input for normalization, segmentation,
    /// provider upload encoding, and engines that do not opt into this capability.
    fn prefers_original_media(&self) -> bool {
        false
    }

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

#[cfg(test)]
mod tests {
    use super::{Segment, Transcript, Word};

    #[test]
    fn shifting_transcript_updates_segments_and_words() {
        let mut transcript = Transcript {
            segments: vec![Segment {
                start_ms: 100,
                end_ms: 500,
                words: vec![Word {
                    start_ms: 120,
                    end_ms: 200,
                    text: "hello".to_string(),
                    punctuation: None,
                }],
                ..Default::default()
            }],
            provider_metadata: None,
        };

        transcript.shift_by(2_000);

        assert_eq!(transcript.segments[0].start_ms, 2_100);
        assert_eq!(transcript.segments[0].end_ms, 2_500);
        assert_eq!(transcript.segments[0].words[0].start_ms, 2_120);
        assert_eq!(transcript.segments[0].words[0].end_ms, 2_200);
    }
}
