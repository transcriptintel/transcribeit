#[cfg(target_os = "macos")]
mod bridge;
#[cfg(any(target_os = "macos", test))]
mod response;

#[cfg(target_os = "macos")]
use std::path::Path;

#[cfg(target_os = "macos")]
use anyhow::{Context, Result};
#[cfg(target_os = "macos")]
use async_trait::async_trait;

#[cfg(target_os = "macos")]
use crate::audio::extract::extract_to_wav;
#[cfg(target_os = "macos")]
use crate::transcriber::{Transcriber, Transcript};

#[cfg(target_os = "macos")]
pub struct AppleSpeech {
    locale: Option<String>,
}

#[cfg(target_os = "macos")]
impl AppleSpeech {
    pub fn new(locale: Option<String>) -> Self {
        Self { locale }
    }

    async fn transcribe_native(&self, path: &Path) -> Result<Transcript> {
        let path = path.to_owned();
        let locale = self.locale.clone();
        tokio::task::spawn_blocking(move || {
            let payload = bridge::transcribe(&path, locale.as_deref())?;
            response::parse(&payload)
        })
        .await?
    }
}

#[cfg(target_os = "macos")]
#[async_trait]
impl Transcriber for AppleSpeech {
    fn prefers_original_media(&self) -> bool {
        true
    }

    async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
        anyhow::bail!("Apple Speech requires file-based transcription")
    }

    async fn transcribe_path(&self, path: &Path) -> Result<Transcript> {
        match self.transcribe_native(path).await {
            Ok(transcript) => Ok(transcript),
            Err(error) if bridge::is_audio_read_failure(&error) => {
                eprintln!(
                    "AVAudioFile could not read the original media; converting to mono 16kHz WAV and retrying..."
                );
                let wav = extract_to_wav(path, false)
                    .await
                    .context("Apple Speech fallback media conversion failed")?;
                self.transcribe_native(wav.as_ref())
                    .await
                    .context("Apple Speech failed after fallback media conversion")
            }
            Err(error) => Err(error),
        }
    }
}
