#[cfg(target_os = "macos")]
mod bridge;
#[cfg(any(target_os = "macos", test))]
mod response;

use std::path::Path;

#[cfg(target_os = "macos")]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;

#[cfg(target_os = "macos")]
use crate::audio::extract::extract_to_wav;
use crate::transcriber::{Transcriber, Transcript};

pub struct AppleSpeech {
    #[cfg(target_os = "macos")]
    locale: Option<String>,
}

impl AppleSpeech {
    pub fn new(locale: Option<String>) -> Self {
        #[cfg(target_os = "macos")]
        {
            Self { locale }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = locale;
            Self {}
        }
    }

    #[cfg(target_os = "macos")]
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

#[async_trait]
impl Transcriber for AppleSpeech {
    fn prefers_original_media(&self) -> bool {
        true
    }

    async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
        anyhow::bail!("Apple Speech requires file-based transcription")
    }

    async fn transcribe_path(&self, path: &Path) -> Result<Transcript> {
        #[cfg(target_os = "macos")]
        {
            match self.transcribe_native(path).await {
                Ok(transcript) => return Ok(transcript),
                Err(error) if bridge::is_audio_read_failure(&error) => {
                    eprintln!(
                        "AVAudioFile could not read the original media; converting to mono 16kHz WAV and retrying..."
                    );
                    let wav = extract_to_wav(path, false)
                        .await
                        .context("Apple Speech fallback media conversion failed")?;
                    return self
                        .transcribe_native(wav.as_ref())
                        .await
                        .context("Apple Speech failed after fallback media conversion");
                }
                Err(error) => return Err(error),
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = path;
            anyhow::bail!("provider 'apple-speech' requires macOS 26 or later")
        }
    }
}
