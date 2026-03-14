use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::multipart;
use serde::Deserialize;

use crate::audio::wav::encode_wav;
use crate::transcriber::{Segment, Transcriber, Transcript};

pub struct OpenAiApi {
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAiApi {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
        }
    }
}

#[derive(Deserialize)]
struct ApiResponse {
    text: String,
}

#[async_trait]
impl Transcriber for OpenAiApi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;

        let form = multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "json")
            .part(
                "file",
                multipart::Part::bytes(wav_bytes)
                    .file_name("audio.wav")
                    .mime_str("audio/wav")?,
            );

        let url = format!("{}/v1/audio/transcriptions", self.base_url);

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .context("Failed to send request to transcription API")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("API returned {status}: {body}");
        }

        let api_resp: ApiResponse = resp.json().await.context("Failed to parse API response")?;

        Ok(Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 0,
                text: api_resp.text,
            }],
        })
    }
}
