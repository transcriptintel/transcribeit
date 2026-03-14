use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::multipart;
use serde::Deserialize;

use crate::audio::wav::encode_wav;
use crate::transcriber::{Segment, Transcriber, Transcript};

pub struct AzureOpenAi {
    endpoint: String,
    deployment: String,
    api_version: String,
    api_key: String,
}

impl AzureOpenAi {
    pub fn new(endpoint: String, deployment: String, api_version: String, api_key: String) -> Self {
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            deployment,
            api_version,
            api_key,
        }
    }
}

#[derive(Deserialize)]
struct ApiResponse {
    text: String,
}

#[async_trait]
impl Transcriber for AzureOpenAi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;

        let form = multipart::Form::new().text("response_format", "json").part(
            "file",
            multipart::Part::bytes(wav_bytes)
                .file_name("audio.wav")
                .mime_str("audio/wav")?,
        );

        let url = format!(
            "{}/openai/deployments/{}/audio/transcriptions?api-version={}",
            self.endpoint, self.deployment, self.api_version
        );

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("api-key", &self.api_key)
            .multipart(form)
            .send()
            .await
            .context("Failed to send request to Azure transcription API")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Azure API returned {status}: {body}");
        }

        let api_resp: ApiResponse = resp
            .json()
            .await
            .context("Failed to parse Azure API response")?;

        Ok(Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 0,
                text: api_resp.text,
            }],
        })
    }
}
