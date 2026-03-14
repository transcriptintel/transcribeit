use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::multipart;

use crate::audio::wav::encode_wav;
use crate::engines::openai_api::{is_response_format_not_supported, parse_response_bytes};
use crate::engines::rate_limit::{self, RateLimitCheck};
use crate::transcriber::{Transcriber, Transcript};

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

#[async_trait]
impl Transcriber for AzureOpenAi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        let url = format!(
            "{}/openai/deployments/{}/audio/transcriptions?api-version={}",
            self.endpoint, self.deployment, self.api_version
        );

        let client = reqwest::Client::new();
        let mut last_error: Option<(reqwest::StatusCode, String)> = None;
        for response_format in [Some("verbose_json"), None] {
            for attempt in 0..=rate_limit::max_retries() {
                let mut form = multipart::Form::new();
                if let Some(fmt) = response_format {
                    form = form.text("response_format", fmt);
                }
                form = form.part(
                    "file",
                    multipart::Part::bytes(wav_bytes.clone())
                        .file_name("audio.wav")
                        .mime_str("audio/wav")?,
                );

                let resp = client
                    .post(&url)
                    .header("api-key", &self.api_key)
                    .multipart(form)
                    .send()
                    .await
                    .context("Failed to send request to Azure transcription API")?;

                match rate_limit::check_response(resp).await {
                    RateLimitCheck::Ok(body) => return Ok(parse_response_bytes(&body)),
                    RateLimitCheck::RetryAfter(wait) => {
                        if attempt == rate_limit::max_retries() {
                            anyhow::bail!(
                                "Rate limited after {} retries, last wait was {}s",
                                rate_limit::max_retries(),
                                wait.as_secs()
                            );
                        }
                        eprintln!(
                            "    Rate limited, retrying in {}s (attempt {}/{})...",
                            wait.as_secs(),
                            attempt + 1,
                            rate_limit::max_retries()
                        );
                        tokio::time::sleep(wait).await;
                    }
                    RateLimitCheck::Error(status, body) => {
                        if response_format.is_some() && is_response_format_not_supported(&body) {
                            last_error = Some((status, body));
                            break; // try next format
                        }
                        anyhow::bail!("Azure API returned {status}: {body}");
                    }
                }
            }
        }

        let (status, body) = last_error
            .context("No compatible response format found for Azure transcription API")?;
        anyhow::bail!("Azure API returned {status}: {body}");
    }
}
