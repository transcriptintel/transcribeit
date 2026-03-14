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

#[async_trait]
impl Transcriber for OpenAiApi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;

        let form = multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "verbose_json")
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

        let body = resp
            .bytes()
            .await
            .context("Failed to read API response body")?;

        Ok(parse_response_bytes(&body))
    }
}

/// Response with segments (verbose_json format).
#[derive(Deserialize)]
struct VerboseResponse {
    segments: Vec<ApiSegment>,
    #[allow(dead_code)]
    text: String,
}

/// Minimal response (json format, or verbose_json without segments).
#[derive(Deserialize)]
struct PlainResponse {
    text: String,
}

#[derive(Deserialize)]
struct ApiSegment {
    start: f64,
    end: f64,
    text: String,
}

/// Parse response bytes, trying verbose_json first then falling back to plain json.
/// This ensures compatibility with endpoints that don't support verbose_json.
pub fn parse_response_bytes(body: &[u8]) -> Transcript {
    // Try verbose format with segments first
    if let Ok(resp) = serde_json::from_slice::<VerboseResponse>(body)
        && !resp.segments.is_empty()
    {
        return Transcript {
            segments: resp
                .segments
                .into_iter()
                .map(|s| Segment {
                    start_ms: (s.start * 1000.0) as i64,
                    end_ms: (s.end * 1000.0) as i64,
                    text: s.text,
                })
                .collect(),
        };
    }

    // Fall back to plain text response
    if let Ok(resp) = serde_json::from_slice::<PlainResponse>(body) {
        return Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 0,
                text: resp.text,
            }],
        };
    }

    // Last resort: treat entire body as text
    Transcript {
        segments: vec![Segment {
            start_ms: 0,
            end_ms: 0,
            text: String::from_utf8_lossy(body).into_owned(),
        }],
    }
}
