use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::multipart;
use serde::Deserialize;

use crate::audio::wav::encode_wav;
use crate::engines::rate_limit::{self, RateLimitCheck};
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
        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        let client = reqwest::Client::new();

        let mut last_error: Option<(reqwest::StatusCode, String)> = None;
        for response_format in [Some("verbose_json"), None] {
            for attempt in 0..=rate_limit::max_retries() {
                let mut form = multipart::Form::new().text("model", self.model.clone());
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
                    .bearer_auth(&self.api_key)
                    .multipart(form)
                    .send()
                    .await
                    .context("Failed to send request to transcription API")?;

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
                        anyhow::bail!("API returned {status}: {body}");
                    }
                }
            }
        }

        let (status, body) =
            last_error.context("No compatible response format found for transcription API")?;
        anyhow::bail!("API returned {status}: {body}");
    }
}

/// Response with segments (verbose_json format).
#[derive(Deserialize)]
struct VerboseResponse {
    segments: Option<Vec<ApiSegment>>,
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
    // Try verbose format with segments first.
    if let Ok(resp) = serde_json::from_slice::<VerboseResponse>(body)
        && let Some(segs) = resp.segments
        && !segs.is_empty()
    {
        return Transcript {
            segments: segs
                .into_iter()
                .map(|s| Segment {
                    start_ms: (s.start * 1000.0) as i64,
                    end_ms: (s.end * 1000.0) as i64,
                    text: s.text,
                })
                .collect(),
        };
    }

    // Fall back to plain text response.
    if let Ok(resp) = serde_json::from_slice::<PlainResponse>(body) {
        return Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 0,
                text: resp.text,
            }],
        };
    }

    // Last resort: treat entire body as text.
    Transcript {
        segments: vec![Segment {
            start_ms: 0,
            end_ms: 0,
            text: String::from_utf8_lossy(body).into_owned(),
        }],
    }
}

pub(crate) fn is_response_format_not_supported(body: &str) -> bool {
    // Prefer structured error objects when available.
    #[derive(Deserialize)]
    struct ErrorPayload {
        #[serde(default)]
        error: ErrorDetails,
    }
    #[derive(Deserialize, Default)]
    struct ErrorDetails {
        #[serde(rename = "type", default)]
        _type: Option<String>,
        #[serde(default)]
        message: Option<String>,
        #[serde(default)]
        code: Option<String>,
        #[serde(default)]
        param: Option<String>,
    }

    if let Ok(err) = serde_json::from_str::<ErrorPayload>(body) {
        if err.error.param.as_deref() == Some("response_format") {
            return true;
        }
        if err
            .error
            .message
            .as_deref()
            .is_some_and(|m| m.to_ascii_lowercase().contains("response_format"))
        {
            return true;
        }
        if err
            .error
            .code
            .as_deref()
            .is_some_and(|code| code.to_ascii_lowercase().contains("response_format"))
        {
            return true;
        }
    }

    // Fallback to substring matching for providers returning plain text errors.
    let body = body.to_ascii_lowercase();
    body.contains("response_format")
        && (body.contains("invalid")
            || body.contains("unsupported")
            || body.contains("unsupported value"))
}

#[cfg(test)]
mod tests {
    use super::is_response_format_not_supported;

    #[test]
    fn detects_structured_openai_error_param() {
        let body = r#"{"error":{"message":"Unsupported value: 'response_format'","param":"response_format","type":"invalid_request_error"}}"#;
        assert!(is_response_format_not_supported(body));
    }

    #[test]
    fn detects_structured_azure_error_message() {
        let body = r#"{"error":{"code":"BadRequest","message":"The request is invalid. Parameter 'response_format' is invalid."}}"#;
        assert!(is_response_format_not_supported(body));
    }

    #[test]
    fn falls_back_to_structured_error_code() {
        let body = r#"{"error":{"code":"response_format","message":"Unsupported value"}}"#;
        assert!(is_response_format_not_supported(body));
    }

    #[test]
    fn rejects_unrelated_errors() {
        let body =
            r#"{"error":{"message":"Model not found","param":"model","code":"model_not_found"}}"#;
        assert!(!is_response_format_not_supported(body));
    }

    #[test]
    fn detects_plaintext_error_when_param_missing() {
        let body = "unsupported value 'response_format' for audio transcription";
        assert!(is_response_format_not_supported(body));
    }
}
