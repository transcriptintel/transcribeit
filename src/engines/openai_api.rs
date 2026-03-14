use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use reqwest::multipart;
use serde::Deserialize;
use std::path::Path;

use crate::audio::wav::encode_wav;
use crate::engines::rate_limit::{self, send_with_retry};
use crate::transcriber::{Segment, Transcriber, Transcript};

pub struct OpenAiApi {
    base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    settings: rate_limit::ApiRequestSettings,
    client: Client,
}

impl OpenAiApi {
    pub fn new(
        base_url: String,
        api_key: String,
        model: String,
        language: Option<String>,
        settings: rate_limit::ApiRequestSettings,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(settings.request_timeout)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            language,
            settings,
            client,
        })
    }

    fn audio_mime(path: &Path) -> &'static str {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("mp3") => "audio/mpeg",
            Some(ext) if ext.eq_ignore_ascii_case("wav") => "audio/wav",
            _ => "audio/wav",
        }
    }

    /// Run the format-fallback + retry loop for a given form builder closure.
    async fn transcribe_with_fallback<F>(&self, build_form: F) -> Result<Transcript>
    where
        F: Fn(Option<&'static str>) -> Result<multipart::Form>,
    {
        let url = format!("{}/v1/audio/transcriptions", self.base_url);

        let mut last_error: Option<(reqwest::StatusCode, String)> = None;
        for response_format in [Some("verbose_json"), None] {
            let result = {
                let url = &url;
                let client = &self.client;
                let api_key = &self.api_key;
                let build_form = &build_form;
                let rf = response_format;
                send_with_retry(&self.settings, "transcription API", || {
                    let url = url.clone();
                    let client = client.clone();
                    let api_key = api_key.clone();
                    let form = build_form(rf);
                    Box::pin(async move {
                        let form = form?;
                        client
                            .post(&url)
                            .bearer_auth(&api_key)
                            .multipart(form)
                            .send()
                            .await
                            .context("Failed to send request to transcription API")
                    })
                })
                .await
            };

            match result {
                Ok(body) => return Ok(parse_response_bytes(&body)),
                Err((_, ref body_text))
                    if response_format.is_some() && is_response_format_not_supported(body_text) =>
                {
                    last_error = Some(result.unwrap_err());
                    continue;
                }
                Err((status, body)) => anyhow::bail!("API returned {status}: {body}"),
            }
        }

        let (status, body) =
            last_error.context("No compatible response format found for transcription API")?;
        anyhow::bail!("API returned {status}: {body}");
    }
}

#[async_trait]
impl Transcriber for OpenAiApi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_path(&self, wav_path: &Path) -> Result<Transcript> {
        let path = wav_path.to_path_buf();
        self.transcribe_with_fallback(|response_format| {
            let mut form = multipart::Form::new().text("model", self.model.clone());
            if let Some(fmt) = response_format {
                form = form.text("response_format", fmt);
            }
            if let Some(lang) = self.language.as_deref() {
                form = form.text("language", lang.to_string());
            }
            // Note: Part::file is async but we need a sync closure here.
            // Use blocking read since form building happens before the async send.
            let bytes = std::fs::read(&path)
                .with_context(|| format!("Failed to read audio file: {}", path.display()))?;
            let mime = Self::audio_mime(&path);
            let file_name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            form = form.part(
                "file",
                multipart::Part::bytes(bytes)
                    .file_name(file_name)
                    .mime_str(mime)?,
            );
            Ok(form)
        })
        .await
    }

    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript> {
        self.transcribe_with_fallback(|response_format| {
            let mut form = multipart::Form::new().text("model", self.model.clone());
            if let Some(fmt) = response_format {
                form = form.text("response_format", fmt);
            }
            if let Some(lang) = self.language.as_deref() {
                form = form.text("language", lang.to_string());
            }
            form = form.part(
                "file",
                multipart::Part::bytes(wav_bytes.clone())
                    .file_name("audio.wav")
                    .mime_str("audio/wav")?,
            );
            Ok(form)
        })
        .await
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
