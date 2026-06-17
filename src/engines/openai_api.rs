use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use reqwest::multipart;
use serde::Deserialize;
use serde_json::Value;
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

    fn is_diarize_model(&self) -> bool {
        self.model.eq_ignore_ascii_case("gpt-4o-transcribe-diarize")
    }

    fn response_formats(&self) -> Vec<Option<&'static str>> {
        if self.is_diarize_model() {
            vec![Some("diarized_json"), None]
        } else {
            vec![Some("verbose_json"), None]
        }
    }

    fn base_form(&self, response_format: Option<&'static str>) -> multipart::Form {
        let mut form = multipart::Form::new().text("model", self.model.clone());
        if let Some(fmt) = response_format {
            form = form.text("response_format", fmt);
        }
        if self.is_diarize_model() {
            form = form.text("chunking_strategy", "auto");
        }
        if let Some(lang) = self.language.as_deref() {
            form = form.text("language", lang.to_string());
        }
        form
    }

    /// Run the format-fallback + retry loop for a given form builder closure.
    async fn transcribe_with_fallback<F>(&self, build_form: F) -> Result<Transcript>
    where
        F: Fn(Option<&'static str>) -> Result<multipart::Form>,
    {
        let url = format!("{}/v1/audio/transcriptions", self.base_url);

        let mut last_error: Option<(reqwest::StatusCode, String)> = None;
        for response_format in self.response_formats() {
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
                Ok(body) => {
                    return Ok(self.with_provider_metadata(parse_response_bytes(&body), &body));
                }
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

    fn with_provider_metadata(&self, mut transcript: Transcript, body: &[u8]) -> Transcript {
        let response = serde_json::from_slice::<Value>(body).ok();
        transcript.provider_metadata = Some(serde_json::json!({
            "provider": "openai",
            "schema_version": "openai.metadata.v1",
            "data": {
                "model": self.model,
                "base_url": self.base_url,
                "response": {
                    "usage": response
                        .as_ref()
                        .and_then(|value| value.get("usage").cloned())
                        .unwrap_or(Value::Null),
                }
            }
        }));
        transcript
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
            let mut form = self.base_form(response_format);
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
            let mut form = self.base_form(response_format);
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

/// Parse response bytes, trying verbose_json first then falling back to plain json.
/// This ensures compatibility with endpoints that don't support verbose_json.
pub fn parse_response_bytes(body: &[u8]) -> Transcript {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        if let Some(segments) = parse_json_segments(&value)
            && !segments.is_empty()
        {
            return Transcript {
                segments,
                provider_metadata: None,
            };
        }

        if let Some(text) = value.get("text").and_then(Value::as_str)
            && !text.trim().is_empty()
        {
            return Transcript {
                segments: vec![Segment {
                    start_ms: 0,
                    end_ms: 0,
                    text: text.to_string(),
                    speaker: None,
                    ..Default::default()
                }],
                provider_metadata: None,
            };
        }
    }

    // Last resort: treat entire body as text.
    Transcript {
        segments: vec![Segment {
            start_ms: 0,
            end_ms: 0,
            text: String::from_utf8_lossy(body).into_owned(),
            speaker: None,
            ..Default::default()
        }],
        provider_metadata: None,
    }
}

fn parse_json_segments(value: &Value) -> Option<Vec<Segment>> {
    let segments = value.get("segments")?.as_array()?;
    let parsed = segments.iter().filter_map(parse_json_segment).collect();
    Some(parsed)
}

fn parse_json_segment(value: &Value) -> Option<Segment> {
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())?;

    let start_ms = timestamp_ms(value.get("start")).unwrap_or(0);
    let end_ms = timestamp_ms(value.get("end")).unwrap_or(start_ms);
    let speaker = value
        .get("speaker")
        .or_else(|| value.get("speaker_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    Some(Segment {
        start_ms,
        end_ms,
        text: text.to_string(),
        speaker,
        ..Default::default()
    })
}

fn timestamp_ms(value: Option<&Value>) -> Option<i64> {
    let seconds = match value? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse().ok()?,
        _ => return None,
    };
    Some((seconds * 1000.0).round() as i64)
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
mod tests;
