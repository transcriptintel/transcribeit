use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use reqwest::multipart;
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::audio::wav::encode_wav;
use crate::engines::openai_api::{is_response_format_not_supported, parse_response_bytes};
use crate::engines::rate_limit::{self, send_with_retry};
use crate::transcriber::{Transcriber, Transcript};

const RESPONSE_FORMAT_UNKNOWN: u8 = 0;
const RESPONSE_FORMAT_SUPPORTED: u8 = 1;
const RESPONSE_FORMAT_UNSUPPORTED: u8 = 2;

pub struct AzureOpenAi {
    endpoint: String,
    deployment: String,
    api_version: String,
    api_key: String,
    language: Option<String>,
    settings: rate_limit::ApiRequestSettings,
    verbose_json_support: AtomicU8,
    client: Client,
}

impl AzureOpenAi {
    pub fn new(
        endpoint: String,
        deployment: String,
        api_version: String,
        api_key: String,
        language: Option<String>,
        settings: rate_limit::ApiRequestSettings,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(settings.request_timeout)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            deployment,
            api_version,
            api_key,
            language,
            settings,
            verbose_json_support: AtomicU8::new(RESPONSE_FORMAT_UNKNOWN),
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

    fn response_formats(&self) -> Vec<Option<&'static str>> {
        match self.verbose_json_support.load(Ordering::Acquire) {
            RESPONSE_FORMAT_SUPPORTED => vec![Some("verbose_json")],
            RESPONSE_FORMAT_UNSUPPORTED => vec![None],
            _ => vec![Some("verbose_json"), None],
        }
    }

    /// Run the format-fallback + retry loop for a given form builder closure.
    async fn transcribe_with_fallback<F>(&self, build_form: F) -> Result<Transcript>
    where
        F: Fn(Option<&'static str>) -> Result<multipart::Form>,
    {
        let url = format!(
            "{}/openai/deployments/{}/audio/transcriptions?api-version={}",
            self.endpoint, self.deployment, self.api_version
        );

        let mut last_error: Option<(reqwest::StatusCode, String)> = None;
        for response_format in self.response_formats() {
            let result = {
                let url = &url;
                let client = &self.client;
                let api_key = &self.api_key;
                let build_form = &build_form;
                let rf = response_format;
                send_with_retry(&self.settings, "Azure transcription API", || {
                    let url = url.clone();
                    let client = client.clone();
                    let api_key = api_key.clone();
                    let form = build_form(rf);
                    Box::pin(async move {
                        let form = form?;
                        client
                            .post(&url)
                            .header("api-key", &api_key)
                            .multipart(form)
                            .send()
                            .await
                            .context("Failed to send request to Azure transcription API")
                    })
                })
                .await
            };

            match result {
                Ok(body) => {
                    if response_format == Some("verbose_json") {
                        self.verbose_json_support
                            .store(RESPONSE_FORMAT_SUPPORTED, Ordering::Release);
                    }
                    return Ok(parse_response_bytes(&body));
                }
                Err((_, ref body_text))
                    if response_format.is_some() && is_response_format_not_supported(body_text) =>
                {
                    self.verbose_json_support
                        .store(RESPONSE_FORMAT_UNSUPPORTED, Ordering::Release);
                    last_error = Some(result.unwrap_err());
                    continue;
                }
                Err((status, body)) => {
                    anyhow::bail!("Azure API returned {status}: {body}");
                }
            }
        }

        let (status, body) = last_error
            .context("No compatible response format found for Azure transcription API")?;
        anyhow::bail!("Azure API returned {status}: {body}");
    }
}

#[async_trait]
impl Transcriber for AzureOpenAi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_path(&self, wav_path: &Path) -> Result<Transcript> {
        let path = wav_path.to_path_buf();
        self.transcribe_with_fallback(|response_format| {
            let mut form = multipart::Form::new();
            if let Some(fmt) = response_format {
                form = form.text("response_format", fmt);
            }
            if let Some(lang) = self.language.as_deref() {
                form = form.text("language", lang.to_string());
            }
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
            let mut form = multipart::Form::new();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::rate_limit::ApiRequestSettings;
    use std::sync::atomic::Ordering;

    #[test]
    fn azure_response_formats_default_prefers_verbose_json() {
        let engine = make_engine();
        assert_eq!(engine.response_formats(), vec![Some("verbose_json"), None]);
    }

    #[test]
    fn azure_response_formats_supported_skips_fallback() {
        let engine = make_engine();
        engine
            .verbose_json_support
            .store(RESPONSE_FORMAT_SUPPORTED, Ordering::Release);
        assert_eq!(engine.response_formats(), vec![Some("verbose_json")]);
    }

    #[test]
    fn azure_response_formats_unsupported_uses_default_format_only() {
        let engine = make_engine();
        engine
            .verbose_json_support
            .store(RESPONSE_FORMAT_UNSUPPORTED, Ordering::Release);
        assert_eq!(engine.response_formats(), vec![None]);
    }

    #[test]
    fn azure_verbose_json_support_cache_is_persistent_across_calls() {
        let engine = make_engine();

        assert_eq!(
            engine.verbose_json_support.load(Ordering::Acquire),
            RESPONSE_FORMAT_UNKNOWN
        );

        engine
            .verbose_json_support
            .store(RESPONSE_FORMAT_UNSUPPORTED, Ordering::Release);
        assert_eq!(engine.response_formats(), vec![None]);

        engine
            .verbose_json_support
            .store(RESPONSE_FORMAT_SUPPORTED, Ordering::Release);
        assert_eq!(engine.response_formats(), vec![Some("verbose_json")]);
    }

    fn make_engine() -> AzureOpenAi {
        AzureOpenAi::new(
            "https://example.com".to_string(),
            "deployment".to_string(),
            "2024-06-01".to_string(),
            "api-key".to_string(),
            None,
            ApiRequestSettings::default(),
        )
        .unwrap()
    }
}
