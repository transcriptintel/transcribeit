use std::path::Path;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use crate::audio::segment::get_duration;
use crate::audio::wav::encode_wav;
use crate::engines::rate_limit;
use crate::storage::s3::{S3CleanupResult, S3Uploader};
use crate::transcriber::{Transcriber, Transcript};

mod analysis;
mod cached_content;
mod file_cache;
mod files;
mod response;
mod schema;
mod streaming;

use cached_content::add_cached_content_metadata;
use file_cache::GeminiFileCache;
pub use file_cache::GeminiFileCacheConfig;
use files::{GeminiUploadRef, with_file_cleanup_metadata};
use response::{GeminiResponseContext, parse_stream_generate_response};
use schema::{
    audio_mime, generate_payload, generate_payload_with_cached_content, prompt_text,
    upload_base_url,
};

pub struct GeminiApi {
    api_base_url: String,
    upload_base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    settings: rate_limit::ApiRequestSettings,
    client: Client,
    file_cache: Option<GeminiFileCache>,
    signed_url_uploader: Option<S3Uploader>,
    autoclean: bool,
}

pub struct GeminiConfig {
    pub api_base_url: String,
    pub api_key: String,
    pub model: String,
    pub language: Option<String>,
    pub settings: rate_limit::ApiRequestSettings,
    pub file_cache: Option<GeminiFileCacheConfig>,
    pub signed_url_uploader: Option<S3Uploader>,
    pub autoclean: bool,
}

const GEMINI_SIGNED_URL_MAX_BYTES: u64 = 100 * 1024 * 1024;

struct GeminiGenerateInput<'a> {
    file_uri: &'a str,
    mime_type: &'a str,
    input_bytes: u64,
    duration_secs: Option<f64>,
    upload: Option<&'a GeminiUploadRef>,
    upload_method: &'a str,
    file_url_present: bool,
}

impl GeminiApi {
    pub fn new(config: GeminiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.settings.request_timeout)
            .build()
            .context("Failed to build HTTP client")?;
        let api_base_url = config.api_base_url.trim_end_matches('/').to_string();
        let upload_base_url = upload_base_url(&api_base_url);

        Ok(Self {
            api_base_url,
            upload_base_url,
            api_key: config.api_key,
            model: config.model,
            language: config.language,
            settings: config.settings,
            client,
            file_cache: config.file_cache.map(GeminiFileCache::new),
            signed_url_uploader: config.signed_url_uploader,
            autoclean: config.autoclean,
        })
    }

    async fn transcribe_file(&self, audio_path: &Path) -> Result<Transcript> {
        let bytes = tokio::fs::read(audio_path)
            .await
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        let mime_type = audio_mime(audio_path);
        let duration_secs = get_duration(audio_path).await.ok();
        if let Some(uploader) = &self.signed_url_uploader {
            anyhow::ensure!(
                bytes.len() as u64 <= GEMINI_SIGNED_URL_MAX_BYTES,
                "Gemini signed URL input supports files up to 100 MB; prepared input is {:.2} MB",
                bytes.len() as f64 / (1024.0 * 1024.0)
            );
            anyhow::ensure!(
                !is_gemini_2_0_model(&self.model),
                "Gemini signed URL input is not supported for Gemini 2.0 family models; use Gemini Files API mode instead"
            );
            let upload = uploader.upload_and_presign_object(audio_path).await?;
            let mut transcript = self
                .generate_transcript(GeminiGenerateInput {
                    file_uri: &upload.url,
                    mime_type,
                    input_bytes: bytes.len() as u64,
                    duration_secs,
                    upload: None,
                    upload_method: "signed_url",
                    file_url_present: true,
                })
                .await?;
            let cleanup = if self.autoclean {
                uploader.cleanup_uploaded_object(&upload).await
            } else {
                S3CleanupResult::skipped(&upload)
            };
            if let Some(error) = cleanup.error.as_deref() {
                eprintln!(
                    "Failed to delete staged Gemini object s3://{}/{}: {error}",
                    cleanup.bucket, cleanup.key
                );
            }
            add_gemini_staging_metadata(&mut transcript, cleanup);
            return Ok(transcript);
        }

        let upload = self.resolve_file(audio_path, &bytes, mime_type).await?;
        let response = self
            .generate_transcript(GeminiGenerateInput {
                file_uri: &upload.file.uri,
                mime_type,
                input_bytes: bytes.len() as u64,
                duration_secs,
                upload: Some(&upload),
                upload_method: "files_api",
                file_url_present: false,
            })
            .await;
        let cleanup = self.cleanup_file_after_run(&upload).await;

        let mut transcript = response?;
        transcript.provider_metadata = Some(with_file_cleanup_metadata(
            transcript.provider_metadata.take(),
            upload,
            mime_type,
            bytes.len() as u64,
            cleanup,
        ));
        Ok(transcript)
    }

    async fn generate_transcript(&self, input: GeminiGenerateInput<'_>) -> Result<Transcript> {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.api_base_url,
            urlencoding::encode(&self.model)
        );
        let prompt = prompt_text(self.language.as_deref(), input.duration_secs);
        let cached_content = if let Some(upload) = input.upload {
            self.resolve_cached_content(upload, input.file_uri, input.mime_type)
                .await?
        } else {
            None
        };
        let payload = if let Some(cached_content) = &cached_content {
            generate_payload_with_cached_content(&cached_content.name, &prompt)
        } else {
            generate_payload(input.file_uri, input.mime_type, &prompt)
        };
        let chunks = self.stream_generate_chunks(&url, &payload).await?;

        let mut transcript = parse_stream_generate_response(
            &chunks,
            GeminiResponseContext {
                model: &self.model,
                api_base_url: &self.api_base_url,
                mime_type: input.mime_type,
                input_bytes: input.input_bytes,
                duration_secs: input.duration_secs,
                upload_method: input.upload_method,
                file_url_present: input.file_url_present,
            },
        );
        if let Some(cached_content) = cached_content {
            add_cached_content_metadata(&mut transcript, cached_content);
        }
        Ok(transcript)
    }
}

fn add_gemini_staging_metadata(transcript: &mut Transcript, cleanup: S3CleanupResult) {
    let metadata = transcript
        .provider_metadata
        .get_or_insert_with(|| serde_json::json!({ "gemini": {} }));
    if let Some(gemini) = metadata
        .get_mut("gemini")
        .and_then(serde_json::Value::as_object_mut)
    {
        gemini.insert(
            "staging".to_string(),
            serde_json::json!({
                "provider": "s3",
                "cleanup": cleanup.to_metadata(),
            }),
        );
    }
}

fn is_gemini_2_0_model(model: &str) -> bool {
    let model = model
        .strip_prefix("models/")
        .unwrap_or(model)
        .to_ascii_lowercase();
    model.starts_with("gemini-2.0")
}

#[async_trait]
impl Transcriber for GeminiApi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_path(&self, audio_path: &Path) -> Result<Transcript> {
        self.transcribe_file(audio_path).await
    }

    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript> {
        let tmp = tempfile::Builder::new()
            .prefix("transcribeit-gemini-")
            .suffix(".wav")
            .tempfile()
            .context("Failed to create temporary WAV file")?;
        tokio::fs::write(tmp.path(), wav_bytes)
            .await
            .context("Failed to write temporary WAV file")?;
        self.transcribe_path(tmp.path()).await
    }
}
