use std::path::Path;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;

use crate::audio::segment::get_duration;
use crate::audio::wav::encode_wav;
use crate::engines::rate_limit;
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
use response::parse_stream_generate_response;
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
}

impl GeminiApi {
    pub fn new(
        api_base_url: String,
        api_key: String,
        model: String,
        language: Option<String>,
        settings: rate_limit::ApiRequestSettings,
        file_cache: Option<GeminiFileCacheConfig>,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(settings.request_timeout)
            .build()
            .context("Failed to build HTTP client")?;
        let api_base_url = api_base_url.trim_end_matches('/').to_string();
        let upload_base_url = upload_base_url(&api_base_url);

        Ok(Self {
            api_base_url,
            upload_base_url,
            api_key,
            model,
            language,
            settings,
            client,
            file_cache: file_cache.map(GeminiFileCache::new),
        })
    }

    async fn transcribe_file(&self, audio_path: &Path) -> Result<Transcript> {
        let bytes = tokio::fs::read(audio_path)
            .await
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        let mime_type = audio_mime(audio_path);
        let duration_secs = get_duration(audio_path).await.ok();
        let upload = self.resolve_file(audio_path, &bytes, mime_type).await?;
        let response = self
            .generate_transcript(
                &upload.file.uri,
                mime_type,
                bytes.len() as u64,
                duration_secs,
                &upload,
            )
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

    async fn generate_transcript(
        &self,
        file_uri: &str,
        mime_type: &str,
        input_bytes: u64,
        duration_secs: Option<f64>,
        upload: &GeminiUploadRef,
    ) -> Result<Transcript> {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.api_base_url,
            urlencoding::encode(&self.model)
        );
        let prompt = prompt_text(self.language.as_deref(), duration_secs);
        let cached_content = self
            .resolve_cached_content(upload, file_uri, mime_type)
            .await?;
        let payload = if let Some(cached_content) = &cached_content {
            generate_payload_with_cached_content(&cached_content.name, &prompt)
        } else {
            generate_payload(file_uri, mime_type, &prompt)
        };
        let chunks = self.stream_generate_chunks(&url, &payload).await?;

        let mut transcript = parse_stream_generate_response(
            &chunks,
            &self.model,
            &self.api_base_url,
            mime_type,
            input_bytes,
            duration_secs,
        );
        if let Some(cached_content) = cached_content {
            add_cached_content_metadata(&mut transcript, cached_content);
        }
        Ok(transcript)
    }
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
