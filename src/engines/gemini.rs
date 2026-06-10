use std::path::Path;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::audio::segment::get_duration;
use crate::audio::wav::encode_wav;
use crate::engines::rate_limit::{self, send_with_retry};
use crate::transcriber::{Transcriber, Transcript};

mod response;
mod schema;

use response::parse_generate_response;
use schema::{audio_mime, generate_payload, prompt_text, upload_base_url};

pub struct GeminiApi {
    api_base_url: String,
    upload_base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    settings: rate_limit::ApiRequestSettings,
    client: Client,
}

impl GeminiApi {
    pub fn new(
        api_base_url: String,
        api_key: String,
        model: String,
        language: Option<String>,
        settings: rate_limit::ApiRequestSettings,
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
        })
    }

    async fn transcribe_file(&self, audio_path: &Path) -> Result<Transcript> {
        let bytes = tokio::fs::read(audio_path)
            .await
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        let mime_type = audio_mime(audio_path);
        let duration_secs = get_duration(audio_path).await.ok();
        let file = self.upload_file(audio_path, &bytes, mime_type).await?;
        let active_file = self.wait_for_active_file(file).await?;
        let response = self
            .generate_transcript(
                &active_file.uri,
                mime_type,
                bytes.len() as u64,
                duration_secs,
            )
            .await;
        let delete_result = self.delete_file(&active_file.name).await;

        let mut transcript = response?;
        transcript.provider_metadata = Some(with_file_cleanup_metadata(
            transcript.provider_metadata.take(),
            active_file.name,
            mime_type,
            bytes.len() as u64,
            delete_result,
        ));
        Ok(transcript)
    }

    async fn upload_file(
        &self,
        audio_path: &Path,
        bytes: &[u8],
        mime_type: &str,
    ) -> Result<FileRef> {
        let display_name = audio_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("audio");
        let start_url = format!("{}/files", self.upload_base_url);
        let start_response = self
            .client
            .post(start_url)
            .header("x-goog-api-key", &self.api_key)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header(
                "X-Goog-Upload-Header-Content-Length",
                bytes.len().to_string(),
            )
            .header("X-Goog-Upload-Header-Content-Type", mime_type)
            .json(&json!({ "file": { "display_name": display_name } }))
            .send()
            .await
            .context("Failed to start Gemini file upload")?;

        let status = start_response.status();
        if !status.is_success() {
            let body = start_response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini file upload start returned {status}: {body}");
        }

        let upload_url = start_response
            .headers()
            .get("x-goog-upload-url")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned)
            .context("Gemini file upload start response did not include x-goog-upload-url")?;

        let upload_response = self
            .client
            .post(upload_url)
            .header("Content-Length", bytes.len().to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(bytes.to_vec())
            .send()
            .await
            .context("Failed to upload audio bytes to Gemini Files API")?;

        let status = upload_response.status();
        let body = upload_response
            .bytes()
            .await
            .context("Failed to read Gemini file upload response")?;
        if !status.is_success() {
            anyhow::bail!(
                "Gemini file upload returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }

        let response: FileResponse =
            serde_json::from_slice(&body).context("Failed to parse Gemini file upload response")?;
        response
            .file
            .context("Gemini file upload response did not include file metadata")
    }

    async fn wait_for_active_file(&self, mut file: FileRef) -> Result<FileRef> {
        for _ in 0..60 {
            match file.state.as_deref() {
                None | Some("ACTIVE") => return Ok(file),
                Some("FAILED") => anyhow::bail!("Gemini file processing failed for {}", file.name),
                _ => {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    file = self.get_file(&file.name).await?;
                }
            }
        }

        anyhow::bail!("Gemini file did not become ACTIVE within 60 seconds");
    }

    async fn get_file(&self, name: &str) -> Result<FileRef> {
        let url = format!("{}/{}", self.api_base_url, name);
        let response = self
            .client
            .get(url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .context("Failed to get Gemini file metadata")?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .context("Failed to read Gemini file metadata response")?;
        if !status.is_success() {
            anyhow::bail!(
                "Gemini file metadata returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        let response: FileResponse =
            serde_json::from_slice(&body).context("Failed to parse Gemini file metadata")?;
        response
            .file
            .context("Gemini file metadata response did not include file")
    }

    async fn generate_transcript(
        &self,
        file_uri: &str,
        mime_type: &str,
        input_bytes: u64,
        duration_secs: Option<f64>,
    ) -> Result<Transcript> {
        let url = format!(
            "{}/models/{}:generateContent",
            self.api_base_url,
            urlencoding::encode(&self.model)
        );
        let prompt = prompt_text(self.language.as_deref(), duration_secs);
        let payload = generate_payload(file_uri, mime_type, &prompt);
        let body = send_with_retry(&self.settings, "Gemini generateContent", || {
            let client = self.client.clone();
            let api_key = self.api_key.clone();
            let url = url.clone();
            let payload = payload.clone();
            Box::pin(async move {
                client
                    .post(url)
                    .header("x-goog-api-key", api_key)
                    .json(&payload)
                    .send()
                    .await
                    .context("Failed to send request to Gemini generateContent")
            })
        })
        .await
        .map_err(|(status, body)| {
            anyhow::anyhow!("Gemini generateContent returned {status}: {body}")
        })?;

        Ok(parse_generate_response(
            &body,
            &self.model,
            &self.api_base_url,
            mime_type,
            input_bytes,
            duration_secs,
        ))
    }

    async fn delete_file(&self, name: &str) -> Result<()> {
        let url = format!("{}/{}", self.api_base_url, name);
        let response = self
            .client
            .delete(url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .context("Failed to delete Gemini file")?;
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini file delete returned {status}: {body}");
        }
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

#[derive(Clone, Deserialize)]
struct FileResponse {
    file: Option<FileRef>,
}

#[derive(Clone, Deserialize)]
struct FileRef {
    name: String,
    uri: String,
    #[serde(default)]
    state: Option<String>,
}

fn with_file_cleanup_metadata(
    metadata: Option<Value>,
    file_name: String,
    mime_type: &str,
    input_bytes: u64,
    delete_result: Result<()>,
) -> Value {
    let mut metadata = metadata.unwrap_or_else(|| json!({ "gemini": {} }));
    if let Some(gemini) = metadata.get_mut("gemini").and_then(Value::as_object_mut) {
        gemini.insert(
            "file".to_string(),
            json!({
                "name": file_name,
                "mime_type": mime_type,
                "bytes": input_bytes,
                "deleted": delete_result.is_ok(),
                "delete_error": delete_result.err().map(|err| err.to_string()),
            }),
        );
    }
    metadata
}
