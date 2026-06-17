pub(crate) mod limits;
mod types;

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

use crate::audio::wav::encode_wav;
use crate::engines::qwen_filetrans::limits::validate_model_for_path;
use crate::engines::qwen_filetrans::types::{
    FileInput, QueryResponse, RequestParameters, ResultDocument, SubmitRequest, SubmitResponse,
    TaskResult, normalize_api_base_url,
};
use crate::engines::rate_limit::{self, send_with_retry};
use crate::storage::s3::{S3CleanupResult, S3Uploader};
use crate::transcriber::{Transcriber, Transcript};

pub struct QwenFileTrans {
    api_base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    settings: rate_limit::ApiRequestSettings,
    client: Client,
    uploader: S3Uploader,
    autoclean: bool,
    poll_interval: Duration,
    max_polls: u32,
}

impl QwenFileTrans {
    pub fn new(
        api_base_url: String,
        api_key: String,
        model: String,
        language: Option<String>,
        settings: rate_limit::ApiRequestSettings,
        uploader: S3Uploader,
        autoclean: bool,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(settings.request_timeout)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            api_base_url: normalize_api_base_url(&api_base_url),
            api_key,
            model,
            language,
            settings,
            client,
            uploader,
            autoclean,
            poll_interval: Duration::from_secs(2),
            max_polls: 900,
        })
    }

    async fn transcribe_file_url(&self, file_url: String) -> Result<Transcript> {
        let task_id = self.submit_task(file_url).await?;
        let task = self.wait_for_result(&task_id).await?;
        self.download_result(&task.transcription_url, task.metadata)
            .await
    }

    async fn submit_task(&self, file_url: String) -> Result<String> {
        let url = format!("{}/services/audio/asr/transcription", self.api_base_url);
        let payload = SubmitRequest {
            model: self.model.clone(),
            input: FileInput { file_url },
            parameters: RequestParameters {
                channel_id: vec![0],
                enable_itn: false,
                enable_words: true,
                language: self.language.clone(),
            },
        };

        let body = send_with_retry(&self.settings, "Qwen ASR submit", || {
            let client = self.client.clone();
            let api_key = self.api_key.clone();
            let url = url.clone();
            let payload = payload.clone();
            Box::pin(async move {
                client
                    .post(url)
                    .bearer_auth(api_key)
                    .header("X-DashScope-Async", "enable")
                    .json(&payload)
                    .send()
                    .await
                    .context("Failed to submit Qwen ASR task")
            })
        })
        .await
        .map_err(|(status, body)| anyhow::anyhow!("Qwen ASR submit returned {status}: {body}"))?;

        let response: SubmitResponse =
            serde_json::from_slice(&body).context("Failed to parse Qwen ASR submit response")?;

        response
            .output
            .task_id
            .context("Qwen ASR submit response did not include output.task_id")
    }

    async fn wait_for_result(&self, task_id: &str) -> Result<TaskResult> {
        let url = format!("{}/tasks/{}", self.api_base_url, task_id);

        for _ in 0..self.max_polls {
            tokio::time::sleep(self.poll_interval).await;

            let body = send_with_retry(&self.settings, "Qwen ASR task query", || {
                let client = self.client.clone();
                let api_key = self.api_key.clone();
                let url = url.clone();
                Box::pin(async move {
                    client
                        .get(url)
                        .bearer_auth(api_key)
                        .header("X-DashScope-Async", "enable")
                        .send()
                        .await
                        .context("Failed to query Qwen ASR task")
                })
            })
            .await
            .map_err(|(status, body)| {
                anyhow::anyhow!("Qwen ASR task query returned {status}: {body}")
            })?;

            let response: QueryResponse = serde_json::from_slice(&body)
                .context("Failed to parse Qwen ASR task query response")?;

            match response.output.task_status.as_deref() {
                Some("SUCCEEDED") => {
                    let transcription_url = response
                        .output
                        .result
                        .as_ref()
                        .and_then(|result| result.transcription_url.clone())
                        .context("Qwen ASR task succeeded without transcription_url")?;
                    return Ok(response.into_task_result(transcription_url));
                }
                Some("FAILED") | Some("UNKNOWN") => {
                    anyhow::bail!("Qwen ASR task failed: {}", String::from_utf8_lossy(&body));
                }
                _ => {}
            }
        }

        anyhow::bail!(
            "Qwen ASR task did not finish after {} polls",
            self.max_polls
        );
    }

    async fn download_result(
        &self,
        transcription_url: &str,
        task_metadata: serde_json::Value,
    ) -> Result<Transcript> {
        let body = self
            .client
            .get(transcription_url)
            .send()
            .await
            .context("Failed to download Qwen ASR result")?
            .error_for_status()
            .context("Qwen ASR result download returned an error")?
            .bytes()
            .await
            .context("Failed to read Qwen ASR result body")?;

        let result: ResultDocument =
            serde_json::from_slice(&body).context("Failed to parse Qwen ASR result JSON")?;

        Ok(result.into_transcript(&self.model, &self.api_base_url, task_metadata))
    }
}

#[async_trait]
impl Transcriber for QwenFileTrans {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_path(&self, audio_path: &Path) -> Result<Transcript> {
        validate_model_for_path(&self.model, audio_path).await?;
        let upload = self.uploader.upload_and_presign_object(audio_path).await?;
        let mut transcript = self.transcribe_file_url(upload.url.clone()).await?;
        let cleanup = if self.autoclean {
            self.uploader.cleanup_uploaded_object(&upload).await
        } else {
            S3CleanupResult::skipped(&upload)
        };
        if let Some(error) = cleanup.error.as_deref() {
            eprintln!(
                "Failed to delete staged Qwen object s3://{}/{}: {error}",
                cleanup.bucket, cleanup.key
            );
        }
        add_qwen_staging_metadata(&mut transcript, cleanup);
        Ok(transcript)
    }

    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript> {
        let tmp = tempfile::Builder::new()
            .prefix("transcribeit-qwen-")
            .suffix(".wav")
            .tempfile()
            .context("Failed to create temporary WAV file")?;
        tokio::fs::write(tmp.path(), wav_bytes)
            .await
            .context("Failed to write temporary WAV file")?;
        self.transcribe_path(tmp.path()).await
    }
}

fn add_qwen_staging_metadata(transcript: &mut Transcript, cleanup: S3CleanupResult) {
    let metadata = transcript
        .provider_metadata
        .get_or_insert_with(|| serde_json::json!({ "qwen": {} }));
    if let Some(qwen) = metadata.get_mut("qwen").and_then(Value::as_object_mut) {
        qwen.insert(
            "staging".to_string(),
            serde_json::json!({
                "provider": "s3",
                "cleanup": cleanup.to_metadata(),
            }),
        );
    }
}
