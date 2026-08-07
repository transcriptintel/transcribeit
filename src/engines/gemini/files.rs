use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::engines::rate_limit::{
    MAX_ERROR_RESPONSE_BYTES, MAX_RESULT_RESPONSE_BYTES, read_response_limited,
};

use super::GeminiApi;
use super::file_cache::GeminiFileCache;

#[derive(Clone, Deserialize)]
pub(super) struct FileRef {
    pub(super) name: String,
    pub(super) uri: String,
    #[serde(default)]
    pub(super) state: Option<String>,
    #[serde(default, rename = "createTime")]
    pub(super) create_time: Option<String>,
    #[serde(default, rename = "expirationTime")]
    pub(super) expiration_time: Option<String>,
}

pub(super) struct GeminiUploadRef {
    pub(super) file: FileRef,
    pub(super) cache_hash: Option<String>,
    pub(super) cache_index_path: Option<PathBuf>,
    pub(super) cache_reused: bool,
    pub(super) cache_enabled: bool,
}

pub(super) struct GeminiFileCleanup {
    attempted: bool,
    deleted: bool,
    error: Option<String>,
}

pub(super) fn finish_file_operation<T>(
    operation: Result<T>,
    cleanup: &GeminiFileCleanup,
    file_name: &str,
) -> Result<T> {
    if let Some(cleanup_error) = cleanup.error.as_deref() {
        eprintln!("Failed to delete Gemini file {file_name}: {cleanup_error}");
    }

    match operation {
        Err(primary_error) if cleanup.error.is_some() => Err(primary_error.context(format!(
            "Gemini file cleanup also failed for {file_name}: {}",
            cleanup.error.as_deref().unwrap_or("unknown cleanup error")
        ))),
        result => result,
    }
}

impl GeminiApi {
    pub(super) async fn resolve_file(
        &self,
        audio_path: &Path,
        bytes: &[u8],
        mime_type: &str,
    ) -> Result<GeminiUploadRef> {
        let Some(cache) = &self.file_cache else {
            let file = self.upload_file(audio_path, bytes, mime_type, None).await?;
            let file = self.wait_for_active_uploaded_file(file).await?;
            return Ok(GeminiUploadRef {
                file,
                cache_hash: None,
                cache_index_path: None,
                cache_reused: false,
                cache_enabled: false,
            });
        };

        let hash = GeminiFileCache::prepared_hash(bytes);
        if let Some(cached) =
            cache.lookup(&self.api_base_url, &hash, bytes.len() as u64, mime_type)?
        {
            match self.get_file(&cached.file_name).await {
                Ok(file) => {
                    let active_file = self.wait_for_active_file(file).await?;
                    cache.record(
                        &self.api_base_url,
                        &hash,
                        bytes.len() as u64,
                        mime_type,
                        &active_file,
                        true,
                    )?;
                    eprintln!("Reusing Gemini Files API upload {}", active_file.name);
                    return Ok(GeminiUploadRef {
                        file: active_file,
                        cache_hash: Some(hash),
                        cache_index_path: Some(cache.index_path.clone()),
                        cache_reused: true,
                        cache_enabled: true,
                    });
                }
                Err(err) => {
                    eprintln!(
                        "Gemini cached file {} is not reusable ({err:#}); uploading again.",
                        cached.file_name
                    );
                    cache.remove(&hash)?;
                }
            }
        }

        let display_name = GeminiFileCache::display_name(&hash, audio_path);
        let file = self
            .upload_file(audio_path, bytes, mime_type, Some(&display_name))
            .await?;
        let active_file = self.wait_for_active_uploaded_file(file).await?;
        cache.record(
            &self.api_base_url,
            &hash,
            bytes.len() as u64,
            mime_type,
            &active_file,
            false,
        )?;
        Ok(GeminiUploadRef {
            file: active_file,
            cache_hash: Some(hash),
            cache_index_path: Some(cache.index_path.clone()),
            cache_reused: false,
            cache_enabled: true,
        })
    }

    async fn upload_file(
        &self,
        audio_path: &Path,
        bytes: &[u8],
        mime_type: &str,
        display_name: Option<&str>,
    ) -> Result<FileRef> {
        let display_name = display_name.unwrap_or_else(|| {
            audio_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("audio")
        });
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
            let body = read_response_limited(
                start_response,
                MAX_ERROR_RESPONSE_BYTES,
                "Gemini file upload start error response",
            )
            .await
            .map(|body| String::from_utf8_lossy(&body).into_owned())
            .unwrap_or_else(|error| error.to_string());
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
        let maximum_bytes = if status.is_success() {
            MAX_RESULT_RESPONSE_BYTES
        } else {
            MAX_ERROR_RESPONSE_BYTES
        };
        let body = read_response_limited(
            upload_response,
            maximum_bytes,
            "Gemini file upload response",
        )
        .await?;
        if !status.is_success() {
            anyhow::bail!(
                "Gemini file upload returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }

        parse_file_response(&body, "Gemini file upload response")
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

    async fn wait_for_active_uploaded_file(&self, file: FileRef) -> Result<FileRef> {
        let file_name = file.name.clone();
        match self.wait_for_active_file(file).await {
            Ok(active) => Ok(active),
            Err(primary_error) => match self.delete_file(&file_name).await {
                Ok(()) => Err(primary_error.context(format!(
                    "new Gemini upload {file_name} was deleted after processing failed"
                ))),
                Err(cleanup_error) => Err(primary_error.context(format!(
                    "cleanup of failed Gemini upload {file_name} also failed: {cleanup_error:#}"
                ))),
            },
        }
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
        let maximum_bytes = if status.is_success() {
            MAX_RESULT_RESPONSE_BYTES
        } else {
            MAX_ERROR_RESPONSE_BYTES
        };
        let body =
            read_response_limited(response, maximum_bytes, "Gemini file metadata response").await?;
        if !status.is_success() {
            anyhow::bail!(
                "Gemini file metadata returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        parse_file_response(&body, "Gemini file metadata response")
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
            let body = read_response_limited(
                response,
                MAX_ERROR_RESPONSE_BYTES,
                "Gemini file delete error response",
            )
            .await
            .map(|body| String::from_utf8_lossy(&body).into_owned())
            .unwrap_or_else(|error| error.to_string());
            anyhow::bail!("Gemini file delete returned {status}: {body}");
        }
    }

    pub(super) async fn cleanup_file_after_run(
        &self,
        upload: &GeminiUploadRef,
    ) -> GeminiFileCleanup {
        let should_delete = self
            .file_cache
            .as_ref()
            .map(|cache| cache.autoclean)
            .unwrap_or(true);

        if !should_delete {
            return GeminiFileCleanup {
                attempted: false,
                deleted: false,
                error: None,
            };
        }

        let delete_result = self.delete_file(&upload.file.name).await;
        let deleted = delete_result.is_ok();
        if deleted
            && let (Some(cache), Some(hash)) = (&self.file_cache, upload.cache_hash.as_deref())
            && let Err(err) = cache.remove(hash)
        {
            return GeminiFileCleanup {
                attempted: true,
                deleted,
                error: Some(err.to_string()),
            };
        }
        GeminiFileCleanup {
            attempted: true,
            deleted,
            error: delete_result.err().map(|err| err.to_string()),
        }
    }
}

fn parse_file_response(body: &[u8], context: &str) -> Result<FileRef> {
    let value: Value =
        serde_json::from_slice(body).with_context(|| format!("Failed to parse {context}"))?;
    if let Some(file) = value.get("file") {
        return serde_json::from_value(file.clone())
            .with_context(|| format!("{context} did not include valid wrapped file metadata"));
    }
    serde_json::from_value(value)
        .with_context(|| format!("{context} did not include valid file metadata"))
}

pub(super) fn with_file_cleanup_metadata(
    metadata: Option<Value>,
    upload: GeminiUploadRef,
    mime_type: &str,
    input_bytes: u64,
    cleanup: GeminiFileCleanup,
) -> Value {
    let mut metadata = metadata.unwrap_or_else(|| json!({ "gemini": {} }));
    if let Some(gemini) = metadata.get_mut("gemini").and_then(Value::as_object_mut) {
        gemini.insert(
            "file".to_string(),
            json!({
                "name": upload.file.name,
                "mime_type": mime_type,
                "bytes": input_bytes,
                "deleted": cleanup.deleted,
                "delete_attempted": cleanup.attempted,
                "delete_error": cleanup.error,
                "cache_enabled": upload.cache_enabled,
                "cache_reused": upload.cache_reused,
                "cache_hash": upload.cache_hash,
                "cache_index_path": upload
                    .cache_index_path
                    .map(|path| path.display().to_string()),
            }),
        );
    }
    metadata
}

#[cfg(test)]
mod tests {
    use super::{GeminiFileCleanup, finish_file_operation, parse_file_response};

    #[test]
    fn parse_file_response_accepts_upload_wrapper() {
        let file = parse_file_response(
            br#"{"file":{"name":"files/abc","uri":"https://example.test/files/abc","state":"ACTIVE"}}"#,
            "test",
        )
        .expect("wrapped file should parse");

        assert_eq!(file.name, "files/abc");
        assert_eq!(file.state.as_deref(), Some("ACTIVE"));
    }

    #[test]
    fn parse_file_response_accepts_direct_file_metadata() {
        let file = parse_file_response(
            br#"{"name":"files/abc","uri":"https://example.test/files/abc","state":"ACTIVE"}"#,
            "test",
        )
        .expect("direct file should parse");

        assert_eq!(file.name, "files/abc");
        assert_eq!(file.uri, "https://example.test/files/abc");
    }

    #[test]
    fn failed_generation_preserves_cleanup_failure() {
        let cleanup = GeminiFileCleanup {
            attempted: true,
            deleted: false,
            error: Some("delete failed".to_string()),
        };
        let error = finish_file_operation::<()>(
            Err(anyhow::anyhow!("generation failed")),
            &cleanup,
            "files/abc",
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("generation failed"));
        assert!(message.contains("delete failed"));
    }
}
