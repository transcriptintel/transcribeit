use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::transcriber::Transcript;

use super::GeminiApi;
use super::file_cache::{CachedGeminiContent, CachedGeminiContentParts};
use super::files::GeminiUploadRef;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CachedContentResponse {
    name: String,
    model: Option<String>,
    display_name: Option<String>,
    create_time: Option<String>,
    update_time: Option<String>,
    expire_time: Option<String>,
    usage_metadata: Option<Value>,
}

impl CachedContentResponse {
    fn into_cached_content(self, ttl_secs: u64, reused: bool) -> CachedGeminiContent {
        CachedGeminiContent::new(CachedGeminiContentParts {
            name: self.name,
            model: self.model.unwrap_or_default(),
            display_name: self.display_name,
            ttl_secs,
            gemini_create_time: self.create_time,
            gemini_update_time: self.update_time,
            gemini_expire_time: self.expire_time,
            usage_metadata: self.usage_metadata,
            reused,
        })
    }
}

impl GeminiApi {
    pub(super) async fn resolve_cached_content(
        &self,
        upload: &GeminiUploadRef,
        file_uri: &str,
        mime_type: &str,
    ) -> Result<Option<CachedGeminiContent>> {
        let Some(cache) = &self.file_cache else {
            return Ok(None);
        };
        if !cache.explicit_cache {
            return Ok(None);
        }
        let Some(hash) = upload.cache_hash.as_deref() else {
            return Ok(None);
        };

        if let Some(cached_content) = cache.lookup_cached_content(hash, &self.model)? {
            match self.get_cached_content(&cached_content.name).await {
                Ok(response) => {
                    let cached_content = cached_content.mark_reused(response.usage_metadata);
                    cache.record_cached_content(hash, &self.model, cached_content.clone())?;
                    eprintln!("Reusing Gemini cachedContent {}", cached_content.name);
                    return Ok(Some(cached_content));
                }
                Err(err) => {
                    eprintln!(
                        "Gemini cachedContent {} is not reusable ({err:#}); creating again.",
                        cached_content.name
                    );
                    cache.remove_cached_content(hash, &self.model)?;
                }
            }
        }

        let cached_content = self
            .create_cached_content(hash, file_uri, mime_type, cache.explicit_cache_ttl_secs)
            .await?;
        cache.record_cached_content(hash, &self.model, cached_content.clone())?;
        Ok(Some(cached_content))
    }

    async fn create_cached_content(
        &self,
        hash: &str,
        file_uri: &str,
        mime_type: &str,
        ttl_secs: u64,
    ) -> Result<CachedGeminiContent> {
        let url = format!("{}/cachedContents", self.api_base_url);
        let display_name = format!("transcribeit-{hash}");
        let response = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&json!({
                "model": model_resource_name(&self.model),
                "displayName": display_name,
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri
                        }
                    }]
                }],
                "ttl": format!("{ttl_secs}s")
            }))
            .send()
            .await
            .context("Failed to create Gemini cachedContent")?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .context("Failed to read Gemini cachedContent response")?;
        if !status.is_success() {
            anyhow::bail!(
                "Gemini cachedContent create returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        let response = parse_cached_content_response(&body, "Gemini cachedContent create")?;
        Ok(response.into_cached_content(ttl_secs, false))
    }

    async fn get_cached_content(&self, name: &str) -> Result<CachedContentResponse> {
        let url = format!("{}/{}", self.api_base_url, name);
        let response = self
            .client
            .get(url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .context("Failed to get Gemini cachedContent")?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .context("Failed to read Gemini cachedContent metadata response")?;
        if !status.is_success() {
            anyhow::bail!(
                "Gemini cachedContent metadata returned {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        parse_cached_content_response(&body, "Gemini cachedContent metadata")
    }
}

fn parse_cached_content_response(body: &[u8], context: &str) -> Result<CachedContentResponse> {
    serde_json::from_slice(body).with_context(|| format!("Failed to parse {context}"))
}

fn model_resource_name(model: &str) -> String {
    if model.starts_with("models/") {
        model.to_string()
    } else {
        format!("models/{model}")
    }
}

pub(super) fn add_cached_content_metadata(
    transcript: &mut Transcript,
    cached_content: CachedGeminiContent,
) {
    let metadata = transcript
        .provider_metadata
        .get_or_insert_with(|| json!({ "gemini": {} }));
    if let Some(gemini) = metadata.get_mut("gemini").and_then(Value::as_object_mut) {
        gemini.insert(
            "cached_content".to_string(),
            json!({
                "enabled": true,
                "reused": cached_content.reused,
                "name": cached_content.name,
                "model": cached_content.model,
                "display_name": cached_content.display_name,
                "ttl_secs": cached_content.ttl_secs,
                "expected_expires_at_unix": cached_content.expected_expires_at_unix,
                "gemini_create_time": cached_content.gemini_create_time,
                "gemini_update_time": cached_content.gemini_update_time,
                "gemini_expire_time": cached_content.gemini_expire_time,
                "usage_metadata": cached_content.usage_metadata,
            }),
        );
    }
}
