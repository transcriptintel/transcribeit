use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::files::FileRef;

const GEMINI_FILE_TTL_SECS: u64 = 48 * 60 * 60;

#[derive(Clone, Debug)]
pub struct GeminiFileCacheConfig {
    pub index_path: Option<PathBuf>,
    pub autoclean: bool,
    pub explicit_cache: bool,
    pub explicit_cache_ttl_secs: u64,
}

#[derive(Clone, Debug)]
pub struct GeminiFileCache {
    pub index_path: PathBuf,
    pub autoclean: bool,
    pub explicit_cache: bool,
    pub explicit_cache_ttl_secs: u64,
}

impl GeminiFileCache {
    pub fn new(config: GeminiFileCacheConfig) -> Self {
        Self {
            index_path: config.index_path.unwrap_or_else(default_index_path),
            autoclean: config.autoclean,
            explicit_cache: config.explicit_cache,
            explicit_cache_ttl_secs: config.explicit_cache_ttl_secs,
        }
    }

    pub fn prepared_hash(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    pub fn display_name(hash: &str, audio_path: &Path) -> String {
        let extension = audio_path
            .extension()
            .and_then(|extension| extension.to_str())
            .filter(|extension| !extension.is_empty())
            .unwrap_or("bin");
        format!("transcribeit-{hash}.{extension}")
    }

    pub fn lookup(
        &self,
        api_base_url: &str,
        hash: &str,
        bytes: u64,
        mime_type: &str,
    ) -> Result<Option<CachedGeminiFile>> {
        let index = self.read_index()?;
        Ok(index
            .entries
            .get(hash)
            .filter(|entry| {
                entry.api_base_url == api_base_url
                    && entry.bytes == bytes
                    && entry.mime_type == mime_type
                    && !entry.is_expired()
            })
            .cloned())
    }

    pub fn record(
        &self,
        api_base_url: &str,
        hash: &str,
        bytes: u64,
        mime_type: &str,
        file: &FileRef,
        reused: bool,
    ) -> Result<()> {
        let mut index = self.read_index()?;
        index.entries.insert(
            hash.to_string(),
            CachedGeminiFile {
                hash: hash.to_string(),
                api_base_url: api_base_url.to_string(),
                file_name: file.name.clone(),
                file_uri: file.uri.clone(),
                mime_type: mime_type.to_string(),
                bytes,
                created_at_unix: index
                    .entries
                    .get(hash)
                    .map(|entry| entry.created_at_unix)
                    .unwrap_or_else(now_unix),
                last_seen_at_unix: now_unix(),
                expected_expires_at_unix: now_unix() + GEMINI_FILE_TTL_SECS,
                gemini_create_time: file.create_time.clone(),
                gemini_expiration_time: file.expiration_time.clone(),
                reused,
                cached_contents: index
                    .entries
                    .get(hash)
                    .map(|entry| entry.cached_contents.clone())
                    .unwrap_or_default(),
            },
        );
        self.write_index(&index)
    }

    pub fn remove(&self, hash: &str) -> Result<()> {
        let mut index = self.read_index()?;
        if index.entries.remove(hash).is_some() {
            self.write_index(&index)?;
        }
        Ok(())
    }

    pub fn lookup_cached_content(
        &self,
        hash: &str,
        model: &str,
    ) -> Result<Option<CachedGeminiContent>> {
        let index = self.read_index()?;
        Ok(index
            .entries
            .get(hash)
            .and_then(|entry| entry.cached_contents.get(model))
            .filter(|cached_content| !cached_content.is_expired())
            .cloned())
    }

    pub fn record_cached_content(
        &self,
        hash: &str,
        model: &str,
        cached_content: CachedGeminiContent,
    ) -> Result<()> {
        let mut index = self.read_index()?;
        let entry = index
            .entries
            .get_mut(hash)
            .with_context(|| format!("Gemini file cache entry missing for hash {hash}"))?;
        entry
            .cached_contents
            .insert(model.to_string(), cached_content);
        self.write_index(&index)
    }

    pub fn remove_cached_content(&self, hash: &str, model: &str) -> Result<()> {
        let mut index = self.read_index()?;
        if let Some(entry) = index.entries.get_mut(hash) {
            entry.cached_contents.remove(model);
            self.write_index(&index)?;
        }
        Ok(())
    }

    fn read_index(&self) -> Result<GeminiFileCacheIndex> {
        if !self.index_path.exists() {
            return Ok(GeminiFileCacheIndex::default());
        }
        let bytes = std::fs::read(&self.index_path)
            .with_context(|| format!("Failed to read {}", self.index_path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("Failed to parse {}", self.index_path.display()))
    }

    fn write_index(&self, index: &GeminiFileCacheIndex) -> Result<()> {
        if let Some(parent) = self.index_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let bytes = serde_json::to_vec_pretty(index).context("Failed to serialize Gemini cache")?;
        std::fs::write(&self.index_path, bytes)
            .with_context(|| format!("Failed to write {}", self.index_path.display()))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CachedGeminiFile {
    pub hash: String,
    pub api_base_url: String,
    pub file_name: String,
    pub file_uri: String,
    pub mime_type: String,
    pub bytes: u64,
    pub created_at_unix: u64,
    pub last_seen_at_unix: u64,
    pub expected_expires_at_unix: u64,
    pub gemini_create_time: Option<String>,
    pub gemini_expiration_time: Option<String>,
    pub reused: bool,
    #[serde(default)]
    pub cached_contents: BTreeMap<String, CachedGeminiContent>,
}

impl CachedGeminiFile {
    fn is_expired(&self) -> bool {
        self.expected_expires_at_unix <= now_unix()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CachedGeminiContent {
    pub name: String,
    pub model: String,
    pub display_name: Option<String>,
    pub ttl_secs: u64,
    pub created_at_unix: u64,
    pub last_seen_at_unix: u64,
    pub expected_expires_at_unix: u64,
    pub gemini_create_time: Option<String>,
    pub gemini_update_time: Option<String>,
    pub gemini_expire_time: Option<String>,
    pub usage_metadata: Option<serde_json::Value>,
    pub reused: bool,
}

impl CachedGeminiContent {
    pub fn new(parts: CachedGeminiContentParts) -> Self {
        let now = now_unix();
        Self {
            name: parts.name,
            model: parts.model,
            display_name: parts.display_name,
            ttl_secs: parts.ttl_secs,
            created_at_unix: now,
            last_seen_at_unix: now,
            expected_expires_at_unix: now + parts.ttl_secs,
            gemini_create_time: parts.gemini_create_time,
            gemini_update_time: parts.gemini_update_time,
            gemini_expire_time: parts.gemini_expire_time,
            usage_metadata: parts.usage_metadata,
            reused: parts.reused,
        }
    }

    pub fn mark_reused(mut self, usage_metadata: Option<serde_json::Value>) -> Self {
        self.last_seen_at_unix = now_unix();
        self.usage_metadata = usage_metadata.or(self.usage_metadata);
        self.reused = true;
        self
    }

    fn is_expired(&self) -> bool {
        self.expected_expires_at_unix <= now_unix()
    }
}

pub struct CachedGeminiContentParts {
    pub name: String,
    pub model: String,
    pub display_name: Option<String>,
    pub ttl_secs: u64,
    pub gemini_create_time: Option<String>,
    pub gemini_update_time: Option<String>,
    pub gemini_expire_time: Option<String>,
    pub usage_metadata: Option<serde_json::Value>,
    pub reused: bool,
}

#[derive(Default, Deserialize, Serialize)]
struct GeminiFileCacheIndex {
    entries: BTreeMap<String, CachedGeminiFile>,
}

fn default_index_path() -> PathBuf {
    let model_cache = std::env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| ".cache".to_string());
    PathBuf::from(model_cache).join("transcribeit/gemini-files.json")
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_hash_uses_file_bytes() {
        assert_eq!(
            GeminiFileCache::prepared_hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn display_name_keeps_extension() {
        assert_eq!(
            GeminiFileCache::display_name("abc123", Path::new("/tmp/audio.mp3")),
            "transcribeit-abc123.mp3"
        );
    }
}
