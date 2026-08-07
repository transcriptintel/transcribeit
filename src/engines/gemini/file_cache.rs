use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
        crate::artifacts::sha256_hex(bytes)
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
        self.with_locked_index(|index| {
            let entry = index
                .entries
                .get(hash)
                .filter(|entry| {
                    entry.api_base_url == api_base_url
                        && entry.bytes == bytes
                        && entry.mime_type == mime_type
                        && !entry.is_expired()
                })
                .cloned();
            Ok((entry, false))
        })
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
        self.with_locked_index(|index| {
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
            Ok(((), true))
        })
    }

    pub fn remove(&self, hash: &str) -> Result<()> {
        self.with_locked_index(|index| {
            let changed = index.entries.remove(hash).is_some();
            Ok(((), changed))
        })
    }

    pub fn lookup_cached_content(
        &self,
        hash: &str,
        model: &str,
    ) -> Result<Option<CachedGeminiContent>> {
        self.with_locked_index(|index| {
            let cached_content = index
                .entries
                .get(hash)
                .and_then(|entry| entry.cached_contents.get(model))
                .filter(|cached_content| !cached_content.is_expired())
                .cloned();
            Ok((cached_content, false))
        })
    }

    pub fn record_cached_content(
        &self,
        hash: &str,
        model: &str,
        cached_content: CachedGeminiContent,
    ) -> Result<()> {
        self.with_locked_index(|index| {
            let entry = index
                .entries
                .get_mut(hash)
                .with_context(|| format!("Gemini file cache entry missing for hash {hash}"))?;
            entry
                .cached_contents
                .insert(model.to_string(), cached_content);
            Ok(((), true))
        })
    }

    pub fn remove_cached_content(&self, hash: &str, model: &str) -> Result<()> {
        self.with_locked_index(|index| {
            let changed = index
                .entries
                .get_mut(hash)
                .and_then(|entry| entry.cached_contents.remove(model))
                .is_some();
            Ok(((), changed))
        })
    }

    fn with_locked_index<T>(
        &self,
        operation: impl FnOnce(&mut GeminiFileCacheIndex) -> Result<(T, bool)>,
    ) -> Result<T> {
        if let Some(parent) = self.index_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let lock_path = self.lock_path();
        let lock_file = open_private_lock_file(&lock_path)?;
        lock_file
            .lock()
            .with_context(|| format!("Failed to lock {}", lock_path.display()))?;

        let (mut index, recovered) = self.read_index_unlocked()?;
        let (result, changed) = operation(&mut index)?;
        if recovered || changed {
            self.write_index_unlocked(&index)?;
        }
        File::unlock(&lock_file)
            .with_context(|| format!("Failed to unlock {}", lock_path.display()))?;
        Ok(result)
    }

    fn read_index_unlocked(&self) -> Result<(GeminiFileCacheIndex, bool)> {
        if !self.index_path.exists() {
            return Ok((GeminiFileCacheIndex::default(), false));
        }
        let bytes = std::fs::read(&self.index_path)
            .with_context(|| format!("Failed to read {}", self.index_path.display()))?;
        match serde_json::from_slice(&bytes) {
            Ok(index) => Ok((index, false)),
            Err(parse_error) => {
                let corrupt_path = self.corrupt_index_path();
                std::fs::rename(&self.index_path, &corrupt_path).with_context(|| {
                    format!(
                        "Failed to preserve corrupt Gemini cache index {} as {}",
                        self.index_path.display(),
                        corrupt_path.display()
                    )
                })?;
                eprintln!(
                    "Gemini cache index {} was corrupt ({parse_error}); preserved it as {} and started a new index.",
                    self.index_path.display(),
                    corrupt_path.display()
                );
                Ok((GeminiFileCacheIndex::default(), true))
            }
        }
    }

    fn write_index_unlocked(&self, index: &GeminiFileCacheIndex) -> Result<()> {
        let parent = self.index_path.parent().unwrap_or_else(|| Path::new("."));
        let bytes = serde_json::to_vec_pretty(index).context("Failed to serialize Gemini cache")?;
        let mut temporary = tempfile::Builder::new()
            .prefix(".gemini-files-")
            .tempfile_in(parent)
            .with_context(|| format!("Failed to create temporary file in {}", parent.display()))?;
        set_private_permissions(temporary.as_file())?;
        temporary
            .write_all(&bytes)
            .context("Failed to write temporary Gemini cache index")?;
        temporary
            .as_file_mut()
            .sync_all()
            .context("Failed to sync temporary Gemini cache index")?;
        temporary.persist(&self.index_path).map_err(|error| {
            anyhow::anyhow!(
                "Failed to atomically replace {}: {}",
                self.index_path.display(),
                error.error
            )
        })?;
        Ok(())
    }

    fn lock_path(&self) -> PathBuf {
        let mut path = OsString::from(self.index_path.as_os_str());
        path.push(".lock");
        PathBuf::from(path)
    }

    fn corrupt_index_path(&self) -> PathBuf {
        let mut path = OsString::from(self.index_path.as_os_str());
        path.push(format!(
            ".corrupt-{}-{}-{}",
            now_unix(),
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        PathBuf::from(path)
    }
}

fn open_private_lock_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .with_context(|| format!("Failed to open Gemini cache lock {}", path.display()))?;
    set_private_permissions(&file)?;
    Ok(file)
}

fn set_private_permissions(#[allow(unused_variables)] file: &File) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .context("Failed to set private Gemini cache permissions")?;
    }
    Ok(())
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

    fn cache_at(index_path: PathBuf) -> GeminiFileCache {
        GeminiFileCache::new(GeminiFileCacheConfig {
            index_path: Some(index_path),
            autoclean: false,
            explicit_cache: false,
            explicit_cache_ttl_secs: 60,
        })
    }

    fn file_ref(name: &str) -> FileRef {
        FileRef {
            name: format!("files/{name}"),
            uri: format!("https://example.test/files/{name}"),
            state: Some("ACTIVE".to_string()),
            create_time: None,
            expiration_time: None,
        }
    }

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

    #[test]
    fn corrupt_index_is_quarantined_and_recreated() {
        let directory = tempfile::tempdir().unwrap();
        let index_path = directory.path().join("gemini-files.json");
        std::fs::write(&index_path, b"not json").unwrap();
        let cache = cache_at(index_path.clone());

        assert!(
            cache
                .lookup("https://example.test", "missing", 1, "audio/wav")
                .unwrap()
                .is_none()
        );
        let recreated: GeminiFileCacheIndex =
            serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
        assert!(recreated.entries.is_empty());
        assert!(std::fs::read_dir(directory.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("gemini-files.json.corrupt-")
        }));
    }

    #[test]
    fn concurrent_records_do_not_lose_entries() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache_at(directory.path().join("gemini-files.json"));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));

        std::thread::scope(|scope| {
            for index in 0..8 {
                let cache = cache.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    let hash = format!("hash-{index}");
                    cache
                        .record(
                            "https://example.test",
                            &hash,
                            index + 1,
                            "audio/wav",
                            &file_ref(&index.to_string()),
                            false,
                        )
                        .unwrap();
                });
            }
        });

        let (index, _) = cache.read_index_unlocked().unwrap();
        assert_eq!(index.entries.len(), 8);
    }

    #[cfg(unix)]
    #[test]
    fn cache_index_and_lock_are_private() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let index_path = directory.path().join("gemini-files.json");
        let cache = cache_at(index_path.clone());
        cache
            .record(
                "https://example.test",
                "hash",
                1,
                "audio/wav",
                &file_ref("private"),
                false,
            )
            .unwrap();

        assert_eq!(
            std::fs::metadata(index_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(cache.lock_path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}
