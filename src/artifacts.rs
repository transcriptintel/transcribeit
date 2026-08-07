use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::ProgressBar;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone, Copy)]
pub(crate) struct ArtifactIntegrity {
    pub sha256: &'static str,
    pub size_bytes: u64,
}

pub(crate) async fn write_verified_response(
    response: reqwest::Response,
    destination: &Path,
    progress: &ProgressBar,
    integrity: ArtifactIntegrity,
) -> Result<()> {
    if let Some(content_length) = response.content_length() {
        anyhow::ensure!(
            content_length == integrity.size_bytes,
            "artifact Content-Length mismatch: expected {} bytes, received {content_length}",
            integrity.size_bytes
        );
    }

    let mut file = tokio::fs::File::create(destination)
        .await
        .with_context(|| format!("Failed to create {}", destination.display()))?;
    let mut stream = response.bytes_stream();
    let mut hasher = Sha256::new();
    let mut written = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading download stream")?;
        written = written
            .checked_add(chunk.len() as u64)
            .context("artifact size overflow")?;
        anyhow::ensure!(
            written <= integrity.size_bytes,
            "artifact exceeded expected size of {} bytes",
            integrity.size_bytes
        );
        hasher.update(&chunk);
        file.write_all(&chunk)
            .await
            .context("Failed to write artifact")?;
        progress.inc(chunk.len() as u64);
    }
    file.flush().await?;
    file.sync_all().await?;

    verify_digest(written, &digest_hex(hasher.finalize().as_ref()), integrity)
}

pub(crate) async fn verify_file(path: &Path, integrity: ArtifactIntegrity) -> Result<()> {
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .context("artifact size overflow")?;
        hasher.update(&buffer[..read]);
    }

    verify_digest(size, &digest_hex(hasher.finalize().as_ref()), integrity)
        .with_context(|| format!("Integrity check failed for {}", path.display()))
}

pub(crate) fn verify_file_blocking(path: &Path, integrity: ArtifactIntegrity) -> Result<()> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .context("artifact size overflow")?;
        hasher.update(&buffer[..read]);
    }

    verify_digest(size, &digest_hex(hasher.finalize().as_ref()), integrity)
        .with_context(|| format!("Integrity check failed for {}", path.display()))
}

fn verify_digest(size: u64, digest: &str, integrity: ArtifactIntegrity) -> Result<()> {
    anyhow::ensure!(
        size == integrity.size_bytes,
        "artifact size mismatch: expected {} bytes, received {size}",
        integrity.size_bytes
    );
    anyhow::ensure!(
        digest.eq_ignore_ascii_case(integrity.sha256),
        "artifact SHA-256 mismatch: expected {}, received {digest}",
        integrity.sha256
    );
    Ok(())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    digest_hex(hasher.finalize().as_ref())
}

fn digest_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a string cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn verify_file_rejects_wrong_digest_and_size() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        tokio::fs::write(temp.path(), b"artifact").await.unwrap();
        assert!(
            verify_file(
                temp.path(),
                ArtifactIntegrity {
                    sha256: "0000000000000000000000000000000000000000000000000000000000000000",
                    size_bytes: 8,
                },
            )
            .await
            .is_err()
        );
    }
}
