use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::ProgressBar;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const TREE_DIGEST_FILE: &str = ".transcribeit-tree.sha256";

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

pub(crate) async fn extract_verified_archive(
    archive_path: &Path,
    destination_parent: &Path,
    destination_name: &Path,
) -> Result<()> {
    let archive_path = archive_path.to_path_buf();
    let destination_parent = destination_parent.to_path_buf();
    let destination_name = destination_name.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let staging = tempfile::Builder::new()
            .prefix(".transcribeit-extract-")
            .tempdir_in(&destination_parent)
            .context("Failed to create archive staging directory")?;
        let file = std::fs::File::open(&archive_path).context("Failed to open archive")?;
        let decoder = bzip2::read::BzDecoder::new(file);
        let mut archive = tar::Archive::new(decoder);
        archive
            .unpack(staging.path())
            .context("Failed to extract archive into staging")?;

        let staged_directory = staging.path().join(&destination_name);
        anyhow::ensure!(
            staged_directory.is_dir(),
            "archive did not contain expected directory {}",
            destination_name.display()
        );
        write_tree_digest(&staged_directory)?;
        let destination = destination_parent.join(&destination_name);
        std::fs::rename(&staged_directory, &destination).with_context(|| {
            format!(
                "Failed to atomically install {} to {}",
                staged_directory.display(),
                destination.display()
            )
        })?;
        Ok::<(), anyhow::Error>(())
    })
    .await?
}

pub(crate) fn verify_installed_directory(directory: &Path) -> Result<()> {
    let marker = directory.join(TREE_DIGEST_FILE);
    let expected = std::fs::read_to_string(&marker).with_context(|| {
        format!(
            "Installed artifact {} has no integrity marker; remove it and run setup again",
            directory.display()
        )
    })?;
    let actual = tree_digest(directory)?;
    anyhow::ensure!(
        expected.trim().eq_ignore_ascii_case(&actual),
        "Installed artifact integrity check failed for {}",
        directory.display()
    );
    Ok(())
}

pub(crate) fn seal_installed_directory(directory: &Path) -> Result<()> {
    write_tree_digest(directory)
}

fn write_tree_digest(directory: &Path) -> Result<()> {
    let digest = tree_digest(directory)?;
    std::fs::write(directory.join(TREE_DIGEST_FILE), format!("{digest}\n"))?;
    Ok(())
}

fn tree_digest(directory: &Path) -> Result<String> {
    use std::io::Read;

    let mut files = Vec::new();
    collect_files(directory, directory, &mut files)?;
    files.sort();

    let mut hasher = Sha256::new();
    for file in files {
        let relative = file.strip_prefix(directory)?;
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update([0]);
        let file_length = std::fs::metadata(&file)?.len();
        hasher.update(file_length.to_le_bytes());

        let mut reader = std::io::BufReader::new(std::fs::File::open(&file)?);
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(digest_hex(hasher.finalize().as_ref()))
}

fn collect_files(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, files)?;
        } else if path
            .strip_prefix(root)?
            .to_str()
            .is_some_and(|relative| relative != TREE_DIGEST_FILE)
        {
            files.push(path);
        }
    }
    Ok(())
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

    #[test]
    fn installed_directory_detects_tampering() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("model.onnx"), b"model").unwrap();
        write_tree_digest(temp.path()).unwrap();
        verify_installed_directory(temp.path()).unwrap();

        std::fs::write(temp.path().join("model.onnx"), b"changed").unwrap();
        assert!(verify_installed_directory(temp.path()).is_err());
    }

    #[tokio::test]
    async fn archive_install_is_atomic_and_verifiable() {
        let temp = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let model_dir = source.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"model").unwrap();

        let archive_path = temp.path().join("model.tar.bz2");
        let archive_file = std::fs::File::create(&archive_path).unwrap();
        let encoder = bzip2::write::BzEncoder::new(archive_file, bzip2::Compression::best());
        let mut archive = tar::Builder::new(encoder);
        archive.append_dir_all("model", &model_dir).unwrap();
        archive.into_inner().unwrap().finish().unwrap();

        extract_verified_archive(&archive_path, temp.path(), Path::new("model"))
            .await
            .unwrap();
        verify_installed_directory(&temp.path().join("model")).unwrap();
    }
}
