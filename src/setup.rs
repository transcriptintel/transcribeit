use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

use crate::cli::ModelSize;
use crate::models::{download_model, models_dir};

const SHERPA_ONNX_VERSION: &str = "v1.13.2";

pub(crate) async fn setup_models(
    output_dir: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<String> {
    let dir = output_dir.unwrap_or_else(models_dir);
    let dest = dir.join("ggml-base.bin");
    if dest.exists() {
        println!("models: already present (ggml-base.bin)");
        return Ok("already present".into());
    }
    download_model(&ModelSize::Base, Some(dir), hf_token).await?;
    Ok("installed (ggml-base.bin)".into())
}

pub(crate) async fn setup_vad(output_dir: Option<PathBuf>) -> Result<String> {
    let dir = output_dir.unwrap_or_else(models_dir);
    let dest = dir.join("silero_vad.onnx");
    download_file_with_progress(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        &dest,
        "VAD model (silero_vad.onnx)",
    )
    .await
}

pub(crate) async fn setup_diarize(output_dir: Option<PathBuf>) -> Result<String> {
    let dir = output_dir.unwrap_or_else(models_dir);
    let mut parts = Vec::new();

    let seg_dir = dir.join("sherpa-onnx-pyannote-segmentation-3-0");
    let seg_status = download_and_extract(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        &dir,
        &seg_dir,
        "diarize segmentation model",
    )
    .await?;
    parts.push(format!("segmentation: {seg_status}"));

    let emb_dest = dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    let emb_status = download_file_with_progress(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
        &emb_dest,
        "diarize embedding model (wespeaker)",
    )
    .await?;
    parts.push(format!("embedding: {emb_status}"));

    Ok(parts.join(", "))
}

pub(crate) async fn setup_sherpa_libs() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let archive_suffix = match (os, arch) {
        ("macos", "x86_64") => "osx-x64-shared-lib",
        ("macos", "aarch64") => "osx-arm64-shared-lib",
        ("linux", "x86_64") => "linux-x64-shared-lib",
        ("linux", "aarch64") => "linux-aarch64-shared-cpu-lib",
        _ => anyhow::bail!(
            "Unsupported platform: {os}-{arch}. Download sherpa-onnx shared libraries manually."
        ),
    };

    let archive_name = format!("sherpa-onnx-{SHERPA_ONNX_VERSION}-{archive_suffix}");
    let url = format!(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/{SHERPA_ONNX_VERSION}/{archive_name}.tar.bz2"
    );

    let vendor_dir = PathBuf::from("vendor");
    let check_dir = vendor_dir.join(&archive_name);

    let status = download_and_extract(
        &url,
        &vendor_dir,
        &check_dir,
        "sherpa-onnx shared libraries",
    )
    .await?;

    Ok(format!("{status} ({archive_suffix})"))
}

pub(crate) fn print_setup_summary(summary: &[(&str, String)]) {
    println!("\n=== Setup Summary ===");
    for (name, status) in summary {
        println!("  {name:<14} {status}");
    }

    let dir = models_dir();
    println!("\nAdd to .env (if not already set):");
    println!("  MODEL_CACHE_DIR={}", dir.display());

    let vad_path = dir.join("silero_vad.onnx");
    if vad_path.exists() {
        println!("  VAD_MODEL={}", vad_path.display());
    }

    let seg_path = dir.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx");
    if seg_path.exists() {
        println!("  DIARIZE_SEGMENTATION_MODEL={}", seg_path.display());
    }

    let emb_path = dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    if emb_path.exists() {
        println!("  DIARIZE_EMBEDDING_MODEL={}", emb_path.display());
    }

    if let Some(lib_dir) = sherpa_lib_dir_hint() {
        println!("  SHERPA_ONNX_LIB_DIR={}", lib_dir.display());
    }

    println!();
}

fn sherpa_lib_dir_hint() -> Option<PathBuf> {
    let vendor_dir = PathBuf::from("vendor");
    let expected_prefix = format!("sherpa-onnx-{SHERPA_ONNX_VERSION}-");

    let entries: Vec<_> = std::fs::read_dir(&vendor_dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("lib").exists())
        .collect();

    entries
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&expected_prefix))
        })
        .or_else(|| entries.first())
        .map(|path| path.join("lib"))
}

async fn download_file_with_progress(url: &str, dest: &Path, label: &str) -> Result<String> {
    if dest.exists() {
        println!("{label}: already present at {}", dest.display());
        return Ok("already present".into());
    }

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    println!("Downloading {label}...");
    println!("  from: {url}");
    println!("  to:   {}", dest.display());

    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to start download")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed with status: {}", resp.status());
    }

    let pb = download_progress_bar(resp.content_length().unwrap_or(0))?;
    let tmp_dest = dest.with_extension("part");
    write_response_to_path(resp, &tmp_dest, &pb).await?;

    tokio::fs::rename(&tmp_dest, dest)
        .await
        .context("Failed to finalize download")?;

    pb.finish_and_clear();
    println!("Done: {}", dest.display());
    Ok("installed".into())
}

async fn download_and_extract(
    url: &str,
    extract_to: &Path,
    check_dir: &Path,
    label: &str,
) -> Result<String> {
    if check_dir.exists() {
        println!("{label}: already present at {}", check_dir.display());
        return Ok("already present".into());
    }

    tokio::fs::create_dir_all(extract_to).await?;

    println!("Downloading {label}...");
    println!("  from: {url}");

    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to start download")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed with status: {}", resp.status());
    }

    let pb = download_progress_bar(resp.content_length().unwrap_or(0))?;
    let tmp = tempfile::Builder::new()
        .suffix(".tar.bz2")
        .tempfile_in(extract_to)
        .context("Failed to create temp file")?;
    let tmp_path = tmp.path().to_path_buf();

    write_response_to_path(resp, &tmp_path, &pb).await?;

    pb.finish_and_clear();
    println!("Extracting...");
    extract_archive(&tmp_path, extract_to).await?;
    let _ = tokio::fs::remove_file(&tmp_path).await;

    println!("Done: {}", check_dir.display());
    Ok("installed".into())
}

async fn write_response_to_path(
    resp: reqwest::Response,
    path: &Path,
    pb: &ProgressBar,
) -> Result<()> {
    let mut file = tokio::fs::File::create(path)
        .await
        .context("Failed to create temp file")?;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading download stream")?;
        file.write_all(&chunk).await.context("Failed to write")?;
        pb.inc(chunk.len() as u64);
    }
    file.flush().await?;
    Ok(())
}

async fn extract_archive(archive_path: &Path, extract_to: &Path) -> Result<()> {
    let archive_path = archive_path.to_path_buf();
    let extract_to = extract_to.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(&archive_path).context("Failed to open archive")?;
        let decoder = bzip2::read::BzDecoder::new(file);
        let mut archive = tar::Archive::new(decoder);
        archive.unpack(&extract_to).context("Failed to extract")?;
        Ok::<(), anyhow::Error>(())
    })
    .await?
}

fn download_progress_bar(total_size: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")?
            .progress_chars("##-"),
    );
    Ok(pb)
}
