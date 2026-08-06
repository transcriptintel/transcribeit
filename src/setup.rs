use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use crate::artifacts::{
    ArtifactIntegrity, extract_verified_archive, verify_file, verify_installed_directory,
    write_verified_response,
};
use crate::cli::ModelSize;
use crate::models::{download_model, models_dir};

mod qwen3;
mod sherpa;
#[cfg(feature = "sherpa-onnx")]
pub(crate) use qwen3::QWEN3_ASR_ARCHIVE;
pub(crate) use qwen3::setup_qwen3_asr;
pub(crate) use sherpa::setup_sherpa_libs;
use sherpa::sherpa_lib_dir_hint;

pub(crate) async fn setup_models(
    output_dir: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<String> {
    let dir = output_dir.unwrap_or_else(models_dir);
    let dest = dir.join("ggml-base.bin");
    let existed = dest.exists();
    download_model(&ModelSize::Base, Some(dir), hf_token).await?;
    Ok(if existed {
        "verified (ggml-base.bin)".into()
    } else {
        "installed (ggml-base.bin)".into()
    })
}

pub(crate) async fn setup_vad(output_dir: Option<PathBuf>) -> Result<String> {
    let dir = output_dir.unwrap_or_else(models_dir);
    let dest = dir.join("silero_vad.onnx");
    download_file_with_progress(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        &dest,
        "VAD model (silero_vad.onnx)",
        ArtifactIntegrity {
            sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
            size_bytes: 643_854,
        },
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
        ArtifactIntegrity {
            sha256: "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
            size_bytes: 6_958_444,
        },
    )
    .await?;
    parts.push(format!("segmentation: {seg_status}"));

    let emb_dest = dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    let emb_status = download_file_with_progress(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
        &emb_dest,
        "diarize embedding model (wespeaker)",
        ArtifactIntegrity {
            sha256: "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef",
            size_bytes: 29_292_684,
        },
    )
    .await?;
    parts.push(format!("embedding: {emb_status}"));

    Ok(parts.join(", "))
}

pub(crate) fn print_setup_summary(summary: &[(&str, String)], output_dir: Option<&Path>) {
    println!("\n=== Setup Summary ===");
    for (name, status) in summary {
        println!("  {name:<14} {status}");
    }

    let dir = absolute_path(output_dir.map(Path::to_path_buf).unwrap_or_else(models_dir))
        .unwrap_or_else(|_| output_dir.map(Path::to_path_buf).unwrap_or_else(models_dir));
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

    let sherpa_root = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("vendor"));
    if let Some(lib_dir) = sherpa_lib_dir_hint(&sherpa_root) {
        println!("  SHERPA_ONNX_LIB_DIR={}", lib_dir.display());
    }

    println!();
}

fn absolute_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()
            .context("Failed to resolve current directory")?
            .join(path))
    }
}

async fn download_file_with_progress(
    url: &str,
    dest: &Path,
    label: &str,
    integrity: ArtifactIntegrity,
) -> Result<String> {
    if dest.exists() {
        verify_file(dest, integrity).await?;
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

    let pb = download_progress_bar(integrity.size_bytes)?;
    let tmp_dest = dest.with_extension("part");
    write_verified_response(resp, &tmp_dest, &pb, integrity).await?;

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
    integrity: ArtifactIntegrity,
) -> Result<String> {
    if check_dir.exists() {
        verify_installed_directory(check_dir)?;
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

    let pb = download_progress_bar(integrity.size_bytes)?;
    let tmp = tempfile::Builder::new()
        .suffix(".tar.bz2")
        .tempfile_in(extract_to)
        .context("Failed to create temp file")?;
    let tmp_path = tmp.path().to_path_buf();

    write_verified_response(resp, &tmp_path, &pb, integrity).await?;

    pb.finish_and_clear();
    println!("Extracting...");
    let directory_name = check_dir
        .file_name()
        .context("artifact destination has no directory name")?;
    extract_verified_archive(&tmp_path, extract_to, Path::new(directory_name)).await?;
    let _ = tokio::fs::remove_file(&tmp_path).await;

    println!("Done: {}", check_dir.display());
    Ok("installed".into())
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

#[cfg(test)]
mod tests;
