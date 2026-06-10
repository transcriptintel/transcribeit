use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

use crate::cli::ModelSize;

const HF_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
#[cfg(feature = "sherpa-onnx")]
const SHERPA_ONNX_BASE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

pub(crate) fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| ".cache".to_string()))
}

pub(crate) fn resolve_cached_model_path(model: &str) -> Result<String> {
    let model = model.trim();
    if model.is_empty() {
        anyhow::bail!("Model name cannot be empty");
    }

    let direct_path = Path::new(model);
    if direct_path.exists() {
        return Ok(direct_path.to_string_lossy().into_owned());
    }

    let file_name = match model {
        "tiny" => Some("ggml-tiny.bin"),
        "tiny.en" => Some("ggml-tiny.en.bin"),
        "base" => Some("ggml-base.bin"),
        "base.en" => Some("ggml-base.en.bin"),
        "small" => Some("ggml-small.bin"),
        "small.en" => Some("ggml-small.en.bin"),
        "medium" => Some("ggml-medium.bin"),
        "medium.en" => Some("ggml-medium.en.bin"),
        "large-v3" => Some("ggml-large-v3.bin"),
        "large-v3-turbo" => Some("ggml-large-v3-turbo.bin"),
        _ => None,
    };

    if let Some(file_name) = file_name {
        let cache_path = models_dir().join(file_name);
        if cache_path.exists() {
            return Ok(cache_path.to_string_lossy().into_owned());
        }
        anyhow::bail!(
            "Model '{model}' not found in cache directory '{}'. Download it with: transcribeit download-model -s {model}",
            models_dir().display()
        );
    }

    if !model.contains(std::path::MAIN_SEPARATOR) && model.ends_with(".bin") {
        let cache_path = models_dir().join(model);
        if cache_path.exists() {
            return Ok(cache_path.to_string_lossy().into_owned());
        }
        anyhow::bail!(
            "Model file '{model}' not found in cache directory '{}'. Set --model to an existing path or download it first.",
            models_dir().display()
        );
    }

    anyhow::bail!(
        "Model '{model}' is not a recognized alias. Use a GGML model path or one of: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v3, large-v3-turbo."
    );
}

pub(crate) async fn download_model(
    model_size: &ModelSize,
    output_dir: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<()> {
    let dir = output_dir.unwrap_or_else(models_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    let file_name = model_size.file_name();
    let dest = dir.join(file_name);

    if dest.exists() {
        println!("Model already exists: {}", dest.display());
        return Ok(());
    }

    let url = format!("{HF_BASE_URL}/{file_name}");
    println!("Downloading {file_name} ...");
    println!("  from: {url}");
    println!("  to:   {}", dest.display());

    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if let Some(token) = hf_token {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.context("Failed to start download")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed with status: {}", resp.status());
    }

    let total_size = resp.content_length().unwrap_or(0);
    let pb = download_progress_bar(total_size)?;

    let tmp_dest = dest.with_extension("bin.part");
    write_response_to_path(resp, &tmp_dest, &pb).await?;

    tokio::fs::rename(&tmp_dest, &dest)
        .await
        .context("Failed to finalize download")?;

    pb.finish_and_clear();
    println!("Done: {}", dest.display());
    Ok(())
}

#[cfg(feature = "sherpa-onnx")]
pub(crate) fn resolve_onnx_model_dir(model: &str) -> Result<PathBuf> {
    let model = model.trim();

    let direct = PathBuf::from(model);
    if direct.is_dir() && has_tokens_file(&direct) {
        return Ok(direct);
    }

    let normalized = match model {
        "large-v3-turbo" => "turbo",
        other => other,
    };

    let candidates = [
        format!("sherpa-onnx-whisper-{normalized}"),
        format!("sherpa-onnx-whisper-{model}"),
        model.to_string(),
    ];
    for name in &candidates {
        let cache_path = models_dir().join(name);
        if cache_path.is_dir() && has_tokens_file(&cache_path) {
            return Ok(cache_path);
        }
    }

    let cache_dir = models_dir();
    if cache_dir.is_dir() {
        let pattern = format!("{}/*{}*", cache_dir.display(), normalized);
        if let Ok(paths) = glob::glob(&pattern) {
            for entry in paths.flatten() {
                if entry.is_dir() && has_tokens_file(&entry) {
                    return Ok(entry);
                }
            }
        }
    }

    anyhow::bail!(
        "ONNX model not found for '{model}'. Expected a directory with encoder.onnx, decoder.onnx, and tokens.txt.\n\
         Download with: transcribeit download-model -f onnx -s <size>"
    )
}

#[cfg(feature = "sherpa-onnx")]
pub(crate) async fn download_onnx_model(
    model_size: &ModelSize,
    output_dir: Option<PathBuf>,
) -> Result<()> {
    let archive_name = model_size
        .onnx_archive_name()
        .context("This model size is not available in ONNX format")?;

    let dir = output_dir.unwrap_or_else(models_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    let dest_dir = dir.join(archive_name);
    if dest_dir.exists() {
        println!("Model already exists: {}", dest_dir.display());
        return Ok(());
    }

    let url = format!("{SHERPA_ONNX_BASE_URL}/{archive_name}.tar.bz2");
    println!("Downloading {archive_name}.tar.bz2 ...");
    println!("  from: {url}");
    println!("  to:   {}", dest_dir.display());

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .send()
        .await
        .context("Failed to start ONNX model download")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed with status: {}", resp.status());
    }

    let pb = download_progress_bar(resp.content_length().unwrap_or(0))?;
    let tmp = tempfile::Builder::new()
        .suffix(".tar.bz2")
        .tempfile_in(&dir)
        .context("Failed to create temp file")?;
    let tmp_path = tmp.path().to_path_buf();

    write_response_to_path(resp, &tmp_path, &pb).await?;

    pb.finish_and_clear();
    println!("Extracting...");
    extract_archive(&tmp_path, &dir)
        .await
        .context("Failed to extract ONNX model archive")?;
    let _ = tokio::fs::remove_file(&tmp_path).await;

    if dest_dir.exists() {
        println!("Done: {}", dest_dir.display());
    } else {
        println!("Done: extracted to {}", dir.display());
    }
    Ok(())
}

pub(crate) fn list_models(dir: Option<PathBuf>) -> Result<()> {
    let dir = dir.unwrap_or_else(models_dir);

    if !dir.exists() {
        println!("No models found. Run `transcribeit download-model` first.");
        return Ok(());
    }

    let mut found = false;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .context("Failed to read models directory")?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("bin") {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let size_mb = size as f64 / (1024.0 * 1024.0);
            println!(
                "  {} ({:.0} MB) [ggml]",
                path.file_name().unwrap().to_string_lossy(),
                size_mb
            );
            found = true;
        }
    }

    #[cfg(feature = "sherpa-onnx")]
    for entry in &entries {
        let path = entry.path();
        if path.is_dir() && has_tokens_file(&path) {
            println!("  {}/ [onnx]", path.file_name().unwrap().to_string_lossy());
            found = true;
        }
    }

    if !found {
        println!("No models found in {}", dir.display());
    }

    Ok(())
}

#[cfg(feature = "sherpa-onnx")]
fn has_tokens_file(dir: &Path) -> bool {
    if dir.join("tokens.txt").exists() {
        return true;
    }
    glob::glob(&format!("{}/*-tokens.txt", dir.display()))
        .ok()
        .and_then(|mut paths| paths.next())
        .is_some_and(|p| p.is_ok())
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

#[cfg(feature = "sherpa-onnx")]
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
