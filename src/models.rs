use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use crate::artifacts::{
    ArtifactIntegrity, verify_file, verify_file_blocking, write_verified_response,
};
use crate::cli::ModelSize;

const HF_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

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
            verify_file_blocking(&cache_path, ggml_integrity(file_name)?)?;
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
    let integrity = ggml_integrity(file_name)?;
    let dest = dir.join(file_name);

    if dest.exists() {
        verify_file(&dest, integrity).await?;
        println!("Model already exists: {}", dest.display());
        return Ok(());
    }

    let url =
        format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/{HF_REVISION}/{file_name}");
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

    let pb = download_progress_bar(integrity.size_bytes)?;

    let tmp_dest = dest.with_extension("bin.part");
    write_verified_response(resp, &tmp_dest, &pb, integrity).await?;

    tokio::fs::rename(&tmp_dest, &dest)
        .await
        .context("Failed to finalize download")?;

    pb.finish_and_clear();
    println!("Done: {}", dest.display());
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

    if !found {
        println!("No models found in {}", dir.display());
    }

    Ok(())
}

fn ggml_integrity(file_name: &str) -> Result<ArtifactIntegrity> {
    let integrity = match file_name {
        "ggml-tiny.bin" => (
            "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
            77_691_713,
        ),
        "ggml-tiny.en.bin" => (
            "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
            77_704_715,
        ),
        "ggml-base.bin" => (
            "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
            147_951_465,
        ),
        "ggml-base.en.bin" => (
            "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
            147_964_211,
        ),
        "ggml-small.bin" => (
            "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
            487_601_967,
        ),
        "ggml-small.en.bin" => (
            "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
            487_614_201,
        ),
        "ggml-medium.bin" => (
            "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
            1_533_763_059,
        ),
        "ggml-medium.en.bin" => (
            "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356",
            1_533_774_781,
        ),
        "ggml-large-v3.bin" => (
            "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
            3_095_033_483,
        ),
        "ggml-large-v3-turbo.bin" => (
            "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
            1_624_555_275,
        ),
        _ => anyhow::bail!("No pinned integrity metadata for GGML artifact {file_name}"),
    };
    Ok(ArtifactIntegrity {
        sha256: integrity.0,
        size_bytes: integrity.1,
    })
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
