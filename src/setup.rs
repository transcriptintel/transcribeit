use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::ModelSize;
use crate::models::{download_model, models_dir};

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

pub(crate) fn print_setup_summary(summary: &[(&str, String)], output_dir: Option<&Path>) {
    println!("\n=== Setup Summary ===");
    for (name, status) in summary {
        println!("  {name:<14} {status}");
    }

    let dir = absolute_path(output_dir.map(Path::to_path_buf).unwrap_or_else(models_dir))
        .unwrap_or_else(|_| output_dir.map(Path::to_path_buf).unwrap_or_else(models_dir));
    println!("\nAdd to .env (if not already set):");
    println!("  MODEL_CACHE_DIR={}", dir.display());
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
