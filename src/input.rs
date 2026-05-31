use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub(crate) fn resolve_input_paths(input: &str) -> Result<Vec<PathBuf>> {
    let input_path = Path::new(input);

    if input_path.exists() {
        if input_path.is_file() {
            return Ok(vec![input_path.to_path_buf()]);
        }

        if input_path.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(input_path)
                .with_context(|| format!("Failed to read directory: {}", input_path.display()))?
            {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    files.push(entry.path());
                }
            }

            if files.is_empty() {
                anyhow::bail!("No files found in directory: {}", input_path.display());
            }

            files.sort_unstable();
            return Ok(files);
        }

        anyhow::bail!(
            "Input exists but is not a file or directory: {}",
            input_path.display()
        );
    }

    let mut matches = Vec::new();
    for entry in glob::glob(input).with_context(|| format!("Invalid input pattern: {input}"))? {
        let path =
            entry.with_context(|| format!("Invalid input glob match for pattern: {input}"))?;
        if path.is_file() {
            matches.push(path);
        }
    }

    if matches.is_empty() {
        anyhow::bail!("No files matched input pattern: {input}");
    }

    matches.sort_unstable();
    Ok(matches)
}
