use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub(crate) fn validate_batch_output_plan(
    inputs: &[PathBuf],
    output_dir: Option<&Path>,
) -> Result<()> {
    if inputs.len() <= 1 {
        return Ok(());
    }

    output_dir.context(
        "batch transcription requires --output-dir; concatenating multiple transcripts on stdout is not supported",
    )?;

    let mut stems: HashMap<String, &Path> = HashMap::new();
    for input in inputs {
        let stem = input
            .file_stem()
            .with_context(|| format!("input has no usable output filename: {}", input.display()))?;
        let collision_key = stem.to_string_lossy().to_lowercase();
        if let Some(first) = stems.insert(collision_key, input.as_path()) {
            anyhow::bail!(
                "batch output collision: '{}' and '{}' both map to the stem '{}'; rename an input or run them into separate output directories",
                first.display(),
                input.display(),
                stem.to_string_lossy()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_requires_output_directory() {
        let inputs = vec![PathBuf::from("a.wav"), PathBuf::from("b.wav")];
        assert!(validate_batch_output_plan(&inputs, None).is_err());
    }

    #[test]
    fn batch_rejects_duplicate_stems_before_processing() {
        let inputs = vec![
            PathBuf::from("first/call.wav"),
            PathBuf::from("second/call.mp3"),
        ];
        let error = validate_batch_output_plan(&inputs, Some(Path::new("out"))).unwrap_err();
        assert!(error.to_string().contains("batch output collision"));
    }

    #[test]
    fn batch_rejects_case_only_collisions() {
        let inputs = vec![
            PathBuf::from("first/Call.wav"),
            PathBuf::from("second/call.mp3"),
        ];
        assert!(validate_batch_output_plan(&inputs, Some(Path::new("out"))).is_err());
    }

    #[test]
    fn batch_accepts_unique_stems_with_output_directory() {
        let inputs = vec![PathBuf::from("a.wav"), PathBuf::from("b.wav")];
        assert!(validate_batch_output_plan(&inputs, Some(Path::new("out"))).is_ok());
    }
}
