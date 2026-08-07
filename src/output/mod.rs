use std::fs::{File, OpenOptions};
use std::path::Path;

use anyhow::{Context, Result};

use crate::transcriber::Transcript;

pub mod manifest;
pub mod srt;
pub mod vtt;

pub(crate) fn create_private_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .with_context(|| format!("Failed to create private output {}", path.display()))?;
    set_private_permissions(&file)?;
    Ok(file)
}

pub(crate) fn set_private_permissions(#[allow(unused_variables)] file: &File) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .context("Failed to set private output permissions")?;
    }
    Ok(())
}

pub(crate) fn validate_subtitle_timing(transcript: &Transcript) -> Result<()> {
    let mut previous_start = None;
    for (index, segment) in transcript.segments.iter().enumerate() {
        anyhow::ensure!(
            segment.start_ms >= 0,
            "subtitle segment {} starts before zero ({}ms)",
            index + 1,
            segment.start_ms
        );
        anyhow::ensure!(
            segment.end_ms > segment.start_ms,
            "subtitle segment {} must have a positive duration ({}ms..{}ms)",
            index + 1,
            segment.start_ms,
            segment.end_ms
        );
        if let Some(previous_start) = previous_start {
            anyhow::ensure!(
                segment.start_ms >= previous_start,
                "subtitle segment {} starts before the previous segment",
                index + 1
            );
        }
        previous_start = Some(segment.start_ms);
    }
    Ok(())
}
