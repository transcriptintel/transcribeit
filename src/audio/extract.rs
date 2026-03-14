use std::path::Path;

use anyhow::{Context, Result};
use tempfile::TempPath;

/// Check that ffmpeg is available on the system PATH.
pub fn check_ffmpeg() -> Result<()> {
    let output = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("ffmpeg not found. Please install ffmpeg and ensure it is on your PATH.")?;

    if !output.status.success() {
        anyhow::bail!("ffmpeg -version returned non-zero exit status");
    }

    Ok(())
}

/// Extract/convert any audio or video file to a mono 16kHz 16-bit WAV using ffmpeg.
///
/// Returns a `TempPath` to the generated WAV file, which will be cleaned up when dropped.
pub async fn extract_to_wav(input: &Path) -> Result<TempPath> {
    let tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .context("Failed to create temporary WAV file")?;

    let tmp_path = tmp.into_temp_path();

    let status = tokio::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg("-vn")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(tmp_path.as_os_str())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .context("Failed to run ffmpeg for audio extraction")?;

    if !status.success() {
        anyhow::bail!(
            "ffmpeg exited with status {} while converting {}",
            status,
            input.display()
        );
    }

    Ok(tmp_path)
}

/// Check whether a file needs conversion before it can be used as input.
///
/// Returns `false` (no conversion needed) only if the file is a valid WAV with
/// mono channel, 16kHz sample rate, and 16-bit integer sample format.
pub fn needs_conversion(path: &Path) -> bool {
    let reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(_) => return true,
    };

    let spec = reader.spec();
    !(spec.channels == 1
        && spec.sample_rate == 16000
        && spec.bits_per_sample == 16
        && spec.sample_format == hound::SampleFormat::Int)
}
