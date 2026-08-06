use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use tempfile::TempPath;

use super::LOCAL_MEDIA_PROTOCOLS;

/// A detected silence interval in the audio.
#[derive(Debug)]
pub struct SilenceInterval {
    pub start_secs: f64,
    pub end_secs: f64,
}

/// A segment of audio defined by start and end times.
#[derive(Debug, PartialEq)]
pub struct AudioSegment {
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
}

/// Get the duration of an audio file in seconds using ffprobe.
pub async fn get_duration(input: &Path) -> Result<f64> {
    let output = tokio::process::Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("csv=p=0")
        .arg("-protocol_whitelist")
        .arg(LOCAL_MEDIA_PROTOCOLS)
        .arg(input)
        .output()
        .await
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffprobe failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let duration: f64 = stdout.trim().parse().with_context(|| {
        format!(
            "Failed to parse ffprobe duration output: {:?}",
            stdout.trim()
        )
    })?;

    Ok(duration)
}

/// Detect silence intervals in an audio file using ffmpeg's silencedetect filter.
///
/// - `noise_db`: silence threshold in dB (e.g., -30.0)
/// - `min_duration`: minimum silence duration in seconds (e.g., 0.5)
pub async fn detect_silence(
    input: &Path,
    noise_db: f64,
    min_duration: f64,
) -> Result<Vec<SilenceInterval>> {
    let af = format!("silencedetect=noise={}dB:d={}", noise_db, min_duration);

    let output = tokio::process::Command::new("ffmpeg")
        .arg("-protocol_whitelist")
        .arg(LOCAL_MEDIA_PROTOCOLS)
        .arg("-i")
        .arg(input)
        .arg("-vn")
        .arg("-af")
        .arg(&af)
        .arg("-f")
        .arg("null")
        .arg("-")
        .output()
        .await
        .context("Failed to run ffmpeg silencedetect")?;

    if !output.status.success() {
        anyhow::bail!(
            "ffmpeg silencedetect exited with status {} for {}",
            output.status,
            input.display()
        );
    }

    let stderr = String::from_utf8_lossy(&output.stderr);

    let start_re = Regex::new(r"silence_start:\s*(-?[\d.]+)")
        .context("Failed to compile silence_start regex")?;
    let end_re =
        Regex::new(r"silence_end:\s*(-?[\d.]+)").context("Failed to compile silence_end regex")?;

    // Parse sequentially: walk lines and pair start→end in order,
    // skipping unmatched entries to avoid mispaired intervals.
    let mut intervals = Vec::new();
    let mut pending_start: Option<f64> = None;

    for line in stderr.lines() {
        if let Some(caps) = start_re.captures(line) {
            if let Some(val) = caps.get(1).and_then(|m| m.as_str().parse().ok()) {
                pending_start = Some(val);
            }
        } else if let Some(caps) = end_re.captures(line)
            && let (Some(start), Some(end)) = (
                pending_start.take(),
                caps.get(1).and_then(|m| m.as_str().parse::<f64>().ok()),
            )
            && end > start
        {
            intervals.push(SilenceInterval {
                start_secs: start,
                end_secs: end,
            });
        }
    }

    Ok(intervals)
}

/// Compute audio segments by splitting at silence midpoints.
///
/// - `silences`: detected silence intervals
/// - `total_duration`: total audio duration in seconds
/// - `max_segment_secs`: maximum allowed segment length in seconds
///
/// If no silences are found, falls back to fixed-length segments.
/// Enforces a minimum segment duration of 5 seconds.
pub fn compute_segments(
    silences: &[SilenceInterval],
    total_duration: f64,
    max_segment_secs: f64,
) -> Result<Vec<AudioSegment>> {
    const MIN_SEGMENT_SECS: f64 = 5.0;

    if !total_duration.is_finite() || total_duration <= 0.0 {
        anyhow::bail!("audio duration must be finite and greater than zero");
    }
    if !max_segment_secs.is_finite() || max_segment_secs < 0.001 {
        anyhow::bail!("maximum segment duration must be finite and at least 0.001 seconds");
    }

    if total_duration <= max_segment_secs {
        return Ok(vec![AudioSegment {
            index: 0,
            start_secs: 0.0,
            end_secs: total_duration,
        }]);
    }

    let mut silence_midpoints: Vec<f64> = silences
        .iter()
        .map(|silence| (silence.start_secs + silence.end_secs) / 2.0)
        .filter(|point| point.is_finite() && *point > 0.0 && *point < total_duration)
        .collect();
    silence_midpoints.sort_by(f64::total_cmp);
    silence_midpoints.dedup();

    let mut segments = Vec::new();
    let mut start = 0.0;
    while total_duration - start > max_segment_secs {
        let hard_end = (start + max_segment_secs).min(total_duration);
        let minimum_preferred_end = start + MIN_SEGMENT_SECS.min(max_segment_secs);
        let end = silence_midpoints
            .iter()
            .copied()
            .take_while(|point| *point <= hard_end)
            .filter(|point| *point >= minimum_preferred_end)
            .last()
            .unwrap_or(hard_end);
        anyhow::ensure!(
            end > start,
            "maximum segment duration is too small to make numeric progress"
        );

        segments.push(AudioSegment {
            index: segments.len(),
            start_secs: start,
            end_secs: end,
        });
        start = end;
    }

    if start < total_duration {
        segments.push(AudioSegment {
            index: segments.len(),
            start_secs: start,
            end_secs: total_duration,
        });
    }

    debug_assert!(segments.iter().all(|segment| {
        segment.end_secs > segment.start_secs
            && segment.end_secs - segment.start_secs <= max_segment_secs
    }));

    Ok(segments)
}

/// Split an audio file into segments using ffmpeg, returning temp WAV files.
pub async fn split_audio(input: &Path, segments: &[AudioSegment]) -> Result<Vec<TempPath>> {
    let mut paths = Vec::with_capacity(segments.len());

    for seg in segments {
        let tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .context("Failed to create temporary WAV file for segment")?;

        let tmp_path = tmp.into_temp_path();
        let duration = seg.end_secs - seg.start_secs;

        let status = tokio::process::Command::new("ffmpeg")
            .arg("-y")
            .arg("-ss")
            .arg(format!("{}", seg.start_secs))
            .arg("-protocol_whitelist")
            .arg(LOCAL_MEDIA_PROTOCOLS)
            .arg("-i")
            .arg(input)
            .arg("-t")
            .arg(format!("{}", duration))
            .arg("-ar")
            .arg("16000")
            .arg("-ac")
            .arg("1")
            .arg("-c:a")
            .arg("pcm_s16le")
            .arg(tmp_path.as_os_str())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .with_context(|| {
                format!(
                    "Failed to run ffmpeg for segment {} ({:.2}s - {:.2}s)",
                    seg.index, seg.start_secs, seg.end_secs
                )
            })?;

        if !status.success() {
            anyhow::bail!(
                "ffmpeg exited with status {} for segment {} ({:.2}s - {:.2}s)",
                status,
                seg.index,
                seg.start_secs,
                seg.end_secs
            );
        }

        paths.push(tmp_path);
    }

    Ok(paths)
}

#[cfg(test)]
mod tests;
