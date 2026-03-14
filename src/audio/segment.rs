use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use tempfile::TempPath;

/// A detected silence interval in the audio.
pub struct SilenceInterval {
    pub start_secs: f64,
    pub end_secs: f64,
}

/// A segment of audio defined by start and end times.
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

    let stderr = String::from_utf8_lossy(&output.stderr);

    let start_re = Regex::new(r"silence_start:\s*(-?[\d.]+)")
        .context("Failed to compile silence_start regex")?;
    let end_re =
        Regex::new(r"silence_end:\s*(-?[\d.]+)").context("Failed to compile silence_end regex")?;

    let starts: Vec<f64> = start_re
        .captures_iter(&stderr)
        .filter_map(|c| c.get(1)?.as_str().parse().ok())
        .collect();

    let ends: Vec<f64> = end_re
        .captures_iter(&stderr)
        .filter_map(|c| c.get(1)?.as_str().parse().ok())
        .collect();

    let intervals: Vec<SilenceInterval> = starts
        .into_iter()
        .zip(ends)
        .map(|(start_secs, end_secs)| SilenceInterval {
            start_secs,
            end_secs,
        })
        .collect();

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
) -> Vec<AudioSegment> {
    const MIN_SEGMENT_SECS: f64 = 5.0;

    if total_duration <= max_segment_secs {
        return vec![AudioSegment {
            index: 0,
            start_secs: 0.0,
            end_secs: total_duration,
        }];
    }

    // Collect split points at silence midpoints
    let mut split_points: Vec<f64> = if silences.is_empty() {
        // Fallback: fixed-length segments
        let mut points = Vec::new();
        let mut t = max_segment_secs;
        while t < total_duration {
            points.push(t);
            t += max_segment_secs;
        }
        points
    } else {
        silences
            .iter()
            .map(|s| (s.start_secs + s.end_secs) / 2.0)
            .collect()
    };

    split_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    split_points.dedup();

    // Build segments from split points, respecting max_segment_secs
    let mut segments = Vec::new();
    let mut seg_start = 0.0;

    for &split in &split_points {
        if split <= seg_start {
            continue;
        }

        let seg_len = split - seg_start;

        // If adding up to this split would exceed max, we need to split earlier
        if seg_len > max_segment_secs {
            // Force split at max_segment_secs intervals
            while seg_start + max_segment_secs < split {
                let end = seg_start + max_segment_secs;
                segments.push(AudioSegment {
                    index: segments.len(),
                    start_secs: seg_start,
                    end_secs: end,
                });
                seg_start = end;
            }
            // Remaining portion up to split
            if split - seg_start >= MIN_SEGMENT_SECS {
                segments.push(AudioSegment {
                    index: segments.len(),
                    start_secs: seg_start,
                    end_secs: split,
                });
                seg_start = split;
            }
        } else if seg_len >= MIN_SEGMENT_SECS {
            segments.push(AudioSegment {
                index: segments.len(),
                start_secs: seg_start,
                end_secs: split,
            });
            seg_start = split;
        }
        // If seg_len < MIN_SEGMENT_SECS, skip this split point and let the segment grow
    }

    // Final segment from last split to end
    if total_duration - seg_start > 0.0 {
        // If remaining is very small, merge with previous segment
        if total_duration - seg_start < MIN_SEGMENT_SECS && !segments.is_empty() {
            let last = segments.last_mut().unwrap();
            last.end_secs = total_duration;
        } else {
            segments.push(AudioSegment {
                index: segments.len(),
                start_secs: seg_start,
                end_secs: total_duration,
            });
        }
    }

    // Re-index
    for (i, seg) in segments.iter_mut().enumerate() {
        seg.index = i;
    }

    segments
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
            .arg("-i")
            .arg(input)
            .arg("-to")
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
