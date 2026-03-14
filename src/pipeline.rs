use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use crate::audio::extract::{extract_to_wav, needs_conversion};
use crate::audio::segment::{compute_segments, detect_silence, get_duration, split_audio};
use crate::audio::wav::read_wav;
use crate::output::manifest::{
    InputInfo, Manifest, ProcessingConfig, SegmentInfo, Stats, write_manifest,
};
use crate::output::vtt::write_vtt;
use crate::transcriber::{Segment, Transcriber, Transcript};

/// API file size limit in bytes (25 MB).
const API_MAX_BYTES: u64 = 25 * 1024 * 1024;

/// Estimate WAV file size for a given duration (mono 16kHz 16-bit).
fn estimate_wav_bytes(duration_secs: f64) -> u64 {
    // 16000 samples/s * 2 bytes/sample * 1 channel + 44 byte header
    (duration_secs * 16000.0 * 2.0) as u64 + 44
}

#[derive(Debug, Clone)]
pub enum OutputFormat {
    Text,
    Vtt,
}

pub struct PipelineConfig {
    pub input: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub output_format: OutputFormat,
    pub segment: bool,
    pub silence_threshold: f64,
    pub min_silence_duration: f64,
    pub max_segment_secs: f64,
    pub provider_name: String,
    pub model_name: String,
    pub auto_split_for_api: bool,
}

pub async fn run_pipeline(engine: Box<dyn Transcriber>, config: PipelineConfig) -> Result<()> {
    let started = Instant::now();

    // Convert if needed
    let wav_path_buf;
    let _tmp_path; // keep TempPath alive
    let wav_path: &Path = if needs_conversion(&config.input) {
        eprintln!("Converting to mono 16kHz WAV...");
        let tmp = extract_to_wav(&config.input).await?;
        wav_path_buf = tmp.to_path_buf();
        _tmp_path = Some(tmp);
        &wav_path_buf
    } else {
        _tmp_path = None;
        &config.input
    };

    let total_duration = get_duration(wav_path).await?;

    // Decide whether to segment
    let should_segment = config.segment
        || (config.auto_split_for_api && estimate_wav_bytes(total_duration) > API_MAX_BYTES);

    if should_segment && !config.segment {
        eprintln!(
            "Audio is {:.0}s ({:.0} MB estimated) — auto-splitting for API size limits.",
            total_duration,
            estimate_wav_bytes(total_duration) as f64 / (1024.0 * 1024.0)
        );
    }

    let transcript = if should_segment {
        transcribe_segmented(
            &*engine,
            wav_path,
            total_duration,
            config.silence_threshold,
            config.min_silence_duration,
            config.max_segment_secs,
        )
        .await?
    } else {
        let samples = read_wav(wav_path)?;
        transcribe_with_spinner("Transcribing...", engine.transcribe(samples)).await?
    };

    let processing_time = started.elapsed().as_secs_f64();

    // Output
    match config.output_format {
        OutputFormat::Text => {
            println!("{}", transcript.text());
        }
        OutputFormat::Vtt => {
            if let Some(ref dir) = config.output_dir {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
                let stem = config
                    .input
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy();
                let vtt_path = dir.join(format!("{stem}.vtt"));
                let mut file = std::fs::File::create(&vtt_path)?;
                write_vtt(&transcript, &mut file)?;
                eprintln!("VTT written to {}", vtt_path.display());
            } else {
                let mut stdout = std::io::stdout();
                write_vtt(&transcript, &mut stdout)?;
            }
        }
    }

    // Manifest (always write if output_dir is set)
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let manifest_path = dir.join(format!("{stem}.manifest.json"));

        let manifest = Manifest {
            input: InputInfo {
                file: config.input.display().to_string(),
                duration_secs: total_duration,
            },
            config: ProcessingConfig {
                provider: config.provider_name,
                model: config.model_name,
                segmentation_enabled: should_segment,
                silence_threshold_db: config.silence_threshold,
                min_silence_duration_secs: config.min_silence_duration,
                output_format: format!("{:?}", config.output_format).to_lowercase(),
            },
            segments: transcript
                .segments
                .iter()
                .enumerate()
                .map(|(i, s)| SegmentInfo {
                    index: i,
                    start_secs: s.start_ms as f64 / 1000.0,
                    end_secs: s.end_ms as f64 / 1000.0,
                    text: s.text.trim().to_string(),
                })
                .collect(),
            stats: Stats {
                total_duration_secs: total_duration,
                total_segments: transcript.segments.len(),
                total_characters: transcript.segments.iter().map(|s| s.text.len()).sum(),
                processing_time_secs: processing_time,
            },
        };

        let mut file = std::fs::File::create(&manifest_path)?;
        write_manifest(&manifest, &mut file)?;
        eprintln!("Manifest written to {}", manifest_path.display());
    }

    Ok(())
}

async fn transcribe_segmented(
    engine: &dyn Transcriber,
    wav_path: &Path,
    total_duration: f64,
    silence_threshold: f64,
    min_silence_duration: f64,
    max_segment_secs: f64,
) -> Result<Transcript> {
    eprintln!("Detecting silence intervals...");
    let silences = detect_silence(wav_path, silence_threshold, min_silence_duration).await?;
    eprintln!("Found {} silence intervals.", silences.len());

    let audio_segments = compute_segments(&silences, total_duration, max_segment_secs);
    eprintln!("Processing {} segments...", audio_segments.len());

    let tmp_files = split_audio(wav_path, &audio_segments).await?;

    let mut all_segments: Vec<Segment> = Vec::new();

    for (i, (tmp_path, audio_seg)) in tmp_files.iter().zip(audio_segments.iter()).enumerate() {
        eprintln!(
            "  Transcribing segment {}/{} ({:.1}s - {:.1}s)...",
            i + 1,
            audio_segments.len(),
            audio_seg.start_secs,
            audio_seg.end_secs,
        );

        let samples = read_wav(tmp_path.as_ref())?;
        let transcript = transcribe_with_spinner(
            &format!(
                "Transcribing segment {}/{} ({:.1}s - {:.1}s)...",
                i + 1,
                audio_segments.len(),
                audio_seg.start_secs,
                audio_seg.end_secs
            ),
            engine.transcribe(samples),
        )
        .await?;

        // Offset segment timestamps by the audio segment start time
        let offset_ms = (audio_seg.start_secs * 1000.0) as i64;
        for mut seg in transcript.segments {
            seg.start_ms += offset_ms;
            seg.end_ms += offset_ms;
            all_segments.push(seg);
        }
    }

    Ok(Transcript {
        segments: all_segments,
    })
}

async fn transcribe_with_spinner<T, F>(message: &str, fut: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    let spinner = ProgressBar::new_spinner();
    let style = ProgressStyle::default_spinner()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        .template("{spinner:.green} {msg}")?;
    spinner.set_style(style);
    spinner.set_message(message.to_string());
    spinner.enable_steady_tick(Duration::from_millis(100));

    let result = fut.await;

    match result {
        Ok(value) => {
            spinner.finish_with_message(format!("{message} done"));
            Ok(value)
        }
        Err(err) => {
            spinner.finish_with_message(format!("{message} failed"));
            Err(err)
        }
    }
}
