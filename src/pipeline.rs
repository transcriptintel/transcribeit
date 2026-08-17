use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use futures_util::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};

use crate::analysis::{AnalysisConfig, TranscriptAnalyzer};
use crate::audio::extract::{extract_to_mp3, extract_to_wav, needs_conversion};
use crate::audio::segment::{compute_segments, detect_silence, get_duration, split_audio};
use crate::pipeline_merge::{TranscriptChunk, merge_segmented_transcripts};
use crate::pipeline_output::{write_manifest_output, write_transcript_output};
use crate::transcriber::{Transcriber, Transcript};

/// API file size limit in bytes (25 MB).
pub const API_UPLOAD_MAX_BYTES: u64 = 25 * 1024 * 1024;

#[derive(Debug, Clone)]
pub enum OutputFormat {
    Text,
    Vtt,
    Srt,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub input: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub output_format: OutputFormat,
    pub language: Option<String>,
    pub segment: bool,
    pub silence_threshold: f64,
    pub min_silence_duration: f64,
    pub max_segment_secs: f64,
    pub provider_name: String,
    pub model_name: String,
    pub auto_split_max_bytes: Option<u64>,
    pub upload_as_mp3: bool,
    pub segment_concurrency: usize,
    pub normalize_audio: bool,
    pub analysis: AnalysisConfig,
}

pub async fn run_pipeline(
    engine: &dyn Transcriber,
    analyzer: Option<&dyn TranscriptAnalyzer>,
    config: PipelineConfig,
) -> Result<()> {
    let started = Instant::now();

    let use_original_media = engine.prefers_original_media()
        && !config.normalize_audio
        && !config.segment
        && config.auto_split_max_bytes.is_none()
        && !config.upload_as_mp3;
    let (canonical_path, _canonical_tmp) =
        if !use_original_media && (needs_conversion(&config.input) || config.normalize_audio) {
            eprintln!("Converting to mono 16kHz WAV...");
            let tmp = extract_to_wav(&config.input, config.normalize_audio).await?;
            (tmp.to_path_buf(), Some(tmp))
        } else {
            (config.input.clone(), None)
        };
    let canonical_path = canonical_path.as_path();

    let total_duration = get_duration(canonical_path).await?;

    let direct_upload_tmp = if !config.segment && config.upload_as_mp3 {
        eprintln!("Encoding provider upload as mono 16kHz MP3...");
        Some(extract_to_mp3(canonical_path, false).await?)
    } else {
        None
    };
    let direct_input_path = direct_upload_tmp.as_deref().unwrap_or(canonical_path);
    let direct_input_bytes = tokio::fs::metadata(direct_input_path)
        .await
        .map(|metadata| metadata.len())?;

    // Decide whether to segment
    let should_segment = config.segment
        || config
            .auto_split_max_bytes
            .is_some_and(|maximum| direct_input_bytes > maximum);
    let used_segmentation = should_segment;

    if should_segment && !config.segment {
        eprintln!(
            "Prepared upload is {:.1} MiB for {:.0}s of audio — auto-splitting for provider size limits.",
            direct_input_bytes as f64 / (1024.0 * 1024.0),
            total_duration,
        );
    }

    let transcript = if should_segment {
        transcribe_segmented(engine, canonical_path, total_duration, &config).await?
    } else {
        transcribe_with_spinner("Transcribing...", engine.transcribe_path(direct_input_path))
            .await?
    };

    write_transcript_output(&config, &transcript)?;
    write_manifest_output(
        &config,
        &transcript,
        None,
        None,
        total_duration,
        used_segmentation,
        started.elapsed().as_secs_f64(),
    )?;

    if config.analysis.is_enabled() {
        let analysis = match analyzer {
            Some(analyzer) => {
                transcribe_with_spinner(
                    "Analyzing transcript...",
                    analyzer.analyze_transcript(&transcript, &config.analysis),
                )
                .await
            }
            None => Err(anyhow::anyhow!(
                "--analysis was requested, but provider '{}' does not support transcript analysis yet",
                config.provider_name
            )),
        };

        match analysis {
            Ok(analysis) => write_manifest_output(
                &config,
                &transcript,
                Some(&analysis),
                None,
                total_duration,
                used_segmentation,
                started.elapsed().as_secs_f64(),
            )?,
            Err(analysis_error) => {
                let message = analysis_error.to_string();
                write_manifest_output(
                    &config,
                    &transcript,
                    None,
                    Some(&message),
                    total_duration,
                    used_segmentation,
                    started.elapsed().as_secs_f64(),
                )
                .with_context(|| {
                    format!(
                        "Transcript was persisted, but recording the analysis failure also failed: {message}"
                    )
                })?;
                return Err(analysis_error.context(
                    "Transcription completed and was persisted, but optional analysis failed",
                ));
            }
        }
    }

    Ok(())
}

async fn transcribe_segmented(
    engine: &dyn Transcriber,
    wav_path: &Path,
    total_duration: f64,
    config: &PipelineConfig,
) -> Result<Transcript> {
    eprintln!("Detecting silence intervals...");
    let silences = detect_silence(
        wav_path,
        config.silence_threshold,
        config.min_silence_duration,
    )
    .await?;
    eprintln!("Found {} silence intervals.", silences.len());

    let audio_segments = compute_segments(&silences, total_duration, config.max_segment_secs)?;
    eprintln!("Processing {} segments...", audio_segments.len());

    let tmp_files = split_audio(wav_path, &audio_segments).await?;

    let segment_jobs: Vec<_> = tmp_files
        .iter()
        .zip(audio_segments.iter())
        .enumerate()
        .map(|(i, (tmp_path, audio_seg))| {
            (
                i,
                tmp_path.to_owned(),
                audio_seg.start_secs,
                (audio_seg.start_secs * 1000.0) as i64,
                audio_segments.len(),
            )
        })
        .collect();

    let concurrency = if config.upload_as_mp3 {
        config.segment_concurrency.max(1)
    } else {
        1
    };
    let mut collected_transcripts: Vec<Option<TranscriptChunk>> =
        (0..segment_jobs.len()).map(|_| None).collect();
    for batch in segment_jobs.chunks(concurrency) {
        let mut jobs = Vec::with_capacity(batch.len());
        for &(index, ref segment_path, _, start_ms, total_count) in batch {
            eprintln!(
                "  Transcribing segment {}/{} ({:.1}s - ...)",
                index + 1,
                total_count,
                start_ms as f64 / 1000.0
            );

            let job = async move {
                let segment_offset_ms = start_ms;
                let local_transcript = if config.upload_as_mp3 {
                    let mp3_path = extract_to_mp3(segment_path, false).await?;
                    engine.transcribe_path(mp3_path.as_ref()).await?
                } else {
                    engine.transcribe_path(segment_path.as_ref()).await?
                };

                Ok::<(usize, TranscriptChunk), anyhow::Error>((
                    index,
                    TranscriptChunk {
                        index,
                        offset_ms: segment_offset_ms,
                        transcript: local_transcript,
                    },
                ))
            };
            jobs.push(job);
        }

        let batch_results = join_all(jobs).await;
        for batch_result in batch_results {
            let (index, transcript) = batch_result?;
            collected_transcripts[index] = Some(transcript);
        }
    }

    Ok(merge_segmented_transcripts(
        &config.provider_name,
        collected_transcripts.into_iter().flatten().collect(),
    ))
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

#[cfg(test)]
mod tests;
