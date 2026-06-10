use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

#[cfg(feature = "sherpa-onnx")]
use anyhow::Context;
use anyhow::Result;
use futures_util::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};

use crate::audio::extract::{extract_to_mp3, extract_to_wav, needs_conversion};
use crate::audio::segment::{compute_segments, detect_silence, get_duration, split_audio};
use crate::pipeline_output::write_outputs;
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
    pub auto_split_for_api: bool,
    pub upload_as_mp3: bool,
    pub segment_concurrency: usize,
    pub normalize_audio: bool,
    #[cfg_attr(not(feature = "sherpa-onnx"), allow(dead_code))]
    pub speakers: Option<i32>,
    #[cfg_attr(not(feature = "sherpa-onnx"), allow(dead_code))]
    pub diarize_segmentation_model: Option<String>,
    #[cfg_attr(not(feature = "sherpa-onnx"), allow(dead_code))]
    pub diarize_embedding_model: Option<String>,
    /// Path to Silero VAD model for speech-aware segmentation (sherpa-onnx only)
    #[cfg_attr(not(feature = "sherpa-onnx"), allow(dead_code))]
    pub vad_model: Option<String>,
}

pub async fn run_pipeline(engine: &dyn Transcriber, config: PipelineConfig) -> Result<()> {
    let started = Instant::now();

    let (input_path, _tmp_path) = if config.upload_as_mp3 {
        eprintln!("Converting to mono 16kHz MP3...");
        let tmp = extract_to_mp3(&config.input, config.normalize_audio).await?;
        (tmp.to_path_buf(), Some(tmp))
    } else if needs_conversion(&config.input) {
        eprintln!("Converting to mono 16kHz WAV...");
        let tmp = extract_to_wav(&config.input, config.normalize_audio).await?;
        (tmp.to_path_buf(), Some(tmp))
    } else {
        (config.input.clone(), None)
    };
    let input_path = input_path.as_path();

    let total_duration = get_duration(input_path).await?;

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

    #[allow(unused_mut)]
    let mut transcript = if should_segment {
        // Use VAD-based segmentation when available (sherpa-onnx), fall back to FFmpeg silencedetect
        #[cfg(feature = "sherpa-onnx")]
        if let Some(ref vad_model) = config.vad_model {
            transcribe_vad_segmented(engine, input_path, vad_model, &config).await?
        } else {
            transcribe_segmented(engine, input_path, total_duration, &config).await?
        }
        #[cfg(not(feature = "sherpa-onnx"))]
        transcribe_segmented(engine, input_path, total_duration, &config).await?
    } else {
        transcribe_with_spinner("Transcribing...", engine.transcribe_path(input_path)).await?
    };

    // Speaker diarization (if requested)
    #[cfg(feature = "sherpa-onnx")]
    if let Some(num_speakers) = config.speakers {
        let seg_model = config
            .diarize_segmentation_model
            .as_deref()
            .context("--diarize-segmentation-model is required when --speakers is set")?;
        let emb_model = config
            .diarize_embedding_model
            .as_deref()
            .context("--diarize-embedding-model is required when --speakers is set")?;

        eprintln!("Running speaker diarization ({num_speakers} speakers)...");

        let diarizer = crate::diarize::Diarizer::new(
            std::path::Path::new(seg_model),
            std::path::Path::new(emb_model),
            num_speakers,
        )?;

        // Read the audio samples for diarization
        let wav_bytes = std::fs::read(input_path).with_context(|| {
            format!(
                "Failed to read audio for diarization: {}",
                input_path.display()
            )
        })?;
        let diarize_samples = crate::audio::wav::read_wav_bytes(&wav_bytes)?;
        let diarized =
            transcribe_with_spinner("Diarizing...", diarizer.diarize(diarize_samples)).await?;

        eprintln!(
            "Found {} speaker segments across {} speakers.",
            diarized.len(),
            num_speakers
        );

        crate::diarize::assign_speakers(&mut transcript, &diarized);
    }

    write_outputs(
        &config,
        &transcript,
        total_duration,
        should_segment,
        started.elapsed().as_secs_f64(),
    )?;

    Ok(())
}

#[cfg(feature = "sherpa-onnx")]
async fn transcribe_vad_segmented(
    engine: &dyn Transcriber,
    wav_path: &Path,
    vad_model: &str,
    config: &PipelineConfig,
) -> Result<Transcript> {
    use crate::audio::vad;
    use crate::audio::wav::read_wav_bytes;

    eprintln!("Running VAD-based speech segmentation...");

    // Read audio samples for VAD
    let wav_bytes = std::fs::read(wav_path)
        .with_context(|| format!("Failed to read: {}", wav_path.display()))?;
    let samples = read_wav_bytes(&wav_bytes)?;

    let chunks = vad::vad_segment(&samples, vad_model, config.max_segment_secs as f32)?;

    eprintln!("Found {} speech chunks (VAD).", chunks.len());

    if chunks.is_empty() {
        eprintln!("No speech detected.");
        return Ok(Transcript {
            segments: Vec::new(),
            provider_metadata: None,
        });
    }

    let mut all_segments: Vec<Segment> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!(
            "  Transcribing chunk {}/{} ({:.1}s - {:.1}s, {:.1}s)...",
            i + 1,
            chunks.len(),
            chunk.start_secs(),
            chunk.end_secs(),
            chunk.duration_secs(),
        );

        let chunk_samples = samples[chunk.start_sample..chunk.end_sample].to_vec();
        let transcript = transcribe_with_spinner(
            &format!(
                "Transcribing chunk {}/{} ({:.1}s)...",
                i + 1,
                chunks.len(),
                chunk.duration_secs(),
            ),
            engine.transcribe(chunk_samples),
        )
        .await?;

        // Offset timestamps by the chunk start time
        let offset_ms = (chunk.start_secs() * 1000.0) as i64;
        for mut seg in transcript.segments {
            seg.start_ms += offset_ms;
            seg.end_ms += offset_ms;
            all_segments.push(seg);
        }
    }

    Ok(Transcript {
        segments: all_segments,
        provider_metadata: None,
    })
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

    let audio_segments = compute_segments(&silences, total_duration, config.max_segment_secs);
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
    let mut collected_segments: Vec<Option<Vec<Segment>>> =
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
                    let mp3_path = extract_to_mp3(segment_path, config.normalize_audio).await?;
                    engine.transcribe_path(mp3_path.as_ref()).await?
                } else {
                    engine.transcribe_path(segment_path.as_ref()).await?
                };

                let mut segments = local_transcript.segments;
                for segment in segments.iter_mut() {
                    segment.start_ms += segment_offset_ms;
                    segment.end_ms += segment_offset_ms;
                }

                Ok::<(usize, Vec<Segment>), anyhow::Error>((index, segments))
            };
            jobs.push(job);
        }

        let batch_results = join_all(jobs).await;
        for batch_result in batch_results {
            let (index, segments) = batch_result?;
            collected_segments[index] = Some(segments);
        }
    }

    let mut all_segments = Vec::new();
    for segment_parts in collected_segments.into_iter().flatten() {
        all_segments.extend(segment_parts);
    }

    Ok(Transcript {
        segments: all_segments,
        provider_metadata: None,
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

#[cfg(test)]
mod tests;
