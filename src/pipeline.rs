use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result};
use futures_util::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};

use crate::audio::extract::{extract_to_mp3, extract_to_wav, needs_conversion};
use crate::audio::segment::{compute_segments, detect_silence, get_duration, split_audio};
use crate::output::manifest::{
    InputInfo, Manifest, ProcessingConfig, SegmentInfo, Stats, write_manifest,
};
use crate::output::{srt::write_srt, vtt::write_vtt};
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

    let transcript = if should_segment {
        transcribe_segmented(engine, input_path, total_duration, &config).await?
    } else {
        transcribe_with_spinner("Transcribing...", engine.transcribe_path(input_path)).await?
    };

    let processing_time = started.elapsed().as_secs_f64();

    // Output
    match config.output_format {
        OutputFormat::Text => {
            let text = transcript.text();
            if let Some(ref dir) = config.output_dir {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
                let stem = config
                    .input
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy();
                let text_path = dir.join(format!("{stem}.txt"));
                std::fs::write(&text_path, text.as_bytes()).with_context(|| {
                    format!("Failed to write text output: {}", text_path.display())
                })?;
                eprintln!("Text written to {}", text_path.display());
            } else {
                println!("{text}");
            }
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
        OutputFormat::Srt => {
            if let Some(ref dir) = config.output_dir {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
                let stem = config
                    .input
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy();
                let srt_path = dir.join(format!("{stem}.srt"));
                let mut file = std::fs::File::create(&srt_path)?;
                write_srt(&transcript, &mut file)?;
                eprintln!("SRT written to {}", srt_path.display());
            } else {
                let mut stdout = std::io::stdout();
                write_srt(&transcript, &mut stdout)?;
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
                language: config.language.clone(),
                normalized_audio: config.normalize_audio,
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
mod tests {
    use super::{OutputFormat, PipelineConfig, run_pipeline};
    use crate::transcriber::{Segment, Transcriber, Transcript};
    use anyhow::Result;
    use async_trait::async_trait;
    use hound::WavSpec;
    use std::f32::consts::PI;
    use std::path::Path;
    use std::process::Command;
    use tempfile::tempdir;

    use serde_json::Value;

    #[tokio::test]
    async fn pipeline_end_to_end_writes_vtt_and_manifest() -> Result<()> {
        if !command_exists("ffprobe") {
            eprintln!("Skipping integration test: ffprobe not available");
            return Ok(());
        }

        let workdir = tempdir()?;
        let input_path = workdir.path().join("sample.wav");
        write_test_wav(&input_path, 1_000)?;

        let output_dir = workdir.path().join("out");

        run_pipeline(
            &FakeTranscriber,
            PipelineConfig {
                input: input_path.clone(),
                output_dir: Some(output_dir.clone()),
                output_format: OutputFormat::Vtt,
                language: None,
                segment: false,
                silence_threshold: -40.0,
                min_silence_duration: 0.8,
                max_segment_secs: 600.0,
                provider_name: "fake".into(),
                model_name: "fake-model".into(),
                auto_split_for_api: false,
                upload_as_mp3: false,
                segment_concurrency: 1,
                normalize_audio: false,
            },
        )
        .await?;

        let vtt_path = output_dir.join("sample.vtt");
        let manifest_path = output_dir.join("sample.manifest.json");

        assert!(vtt_path.exists());
        assert!(manifest_path.exists());

        let vtt = std::fs::read_to_string(vtt_path)?;
        assert!(vtt.starts_with("WEBVTT\n"));
        assert!(vtt.contains("integration"));

        let manifest_data = std::fs::read_to_string(manifest_path)?;
        let manifest: Value = serde_json::from_str(&manifest_data)?;

        assert_eq!(
            manifest["input"]["file"],
            input_path.to_string_lossy().to_string()
        );
        assert_eq!(manifest["config"]["provider"], "fake");
        assert_eq!(manifest["config"]["model"], "fake-model");
        assert_eq!(manifest["segments"][0]["start_secs"], 0.0);
        assert_eq!(manifest["segments"][0]["end_secs"], 1.0);
        assert_eq!(manifest["segments"][0]["text"], "integration");

        Ok(())
    }

    #[tokio::test]
    async fn pipeline_end_to_end_writes_text_file_and_manifest() -> Result<()> {
        if !command_exists("ffprobe") {
            eprintln!("Skipping integration test: ffprobe not available");
            return Ok(());
        }

        let workdir = tempdir()?;
        let input_path = workdir.path().join("sample.wav");
        write_test_wav(&input_path, 1_000)?;

        let output_dir = workdir.path().join("out");

        run_pipeline(
            &FakeTranscriber,
            PipelineConfig {
                input: input_path.clone(),
                output_dir: Some(output_dir.clone()),
                output_format: OutputFormat::Text,
                language: None,
                segment: false,
                silence_threshold: -40.0,
                min_silence_duration: 0.8,
                max_segment_secs: 600.0,
                provider_name: "fake".into(),
                model_name: "fake-model".into(),
                auto_split_for_api: false,
                upload_as_mp3: false,
                segment_concurrency: 1,
                normalize_audio: false,
            },
        )
        .await?;

        let text_path = output_dir.join("sample.txt");
        let manifest_path = output_dir.join("sample.manifest.json");

        assert!(text_path.exists());
        assert!(manifest_path.exists());

        let text = std::fs::read_to_string(text_path)?;
        assert_eq!(text, "integration");

        Ok(())
    }

    #[tokio::test]
    async fn pipeline_end_to_end_writes_srt_file_and_manifest() -> Result<()> {
        if !command_exists("ffprobe") {
            eprintln!("Skipping integration test: ffprobe not available");
            return Ok(());
        }

        let workdir = tempdir()?;
        let input_path = workdir.path().join("sample.wav");
        write_test_wav(&input_path, 1_000)?;

        let output_dir = workdir.path().join("out");

        run_pipeline(
            &FakeTranscriber,
            PipelineConfig {
                input: input_path.clone(),
                output_dir: Some(output_dir.clone()),
                output_format: OutputFormat::Srt,
                language: Some("en".to_string()),
                segment: false,
                silence_threshold: -40.0,
                min_silence_duration: 0.8,
                max_segment_secs: 600.0,
                provider_name: "fake".into(),
                model_name: "fake-model".into(),
                auto_split_for_api: false,
                upload_as_mp3: false,
                segment_concurrency: 1,
                normalize_audio: false,
            },
        )
        .await?;

        let srt_path = output_dir.join("sample.srt");
        let manifest_path = output_dir.join("sample.manifest.json");

        assert!(srt_path.exists());
        assert!(manifest_path.exists());

        let srt = std::fs::read_to_string(srt_path)?;
        assert!(srt.contains("1"));
        assert!(srt.contains("00:00:00,000 --> 00:00:01,000"));
        assert!(srt.contains("integration"));

        Ok(())
    }

    #[tokio::test]
    async fn pipeline_segmented_api_uploads_are_processed_concurrently() -> Result<()> {
        if !command_exists("ffprobe") {
            eprintln!("Skipping integration test: ffprobe not available");
            return Ok(());
        }

        let workdir = tempdir()?;
        let input_path = workdir.path().join("sample.wav");
        write_test_wav(&input_path, 10_000)?;

        let output_dir = workdir.path().join("out");

        run_pipeline(
            &FakeApiTranscriber,
            PipelineConfig {
                input: input_path.clone(),
                output_dir: Some(output_dir.clone()),
                output_format: OutputFormat::Text,
                language: None,
                segment: true,
                silence_threshold: -40.0,
                min_silence_duration: 0.8,
                max_segment_secs: 5.0,
                provider_name: "fake-api".into(),
                model_name: "fake-model".into(),
                auto_split_for_api: false,
                upload_as_mp3: true,
                segment_concurrency: 2,
                normalize_audio: false,
            },
        )
        .await?;

        let manifest_path = output_dir.join("sample.manifest.json");
        let manifest_data = std::fs::read_to_string(manifest_path)?;
        let manifest: Value = serde_json::from_str(&manifest_data)?;
        let segments = manifest["segments"]
            .as_array()
            .expect("manifest segments should be an array");
        assert_eq!(segments.len(), 2);
        assert_eq!(manifest["config"]["provider"], "fake-api");

        Ok(())
    }

    fn command_exists(command: &str) -> bool {
        Command::new(command)
            .arg("-version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn write_test_wav(path: &Path, duration_ms: u64) -> Result<()> {
        let sample_rate = 16_000u32;
        let sample_count = (sample_rate as u64 * duration_ms / 1000) as usize;

        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)?;
        for i in 0..sample_count {
            let angle = i as f32 / sample_rate as f32 * 440.0 * 2.0 * PI;
            let sample = (i16::MAX as f32 * 0.25 * angle.sin()) as i16;
            writer.write_sample(sample)?;
        }

        writer.finalize()?;
        Ok(())
    }

    struct FakeTranscriber;

    #[async_trait]
    impl Transcriber for FakeTranscriber {
        async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
            Ok(Transcript {
                segments: vec![Segment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "integration".to_string(),
                }],
            })
        }
    }

    struct FakeApiTranscriber;

    #[async_trait]
    impl Transcriber for FakeApiTranscriber {
        async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
            Ok(Transcript {
                segments: vec![Segment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "integration".to_string(),
                }],
            })
        }

        async fn transcribe_path(&self, _wav_path: &Path) -> Result<Transcript> {
            self.transcribe(Vec::new()).await
        }
    }
}
