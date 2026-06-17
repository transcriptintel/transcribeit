use super::{OutputFormat, PipelineConfig, run_pipeline};
use crate::transcriber::{Segment, Transcriber, Transcript};
use anyhow::Result;
use async_trait::async_trait;
use hound::WavSpec;
use serde_json::{Value, json};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

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
            provider_name: "fake".into(),
            model_name: "fake-model".into(),
            ..test_config(input_path.clone(), output_dir.clone(), OutputFormat::Vtt)
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
    assert_eq!(manifest["schema_version"], "transcribeit.manifest.v2");
    assert_eq!(manifest["input"]["duration_ms"], 1000);
    assert_eq!(manifest["segments"][0]["start_secs"], 0.0);
    assert_eq!(manifest["segments"][0]["end_secs"], 1.0);
    assert_eq!(manifest["segments"][0]["id"], "seg_000001");
    assert_eq!(manifest["segments"][0]["start_ms"], 0);
    assert_eq!(manifest["segments"][0]["end_ms"], 1000);
    assert_eq!(manifest["segments"][0]["text"], "integration");
    assert_eq!(manifest["transcript"]["text"], "integration");
    assert_eq!(manifest["transcript"]["segments"][0]["id"], "seg_000001");
    assert_eq!(manifest["capabilities"]["segments"], true);
    assert_eq!(manifest["capabilities"]["word_timestamps"], false);
    assert_eq!(manifest["quality"]["timing_source"], "unknown");
    assert_eq!(manifest["quality"]["timing_reliable"], false);
    assert!(manifest.get("provider_metadata").is_none());

    Ok(())
}

#[tokio::test]
async fn pipeline_manifest_wraps_provider_metadata_in_stable_envelope() -> Result<()> {
    if !command_exists("ffprobe") {
        eprintln!("Skipping integration test: ffprobe not available");
        return Ok(());
    }

    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeMetadataTranscriber,
        PipelineConfig {
            language: Some("en".to_string()),
            provider_name: "gemini".into(),
            model_name: "gemini-test".into(),
            ..test_config(input_path.clone(), output_dir.clone(), OutputFormat::Text)
        },
    )
    .await?;

    let manifest_path = output_dir.join("sample.manifest.json");
    let manifest_data = std::fs::read_to_string(manifest_path)?;
    let manifest: Value = serde_json::from_str(&manifest_data)?;

    assert_eq!(manifest["provider_metadata"]["provider"], "gemini");
    assert_eq!(
        manifest["provider_metadata"]["schema_version"],
        "gemini.metadata.v1"
    );
    assert_eq!(
        manifest["provider_metadata"]["data"]["response"]["timestamps_clamped"],
        true
    );
    assert_eq!(
        manifest["provider_metadata"]["data"]["file"]["deleted"],
        true
    );
    assert_eq!(manifest["quality"]["timing_source"], "model_generated");
    assert_eq!(manifest["quality"]["timing_reliable"], false);
    assert_eq!(manifest["quality"]["timestamps_clamped"], true);
    assert_eq!(manifest["quality"]["speaker_source"], "model_generated");
    assert!(
        manifest["quality"]["warnings"]
            .as_array()
            .expect("warnings should be an array")
            .iter()
            .any(|warning| warning
                .as_str()
                .is_some_and(|text| text.contains("model-generated")))
    );

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
            provider_name: "fake".into(),
            model_name: "fake-model".into(),
            ..test_config(input_path.clone(), output_dir.clone(), OutputFormat::Text)
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
            language: Some("en".to_string()),
            provider_name: "fake".into(),
            model_name: "fake-model".into(),
            ..test_config(input_path.clone(), output_dir.clone(), OutputFormat::Srt)
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
            segment: true,
            max_segment_secs: 5.0,
            provider_name: "fake-api".into(),
            model_name: "fake-model".into(),
            upload_as_mp3: true,
            segment_concurrency: 2,
            ..test_config(input_path.clone(), output_dir.clone(), OutputFormat::Text)
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

fn test_config(input: PathBuf, output_dir: PathBuf, output_format: OutputFormat) -> PipelineConfig {
    PipelineConfig {
        input,
        output_dir: Some(output_dir),
        output_format,
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
        diarize: false,
        speakers: None,
        diarize_segmentation_model: None,
        diarize_embedding_model: None,
        vad_model: None,
    }
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
                speaker: None,
                ..Default::default()
            }],
            provider_metadata: None,
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
                speaker: None,
                ..Default::default()
            }],
            provider_metadata: None,
        })
    }

    async fn transcribe_path(&self, _wav_path: &Path) -> Result<Transcript> {
        self.transcribe(Vec::new()).await
    }
}

struct FakeMetadataTranscriber;

#[async_trait]
impl Transcriber for FakeMetadataTranscriber {
    async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
        Ok(Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 1000,
                text: "metadata".to_string(),
                speaker: Some("Speaker 1".to_string()),
                language: Some("en".to_string()),
                emotion: Some("neutral".to_string()),
                ..Default::default()
            }],
            provider_metadata: Some(json!({
                "gemini": {
                    "response": {
                        "timestamps_clamped": true
                    },
                    "file": {
                        "deleted": true
                    }
                }
            })),
        })
    }
}
