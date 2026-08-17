use super::{OutputFormat, PipelineConfig, run_pipeline};
use crate::analysis::{AnalysisConfig, AnalysisResult, SummaryAnalysis, TranscriptAnalyzer};
use crate::transcriber::{Segment, Transcriber, Transcript, Word};
use anyhow::Result;
use async_trait::async_trait;
use hound::WavSpec;
use serde_json::{Value, json};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tempfile::tempdir;

#[tokio::test]
async fn pipeline_end_to_end_writes_vtt_and_manifest() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeTranscriber,
        None,
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
async fn pipeline_writes_vtt_for_isolated_zero_duration_segment() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;
    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeZeroDurationTranscriber,
        None,
        PipelineConfig {
            provider_name: "local".into(),
            model_name: "large-v3".into(),
            ..test_config(input_path, output_dir.clone(), OutputFormat::Vtt)
        },
    )
    .await?;

    let vtt = std::fs::read_to_string(output_dir.join("sample.vtt"))?;
    assert_eq!(vtt.matches(" --> ").count(), 1);
    assert!(vtt.contains("timed text\nboundary text"));

    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(
        output_dir.join("sample.manifest.json"),
    )?)?;
    assert_eq!(manifest["transcript"]["segments"][1]["start_ms"], 1_000);
    assert_eq!(manifest["transcript"]["segments"][1]["end_ms"], 1_000);
    assert_eq!(manifest["quality"]["timing_reliable"], false);
    assert!(
        manifest["quality"]["warnings"]
            .as_array()
            .is_some_and(|warnings| warnings.iter().any(|warning| warning
                .as_str()
                .is_some_and(|text| text.contains("zero-duration"))))
    );

    Ok(())
}

#[tokio::test]
async fn pipeline_manifest_wraps_provider_metadata_in_stable_envelope() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeMetadataTranscriber,
        None,
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
async fn pipeline_manifest_includes_analysis_when_requested() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeTranscriber,
        Some(&FakeAnalyzer),
        PipelineConfig {
            analysis: AnalysisConfig { summary: true },
            provider_name: "fake".into(),
            model_name: "fake-model".into(),
            ..test_config(input_path.clone(), output_dir.clone(), OutputFormat::Text)
        },
    )
    .await?;

    let manifest_path = output_dir.join("sample.manifest.json");
    let manifest_data = std::fs::read_to_string(manifest_path)?;
    let manifest: Value = serde_json::from_str(&manifest_data)?;

    assert_eq!(manifest["analysis"]["provider"], "fake-analysis");
    assert_eq!(manifest["analysis"]["summary"]["short"], "short summary");
    assert_eq!(
        manifest["analysis"]["provider_metadata"]["response"]["generated_json_valid"],
        true
    );

    Ok(())
}

#[tokio::test]
async fn pipeline_end_to_end_writes_text_file_and_manifest() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeTranscriber,
        None,
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

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(output_dir.join("sample.txt"))?
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(manifest_path)?.permissions().mode() & 0o777,
            0o600
        );
    }

    Ok(())
}

#[tokio::test]
async fn pipeline_persists_transcript_and_records_analysis_failure() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;
    let output_dir = workdir.path().join("out");

    let error = run_pipeline(
        &FakeTranscriber,
        Some(&FailingAnalyzer),
        PipelineConfig {
            analysis: AnalysisConfig { summary: true },
            provider_name: "fake".into(),
            model_name: "fake-model".into(),
            ..test_config(input_path, output_dir.clone(), OutputFormat::Text)
        },
    )
    .await
    .expect_err("analysis failure must remain visible to the caller");

    assert!(format!("{error:#}").contains("optional analysis failed"));
    assert_eq!(
        std::fs::read_to_string(output_dir.join("sample.txt"))?,
        "integration"
    );
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(output_dir.join("sample.manifest.json"))?)?;
    assert_eq!(manifest["transcript"]["text"], "integration");
    assert_eq!(
        manifest["analysis_error"]["message"],
        "analysis service unavailable"
    );
    assert!(manifest.get("analysis").is_none());

    Ok(())
}

#[tokio::test]
async fn pipeline_end_to_end_writes_srt_file_and_manifest() -> Result<()> {
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 1_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeTranscriber,
        None,
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
    let workdir = tempdir()?;
    let input_path = workdir.path().join("sample.wav");
    write_test_wav(&input_path, 10_000)?;

    let output_dir = workdir.path().join("out");

    run_pipeline(
        &FakeApiTranscriber,
        None,
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
    assert_eq!(segments[1]["start_ms"], 5_000);
    assert_eq!(segments[1]["words"][0]["start_ms"], 5_100);
    assert_eq!(
        manifest["provider_metadata"]["schema_version"],
        "fake-api.segmented-metadata.v1"
    );
    assert_eq!(
        manifest["provider_metadata"]["data"]["chunks"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );

    Ok(())
}

#[tokio::test]
async fn original_media_provider_receives_m4a_without_conversion() -> Result<()> {
    let workdir = tempdir()?;
    let source_wav = workdir.path().join("source.wav");
    let input_path = workdir.path().join("large-input.m4a");
    write_test_wav(&source_wav, 1_000)?;
    encode_test_m4a(&source_wav, &input_path).await?;

    run_pipeline(
        &OriginalMediaTranscriber {
            original_path: input_path.clone(),
            expect_original: true,
        },
        None,
        PipelineConfig {
            provider_name: "original-media".into(),
            model_name: "native".into(),
            ..test_config(
                input_path.clone(),
                workdir.path().join("out"),
                OutputFormat::Text,
            )
        },
    )
    .await?;

    Ok(())
}

#[tokio::test]
async fn normalization_still_prepares_wav_for_original_media_provider() -> Result<()> {
    let workdir = tempdir()?;
    let source_wav = workdir.path().join("source.wav");
    let input_path = workdir.path().join("input.m4a");
    write_test_wav(&source_wav, 1_000)?;
    encode_test_m4a(&source_wav, &input_path).await?;

    run_pipeline(
        &OriginalMediaTranscriber {
            original_path: input_path.clone(),
            expect_original: false,
        },
        None,
        PipelineConfig {
            provider_name: "original-media".into(),
            model_name: "native".into(),
            normalize_audio: true,
            ..test_config(input_path, workdir.path().join("out"), OutputFormat::Text)
        },
    )
    .await?;

    Ok(())
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
        auto_split_max_bytes: None,
        upload_as_mp3: false,
        segment_concurrency: 1,
        normalize_audio: false,
        analysis: AnalysisConfig::default(),
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

async fn encode_test_m4a(input: &Path, output: &Path) -> Result<()> {
    let status = tokio::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-c:a")
        .arg("aac")
        .arg(output)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await?;
    anyhow::ensure!(status.success(), "failed to create M4A test fixture");
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

struct OriginalMediaTranscriber {
    original_path: PathBuf,
    expect_original: bool,
}

#[async_trait]
impl Transcriber for OriginalMediaTranscriber {
    fn prefers_original_media(&self) -> bool {
        true
    }

    async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
        anyhow::bail!("path transcription was expected")
    }

    async fn transcribe_path(&self, path: &Path) -> Result<Transcript> {
        if self.expect_original {
            anyhow::ensure!(
                path == self.original_path,
                "original input path was not preserved"
            );
        } else {
            anyhow::ensure!(
                path != self.original_path,
                "normalization did not prepare media"
            );
            anyhow::ensure!(
                path.extension().and_then(|value| value.to_str()) == Some("wav"),
                "normalization did not produce WAV input"
            );
        }
        Ok(Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 1_000,
                text: "native input".to_string(),
                ..Default::default()
            }],
            provider_metadata: None,
        })
    }
}

struct FakeZeroDurationTranscriber;

#[async_trait]
impl Transcriber for FakeZeroDurationTranscriber {
    async fn transcribe(&self, _audio_samples: Vec<f32>) -> Result<Transcript> {
        Ok(Transcript {
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 1_000,
                    text: "timed text".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 1_000,
                    end_ms: 1_000,
                    text: "boundary text".to_string(),
                    ..Default::default()
                },
            ],
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
                words: vec![Word {
                    start_ms: 100,
                    end_ms: 400,
                    text: "integration".to_string(),
                    punctuation: None,
                }],
                ..Default::default()
            }],
            provider_metadata: Some(json!({"request_id": "fake-request"})),
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

struct FakeAnalyzer;

#[async_trait]
impl TranscriptAnalyzer for FakeAnalyzer {
    async fn analyze_transcript(
        &self,
        _transcript: &Transcript,
        _config: &AnalysisConfig,
    ) -> Result<AnalysisResult> {
        Ok(AnalysisResult {
            provider: "fake-analysis".to_string(),
            model: "fake-analysis-model".to_string(),
            schema_version: "transcribeit.analysis.v1".to_string(),
            summary: Some(SummaryAnalysis {
                short: "short summary".to_string(),
                detailed: "detailed summary".to_string(),
                key_points: vec!["point".to_string()],
                topics: vec!["topic".to_string()],
                action_items: Vec::new(),
                questions: vec!["question".to_string()],
                follow_ups: Vec::new(),
            }),
            provider_metadata: Some(json!({
                "response": {
                    "generated_json_valid": true
                }
            })),
        })
    }
}

struct FailingAnalyzer;

#[async_trait]
impl TranscriptAnalyzer for FailingAnalyzer {
    async fn analyze_transcript(
        &self,
        _transcript: &Transcript,
        _config: &AnalysisConfig,
    ) -> Result<AnalysisResult> {
        anyhow::bail!("analysis service unavailable")
    }
}
