use std::path::PathBuf;

use super::build_quality;
use crate::analysis::AnalysisConfig;
use crate::pipeline::{OutputFormat, PipelineConfig};
use crate::transcriber::{Segment, Transcript};

#[test]
fn quality_warns_on_zero_duration_and_non_monotonic_timestamps() {
    let transcript = Transcript {
        segments: vec![
            Segment {
                start_ms: 1000,
                end_ms: 2000,
                text: "first".to_string(),
                ..Default::default()
            },
            Segment {
                start_ms: 500,
                end_ms: 500,
                text: "second".to_string(),
                ..Default::default()
            },
        ],
        provider_metadata: None,
    };
    let quality = build_quality(&test_config("gemini"), &transcript);

    assert!(
        quality
            .warnings
            .iter()
            .any(|warning| warning.contains("Segment timestamps are not monotonic"))
    );
    assert!(
        quality
            .warnings
            .iter()
            .any(|warning| warning.contains("1 segment(s) have zero-duration timestamps"))
    );
}

#[test]
fn one_valid_segment_does_not_make_all_timing_reliable() {
    let transcript = Transcript {
        segments: vec![
            Segment {
                start_ms: 0,
                end_ms: 1_000,
                text: "valid".to_string(),
                ..Default::default()
            },
            Segment {
                start_ms: 1_000,
                end_ms: 1_000,
                text: "untimed".to_string(),
                ..Default::default()
            },
        ],
        provider_metadata: None,
    };
    let quality = build_quality(&test_config("openai"), &transcript);

    assert!(!quality.timing_reliable);
}

fn test_config(provider: &str) -> PipelineConfig {
    PipelineConfig {
        input: PathBuf::from("sample.wav"),
        output_dir: None,
        output_format: OutputFormat::Vtt,
        language: None,
        normalize_audio: false,
        segment: false,
        silence_threshold: -40.0,
        min_silence_duration: 0.8,
        max_segment_secs: 600.0,
        segment_concurrency: 1,
        auto_split_max_bytes: None,
        upload_as_mp3: false,
        analysis: AnalysisConfig::default(),
        provider_name: provider.to_string(),
        model_name: "test-model".to_string(),
    }
}
