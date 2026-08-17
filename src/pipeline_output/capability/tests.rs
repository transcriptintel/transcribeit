use std::path::PathBuf;

use super::build_capabilities;
use crate::analysis::AnalysisConfig;
use crate::pipeline::{OutputFormat, PipelineConfig};
use crate::transcriber::{Segment, Transcript};

#[test]
fn untimed_openai_text_does_not_claim_native_timestamps() {
    let transcript = Transcript {
        segments: vec![Segment {
            text: "hello".to_string(),
            ..Default::default()
        }],
        provider_metadata: None,
    };
    let config = test_config("openai");

    let capabilities = build_capabilities(&config, &transcript);

    assert!(!capabilities.native_timestamps);
}

#[test]
fn apple_speech_timed_segments_claim_native_timestamps() {
    let transcript = Transcript {
        segments: vec![Segment {
            start_ms: 100,
            end_ms: 900,
            text: "hello".to_string(),
            language: Some("en_US".to_string()),
            ..Default::default()
        }],
        provider_metadata: None,
    };
    let capabilities = build_capabilities(&test_config("apple-speech"), &transcript);

    assert!(capabilities.native_timestamps);
    assert!(capabilities.language_per_segment);
    assert!(!capabilities.speaker_labels);
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
