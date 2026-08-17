use std::path::PathBuf;

use super::build_cache_info;
use crate::analysis::AnalysisConfig;
use crate::pipeline::{OutputFormat, PipelineConfig};
use crate::transcriber::{Segment, Transcript};

#[test]
fn cache_info_extracts_gemini_cached_tokens() {
    let transcript = transcript_with_metadata(serde_json::json!({
        "provider": "gemini",
        "schema_version": "gemini.metadata.v1",
        "data": { "response": { "usage_metadata": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 80,
            "cacheTokensDetails": [{"modality": "AUDIO", "tokenCount": 80}]
        }}}
    }));
    let cache = build_cache_info(&test_config("gemini"), &transcript, None);

    assert_eq!(cache.transcription.mode, "implicit");
    assert!(cache.transcription.hit);
    assert_eq!(cache.transcription.input_tokens, Some(100));
    assert_eq!(cache.transcription.cached_tokens, Some(80));
    assert_eq!(cache.transcription.cached_fraction, Some(0.8));
    assert!(cache.transcription.token_details.is_some());
}

#[test]
fn cache_info_marks_gemini_explicit_cache_without_invalid_fraction() {
    let transcript = transcript_with_metadata(serde_json::json!({
        "provider": "gemini",
        "schema_version": "gemini.metadata.v1",
        "data": {
            "cached_content": { "enabled": true, "name": "cachedContents/test" },
            "response": { "usage_metadata": {
                "promptTokenCount": 100,
                "cachedContentTokenCount": 120,
                "cacheTokensDetails": [{"modality": "AUDIO", "tokenCount": 120}]
            }}
        }
    }));
    let cache = build_cache_info(&test_config("gemini"), &transcript, None);

    assert_eq!(cache.transcription.mode, "explicit");
    assert!(cache.transcription.hit);
    assert_eq!(cache.transcription.cached_fraction, None);
}

#[test]
fn cache_info_extracts_openai_cached_tokens() {
    let transcript = transcript_with_metadata(serde_json::json!({
        "provider": "openai",
        "schema_version": "openai.metadata.v1",
        "data": { "response": { "usage": {
            "prompt_tokens": 2048,
            "prompt_tokens_details": { "cached_tokens": 1024 }
        }}}
    }));
    let cache = build_cache_info(&test_config("openai"), &transcript, None);

    assert_eq!(cache.transcription.mode, "implicit");
    assert!(cache.transcription.hit);
    assert_eq!(cache.transcription.input_tokens, Some(2048));
    assert_eq!(cache.transcription.cached_tokens, Some(1024));
    assert_eq!(cache.transcription.cached_fraction, Some(0.5));
}

#[test]
fn cache_info_marks_qwen_as_no_provider_cache() {
    let transcript = Transcript {
        segments: vec![test_segment()],
        provider_metadata: None,
    };
    let cache = build_cache_info(&test_config("qwen-filetrans"), &transcript, None);

    assert_eq!(cache.transcription.mode, "none");
    assert!(!cache.transcription.hit);
    assert_eq!(cache.transcription.input_tokens, None);
    assert_eq!(cache.transcription.cached_tokens, None);
}

#[test]
fn cache_info_marks_apple_speech_as_no_token_cache() {
    let transcript = Transcript {
        segments: vec![test_segment()],
        provider_metadata: None,
    };
    let cache = build_cache_info(&test_config("apple-speech"), &transcript, None);

    assert_eq!(cache.transcription.mode, "none");
    assert!(!cache.transcription.hit);
    assert_eq!(cache.transcription.input_tokens, None);
}

fn transcript_with_metadata(metadata: serde_json::Value) -> Transcript {
    Transcript {
        segments: vec![test_segment()],
        provider_metadata: Some(metadata),
    }
}

fn test_segment() -> Segment {
    Segment {
        start_ms: 0,
        end_ms: 1000,
        text: "hello".to_string(),
        ..Default::default()
    }
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
