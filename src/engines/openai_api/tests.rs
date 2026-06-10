use std::time::Duration;

use super::{OpenAiApi, is_response_format_not_supported, parse_response_bytes};
use crate::engines::rate_limit::ApiRequestSettings;

fn api_for_model(model: &str) -> OpenAiApi {
    OpenAiApi::new(
        "https://api.openai.com".into(),
        "test-key".into(),
        model.into(),
        Some("en".into()),
        ApiRequestSettings::new(
            Duration::from_secs(30),
            0,
            Duration::from_secs(1),
            Duration::from_secs(1),
        ),
    )
    .unwrap()
}

#[test]
fn diarize_model_prefers_diarized_json_with_plain_fallback() {
    let api = api_for_model("gpt-4o-transcribe-diarize");
    assert_eq!(api.response_formats(), vec![Some("diarized_json"), None]);
}

#[test]
fn non_diarize_model_prefers_verbose_json_with_plain_fallback() {
    let api = api_for_model("gpt-4o-mini-transcribe");
    assert_eq!(api.response_formats(), vec![Some("verbose_json"), None]);
}

#[test]
fn parses_diarized_segments_with_speakers() {
    let body = br#"{
        "text": "hello world",
        "segments": [
            {
                "type": "transcript.text.segment",
                "text": "hello",
                "speaker": "A",
                "start": 1.25,
                "end": 2.5,
                "id": "seg_0"
            },
            {
                "type": "transcript.text.segment",
                "text": "world",
                "speaker": "B",
                "start": "2.50",
                "end": "3.75"
            }
        ],
        "future_field": {"kept_by_openai": true}
    }"#;

    let transcript = parse_response_bytes(body);
    assert_eq!(transcript.segments.len(), 2);
    assert_eq!(transcript.segments[0].text, "hello");
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("A"));
    assert_eq!(transcript.segments[0].start_ms, 1250);
    assert_eq!(transcript.segments[0].end_ms, 2500);
    assert_eq!(transcript.segments[1].speaker.as_deref(), Some("B"));
    assert_eq!(transcript.segments[1].start_ms, 2500);
    assert_eq!(transcript.segments[1].end_ms, 3750);
}

#[test]
fn skips_invalid_segments_and_falls_back_to_text_when_needed() {
    let body = br#"{
        "text": "fallback transcript",
        "segments": [
            {"speaker": "A", "start": 0, "end": 1},
            {"text": "   "}
        ]
    }"#;

    let transcript = parse_response_bytes(body);
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "fallback transcript");
    assert_eq!(transcript.segments[0].start_ms, 0);
    assert_eq!(transcript.segments[0].end_ms, 0);
}

#[test]
fn parses_segments_without_timestamps_without_crashing() {
    let body = br#"{
        "segments": [
            {"text": "untimed", "speaker_id": "speaker-1"}
        ]
    }"#;

    let transcript = parse_response_bytes(body);
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "untimed");
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("speaker-1"));
    assert_eq!(transcript.segments[0].start_ms, 0);
    assert_eq!(transcript.segments[0].end_ms, 0);
}

#[test]
fn detects_structured_openai_error_param() {
    let body = r#"{"error":{"message":"Unsupported value: 'response_format'","param":"response_format","type":"invalid_request_error"}}"#;
    assert!(is_response_format_not_supported(body));
}

#[test]
fn detects_structured_azure_error_message() {
    let body = r#"{"error":{"code":"BadRequest","message":"The request is invalid. Parameter 'response_format' is invalid."}}"#;
    assert!(is_response_format_not_supported(body));
}

#[test]
fn falls_back_to_structured_error_code() {
    let body = r#"{"error":{"code":"response_format","message":"Unsupported value"}}"#;
    assert!(is_response_format_not_supported(body));
}

#[test]
fn rejects_unrelated_errors() {
    let body =
        r#"{"error":{"message":"Model not found","param":"model","code":"model_not_found"}}"#;
    assert!(!is_response_format_not_supported(body));
}

#[test]
fn detects_plaintext_error_when_param_missing() {
    let body = "unsupported value 'response_format' for audio transcription";
    assert!(is_response_format_not_supported(body));
}
