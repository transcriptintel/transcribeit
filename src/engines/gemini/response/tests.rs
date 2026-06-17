use super::{GeminiResponseContext, parse_generate_response, parse_stream_generate_response};
use serde_json::json;

fn context(upload_method: &'static str, file_url_present: bool) -> GeminiResponseContext<'static> {
    GeminiResponseContext {
        model: "gemini-test",
        api_base_url: "https://example.com",
        mime_type: "audio/mp3",
        input_bytes: 12,
        duration_secs: None,
        upload_method,
        file_url_present,
    }
}

#[test]
fn parses_structured_transcript_segments() {
    let body = br#"{
        "candidates": [{
            "finishReason": "STOP",
            "content": {
                "parts": [{
                    "text": "{\"text\":\"hello world\",\"segments\":[{\"start_secs\":1.2,\"end_secs\":2.4,\"speaker\":\"A\",\"language\":\"en\",\"emotion\":\"Neutral\",\"text\":\"hello\"}]}"
                }]
            }
        }],
        "usageMetadata": {"totalTokenCount": 42}
    }"#;

    let transcript = parse_generate_response(body, context("files_api", false));
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "hello");
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("A"));
    assert_eq!(transcript.segments[0].language.as_deref(), Some("en"));
    assert_eq!(transcript.segments[0].emotion.as_deref(), Some("Neutral"));
    assert_eq!(transcript.segments[0].start_ms, 1200);
    assert_eq!(transcript.segments[0].end_ms, 2400);
}

#[test]
fn falls_back_to_top_level_text_when_segments_are_invalid() {
    let body = br#"{
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "```json\n{\"text\":\"fallback text\",\"segments\":[{\"start_secs\":null,\"end_secs\":null,\"speaker\":null,\"language\":null,\"emotion\":null,\"text\":\"  \"}]}\n```"
                }]
            }
        }]
    }"#;

    let transcript = parse_generate_response(body, context("files_api", false));
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "fallback text");
    assert_eq!(transcript.segments[0].start_ms, 0);
}

#[test]
fn falls_back_to_raw_generated_text_when_json_is_invalid() {
    let body = br#"{
        "candidates": [{
            "content": {
                "parts": [{"text": "plain transcript"}]
            }
        }]
    }"#;

    let transcript = parse_generate_response(body, context("files_api", false));
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "plain transcript");
}

#[test]
fn clamps_timestamps_to_known_audio_duration() {
    let body = br#"{
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "{\"text\":\"hello\",\"segments\":[{\"start_secs\":299,\"end_secs\":500,\"text\":\"hello\"}]}"
                }]
            }
        }]
    }"#;

    let transcript = parse_generate_response(
        body,
        GeminiResponseContext {
            duration_secs: Some(300.0),
            ..context("files_api", false)
        },
    );
    assert_eq!(transcript.segments[0].start_ms, 299000);
    assert_eq!(transcript.segments[0].end_ms, 300000);
    assert_eq!(
        transcript
            .provider_metadata
            .as_ref()
            .and_then(|value| value.pointer("/gemini/response/timestamps_clamped"))
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[test]
fn parses_streamed_response_chunks() {
    let chunks = vec![
        json!({
            "candidates": [{
                "content": {"parts": [{"text": "{\"text\":\"hello "}]}
            }]
        }),
        json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": "world\",\"segments\":[{\"start_secs\":0,\"end_secs\":1,\"speaker\":\"A\",\"text\":\"hello world\"}]}"}]}
            }],
            "usageMetadata": {"totalTokenCount": 100}
        }),
    ];

    let transcript = parse_stream_generate_response(&chunks, context("files_api", false));
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "hello world");
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("A"));
    assert_eq!(
        transcript
            .provider_metadata
            .as_ref()
            .and_then(|value| value.pointer("/gemini/response/streaming"))
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(
        transcript
            .provider_metadata
            .as_ref()
            .and_then(|value| value.pointer("/gemini/response/chunk_count"))
            .and_then(serde_json::Value::as_u64),
        Some(2)
    );
}

#[test]
fn records_signed_url_upload_method_without_persisting_url() {
    let body = br#"{
        "candidates": [{
            "content": {
                "parts": [{"text": "{\"text\":\"signed url transcript\",\"segments\":[{\"text\":\"signed url transcript\"}]}"}]
            }
        }]
    }"#;

    let transcript = parse_generate_response(body, context("signed_url", true));
    let metadata = transcript
        .provider_metadata
        .as_ref()
        .expect("metadata should exist");

    assert_eq!(
        metadata
            .pointer("/gemini/upload_method")
            .and_then(serde_json::Value::as_str),
        Some("signed_url")
    );
    assert_eq!(
        metadata
            .pointer("/gemini/request/file_url_present")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert!(!metadata.to_string().contains("X-Amz-Signature"));
}
