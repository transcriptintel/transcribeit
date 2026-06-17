use super::{parse_generate_response, parse_stream_generate_response};
use serde_json::json;

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

    let transcript = parse_generate_response(
        body,
        "gemini-test",
        "https://example.com",
        "audio/mp3",
        12,
        None,
    );
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

    let transcript = parse_generate_response(
        body,
        "gemini-test",
        "https://example.com",
        "audio/mp3",
        12,
        None,
    );
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

    let transcript = parse_generate_response(
        body,
        "gemini-test",
        "https://example.com",
        "audio/mp3",
        12,
        None,
    );
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
        "gemini-test",
        "https://example.com",
        "audio/mp3",
        12,
        Some(300.0),
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

    let transcript = parse_stream_generate_response(
        &chunks,
        "gemini-test",
        "https://example.com",
        "audio/mp3",
        12,
        None,
    );
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
