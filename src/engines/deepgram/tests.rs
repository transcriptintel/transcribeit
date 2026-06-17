use serde_json::json;

use super::parse_response;

#[test]
fn parses_utterances_with_speaker_and_words() {
    let body = json!({
        "metadata": {
            "request_id": "req-1",
            "duration": 2.4,
            "model_info": {
                "model-id": {
                    "name": "nova-3",
                    "version": "2026-01-01",
                    "arch": "nova-3"
                }
            }
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello there.",
                            "confidence": 0.98,
                            "words": []
                        }
                    ]
                }
            ],
            "utterances": [
                {
                    "start": 0.12,
                    "end": 1.5,
                    "speaker": 0,
                    "transcript": "Hello there.",
                    "words": [
                        {
                            "word": "hello",
                            "punctuated_word": "Hello",
                            "start": 0.12,
                            "end": 0.6,
                            "speaker": 0
                        },
                        {
                            "word": "there",
                            "punctuated_word": "there.",
                            "start": 0.6,
                            "end": 1.5,
                            "speaker": 0
                        }
                    ]
                }
            ]
        }
    })
    .to_string();

    let transcript = parse_response(
        body.as_bytes(),
        "nova-3",
        "https://api.deepgram.com/v1",
        "direct_upload",
    )
    .expect("response should parse");

    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].start_ms, 120);
    assert_eq!(transcript.segments[0].end_ms, 1500);
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("Speaker 0"));
    assert_eq!(transcript.segments[0].words.len(), 2);
    assert_eq!(transcript.segments[0].words[1].text, "there.");
    assert_eq!(
        transcript
            .provider_metadata
            .as_ref()
            .and_then(|value| value.pointer("/data/response/utterance_count"))
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
}

#[test]
fn falls_back_to_paragraph_sentences() {
    let body = json!({
        "metadata": {"request_id": "req-2"},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "First sentence. Second sentence.",
                            "confidence": 0.9,
                            "words": [
                                {"word": "first", "punctuated_word": "First", "start": 0.0, "end": 0.4},
                                {"word": "sentence", "punctuated_word": "sentence.", "start": 0.4, "end": 1.0},
                                {"word": "second", "punctuated_word": "Second", "start": 1.1, "end": 1.5},
                                {"word": "sentence", "punctuated_word": "sentence.", "start": 1.5, "end": 2.1}
                            ],
                            "paragraphs": {
                                "paragraphs": [
                                    {
                                        "sentences": [
                                            {"text": "First sentence.", "start": 0.0, "end": 1.0},
                                            {"text": "Second sentence.", "start": 1.1, "end": 2.1}
                                        ]
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    })
    .to_string();

    let transcript = parse_response(
        body.as_bytes(),
        "nova-3",
        "https://api.deepgram.com/v1",
        "direct_upload",
    )
    .expect("response should parse");

    assert_eq!(transcript.segments.len(), 2);
    assert_eq!(transcript.segments[0].text, "First sentence.");
    assert_eq!(transcript.segments[1].start_ms, 1100);
    assert_eq!(transcript.segments[1].words.len(), 2);
}

#[test]
fn clamps_timestamps_to_metadata_duration() {
    let body = json!({
        "metadata": {"duration": 1.0},
        "results": {
            "utterances": [
                {
                    "start": 0.5,
                    "end": 1.2,
                    "transcript": "Too long.",
                    "words": [
                        {"word": "too", "start": 0.5, "end": 0.7},
                        {"word": "long", "punctuated_word": "long.", "start": 0.7, "end": 1.2}
                    ]
                }
            ]
        }
    })
    .to_string();

    let transcript = parse_response(
        body.as_bytes(),
        "nova-3",
        "https://api.deepgram.com/v1",
        "direct_upload",
    )
    .expect("response should parse");

    assert_eq!(transcript.segments[0].end_ms, 1000);
    assert_eq!(transcript.segments[0].words[1].end_ms, 1000);
    assert_eq!(
        transcript
            .provider_metadata
            .as_ref()
            .and_then(|value| value.pointer("/data/response/timestamps_clamped"))
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[test]
fn records_presigned_url_request_metadata_without_url() {
    let body = json!({
        "metadata": {"request_id": "req-3"},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Remote file.",
                            "confidence": 0.9,
                            "words": []
                        }
                    ]
                }
            ]
        }
    })
    .to_string();

    let transcript = parse_response(
        body.as_bytes(),
        "nova-3",
        "https://api.deepgram.com/v1",
        "presigned_url",
    )
    .expect("response should parse");

    let metadata = transcript
        .provider_metadata
        .as_ref()
        .expect("provider metadata should exist");
    assert_eq!(
        metadata
            .pointer("/data/request/audio_source")
            .and_then(serde_json::Value::as_str),
        Some("presigned_url")
    );
    assert_eq!(
        metadata
            .pointer("/data/request/file_url_present")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert!(!metadata.to_string().contains("https://signed.example"));
}
