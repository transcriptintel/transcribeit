use serde_json::{Value, json};

use crate::transcriber::{Segment, Transcript};

pub fn parse_generate_response(
    body: &[u8],
    model: &str,
    api_base_url: &str,
    mime_type: &str,
    input_bytes: u64,
    duration_secs: Option<f64>,
) -> Transcript {
    let response_value = serde_json::from_slice::<Value>(body).unwrap_or_else(|_| {
        json!({
            "raw_text": String::from_utf8_lossy(body).to_string()
        })
    });
    let generated_text = generated_text(&response_value).unwrap_or_default();
    let generated_json = parse_generated_json(&generated_text);
    let segments = generated_json
        .as_ref()
        .and_then(|value| parse_transcript_segments(value, duration_secs))
        .filter(|segments| !segments.is_empty())
        .or_else(|| {
            generated_json
                .as_ref()
                .and_then(|value| value.get("text").and_then(Value::as_str))
                .map(single_text_segment)
        })
        .unwrap_or_else(|| single_text_segment(&generated_text));

    Transcript {
        segments,
        provider_metadata: Some(json!({
            "gemini": {
                "model": model,
                "api_base_url": api_base_url,
                "upload_method": "files_api",
                "input": {
                    "mime_type": mime_type,
                    "bytes": input_bytes,
                    "duration_secs": duration_secs,
                },
                "response": {
                    "generated_json_valid": generated_json.is_some(),
                    "timestamps_clamped": generated_json
                        .as_ref()
                        .is_some_and(|value| timestamps_need_clamp(value, duration_secs)),
                    "candidate_count": response_value
                        .get("candidates")
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len),
                    "finish_reasons": finish_reasons(&response_value),
                    "usage_metadata": response_value.get("usageMetadata").cloned(),
                    "prompt_feedback": response_value.get("promptFeedback").cloned(),
                }
            }
        })),
    }
}

fn generated_text(response: &Value) -> Option<String> {
    let parts = response
        .get("candidates")?
        .as_array()?
        .first()?
        .get("content")?
        .get("parts")?
        .as_array()?;

    let text = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");
    (!text.trim().is_empty()).then_some(text)
}

fn parse_generated_json(text: &str) -> Option<Value> {
    let trimmed = strip_json_fence(text.trim());
    serde_json::from_str(trimmed).ok()
}

fn strip_json_fence(text: &str) -> &str {
    let Some(without_prefix) = text.strip_prefix("```") else {
        return text;
    };
    let without_lang = without_prefix
        .strip_prefix("json")
        .unwrap_or(without_prefix)
        .trim_start();
    without_lang
        .strip_suffix("```")
        .unwrap_or(without_lang)
        .trim()
}

fn parse_transcript_segments(
    value: &Value,
    max_duration_secs: Option<f64>,
) -> Option<Vec<Segment>> {
    let segments = value.get("segments")?.as_array()?;
    Some(
        segments
            .iter()
            .filter_map(|segment| parse_transcript_segment(segment, max_duration_secs))
            .collect(),
    )
}

fn parse_transcript_segment(value: &Value, max_duration_secs: Option<f64>) -> Option<Segment> {
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())?;
    let start_ms = timestamp_ms(
        value
            .get("start_secs")
            .or_else(|| value.get("start"))
            .or_else(|| value.get("begin_secs")),
    )
    .unwrap_or(0);
    let mut end_ms = timestamp_ms(
        value
            .get("end_secs")
            .or_else(|| value.get("end"))
            .or_else(|| value.get("end_time")),
    )
    .unwrap_or(start_ms);
    if let Some(max_ms) = max_duration_secs.map(|seconds| (seconds * 1000.0).round() as i64) {
        end_ms = end_ms.min(max_ms);
    }
    let start_ms = if end_ms > 0 {
        start_ms.min(end_ms)
    } else {
        start_ms
    };

    Some(Segment {
        start_ms,
        end_ms,
        text: text.to_string(),
        speaker: string_field(value, &["speaker", "speaker_id"]),
        language: string_field(value, &["language", "lang"]),
        emotion: string_field(value, &["emotion"]),
        words: Vec::new(),
    })
}

fn timestamps_need_clamp(value: &Value, duration_secs: Option<f64>) -> bool {
    let Some(max_ms) = duration_secs.map(|seconds| (seconds * 1000.0).round() as i64) else {
        return false;
    };
    value
        .get("segments")
        .and_then(Value::as_array)
        .is_some_and(|segments| {
            segments.iter().any(|segment| {
                timestamp_ms(segment.get("start_secs").or_else(|| segment.get("start")))
                    .is_some_and(|start_ms| start_ms > max_ms)
                    || timestamp_ms(segment.get("end_secs").or_else(|| segment.get("end")))
                        .is_some_and(|end_ms| end_ms > max_ms)
            })
        })
}

fn timestamp_ms(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    if value.is_null() {
        return None;
    }
    let seconds = match value {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse().ok()?,
        _ => return None,
    };
    Some((seconds * 1000.0).round() as i64)
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    names
        .iter()
        .filter_map(|name| value.get(name))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

fn single_text_segment(text: &str) -> Vec<Segment> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    vec![Segment {
        start_ms: 0,
        end_ms: 0,
        text: text.to_string(),
        speaker: None,
        ..Default::default()
    }]
}

fn finish_reasons(response: &Value) -> Vec<Value> {
    response
        .get("candidates")
        .and_then(Value::as_array)
        .map(|candidates| {
            candidates
                .iter()
                .filter_map(|candidate| candidate.get("finishReason").cloned())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::parse_generate_response;

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
}
