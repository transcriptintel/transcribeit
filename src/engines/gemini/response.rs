use serde_json::{Map, Value, json};

use crate::transcriber::{Segment, Transcript};

#[derive(Clone, Copy)]
pub(super) struct GeminiResponseContext<'a> {
    pub(super) model: &'a str,
    pub(super) api_base_url: &'a str,
    pub(super) mime_type: &'a str,
    pub(super) input_bytes: u64,
    pub(super) duration_secs: Option<f64>,
    pub(super) upload_method: &'a str,
    pub(super) file_url_present: bool,
}

#[cfg(test)]
pub fn parse_generate_response(body: &[u8], context: GeminiResponseContext<'_>) -> Transcript {
    let response_value = serde_json::from_slice::<Value>(body).unwrap_or_else(|_| {
        json!({
            "raw_text": String::from_utf8_lossy(body).to_string()
        })
    });
    let mut response_metadata = Map::new();
    response_metadata.insert("streaming".to_string(), Value::Bool(false));
    response_metadata.insert(
        "candidate_count".to_string(),
        json!(
            response_value
                .get("candidates")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
        ),
    );
    response_metadata.insert(
        "finish_reasons".to_string(),
        json!(finish_reasons(&response_value)),
    );
    response_metadata.insert(
        "usage_metadata".to_string(),
        response_value
            .get("usageMetadata")
            .cloned()
            .unwrap_or(Value::Null),
    );
    response_metadata.insert(
        "prompt_feedback".to_string(),
        response_value
            .get("promptFeedback")
            .cloned()
            .unwrap_or(Value::Null),
    );

    build_transcript_from_generated_text(
        &generated_text(&response_value).unwrap_or_default(),
        context,
        response_metadata,
    )
}

pub fn parse_stream_generate_response(
    chunks: &[Value],
    context: GeminiResponseContext<'_>,
) -> Transcript {
    let generated_text = chunks
        .iter()
        .filter_map(generated_text)
        .collect::<Vec<_>>()
        .join("");
    let finish_reasons = chunks.iter().flat_map(finish_reasons).collect::<Vec<_>>();
    let usage_metadata = chunks
        .iter()
        .rev()
        .find_map(|chunk| chunk.get("usageMetadata").cloned());
    let prompt_feedback = chunks
        .iter()
        .rev()
        .find_map(|chunk| chunk.get("promptFeedback").cloned());
    let candidate_count = chunks
        .iter()
        .filter_map(|chunk| {
            chunk
                .get("candidates")
                .and_then(Value::as_array)
                .map(Vec::len)
        })
        .max()
        .unwrap_or(0);

    let mut response_metadata = Map::new();
    response_metadata.insert("streaming".to_string(), Value::Bool(true));
    response_metadata.insert("chunk_count".to_string(), json!(chunks.len()));
    response_metadata.insert("candidate_count".to_string(), json!(candidate_count));
    response_metadata.insert("finish_reasons".to_string(), json!(finish_reasons));
    response_metadata.insert(
        "usage_metadata".to_string(),
        usage_metadata.unwrap_or(Value::Null),
    );
    response_metadata.insert(
        "prompt_feedback".to_string(),
        prompt_feedback.unwrap_or(Value::Null),
    );

    build_transcript_from_generated_text(&generated_text, context, response_metadata)
}

fn build_transcript_from_generated_text(
    generated_text: &str,
    context: GeminiResponseContext<'_>,
    mut response_metadata: Map<String, Value>,
) -> Transcript {
    let generated_json = parse_generated_json(generated_text);
    let segments = generated_json
        .as_ref()
        .and_then(|value| parse_transcript_segments(value, context.duration_secs))
        .filter(|segments| !segments.is_empty())
        .or_else(|| {
            generated_json
                .as_ref()
                .and_then(|value| value.get("text").and_then(Value::as_str))
                .map(single_text_segment)
        })
        .unwrap_or_else(|| single_text_segment(generated_text));

    response_metadata.insert(
        "generated_json_valid".to_string(),
        Value::Bool(generated_json.is_some()),
    );
    response_metadata.insert(
        "timestamps_clamped".to_string(),
        Value::Bool(
            generated_json
                .as_ref()
                .is_some_and(|value| timestamps_need_clamp(value, context.duration_secs)),
        ),
    );

    Transcript {
        segments,
        provider_metadata: Some(json!({
            "gemini": {
                "model": context.model,
                "api_base_url": context.api_base_url,
                "upload_method": context.upload_method,
                "request": {
                    "file_url_present": context.file_url_present,
                },
                "input": {
                    "mime_type": context.mime_type,
                    "bytes": context.input_bytes,
                    "duration_secs": context.duration_secs,
                },
                "response": Value::Object(response_metadata)
            }
        })),
    }
}

pub fn generated_text(response: &Value) -> Option<String> {
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
mod tests;
