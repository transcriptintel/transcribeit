use std::path::Path;

use serde_json::{Value, json};

const DEFAULT_PROMPT: &str = r#"Transcribe this audio as accurately and verbatim as possible.

Return only JSON matching the provided schema.
Use domain-specific spelling when clear from the audio.
Create short, readable segments in chronological order.
Use null for timestamps, speaker, language, or emotion when uncertain.
Do not summarize, paraphrase, omit disfluencies that are clearly spoken, or invent content.
"#;

pub(super) fn upload_base_url(api_base_url: &str) -> String {
    if let Some(root) = api_base_url.strip_suffix("/v1beta") {
        format!("{root}/upload/v1beta")
    } else if let Some(root) = api_base_url.strip_suffix("/v1") {
        format!("{root}/upload/v1")
    } else {
        format!("{}/upload/v1beta", api_base_url.trim_end_matches('/'))
    }
}

pub(super) fn audio_mime(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("mp3") => "audio/mp3",
        Some(ext) if ext.eq_ignore_ascii_case("wav") => "audio/wav",
        Some(ext) if ext.eq_ignore_ascii_case("aiff") || ext.eq_ignore_ascii_case("aif") => {
            "audio/aiff"
        }
        Some(ext) if ext.eq_ignore_ascii_case("aac") => "audio/aac",
        Some(ext) if ext.eq_ignore_ascii_case("ogg") || ext.eq_ignore_ascii_case("oga") => {
            "audio/ogg"
        }
        Some(ext) if ext.eq_ignore_ascii_case("flac") => "audio/flac",
        _ => "audio/wav",
    }
}

pub(super) fn prompt_text(language: Option<&str>, duration_secs: Option<f64>) -> String {
    let mut prompt = DEFAULT_PROMPT.to_string();
    if let Some(duration_secs) = duration_secs {
        prompt.push_str(&format!(
            "\nThe audio duration is {duration_secs:.2} seconds. Do not return any timestamp greater than this duration, and do not infer content beyond the end of the audio."
        ));
    }
    match language {
        Some(lang) if !lang.eq_ignore_ascii_case("auto") => {
            prompt.push_str(&format!("\nThe expected spoken language is `{lang}`."));
            prompt
        }
        _ => prompt,
    }
}

pub(super) fn generate_payload(file_uri: &str, mime_type: &str, prompt: &str) -> Value {
    json!({
        "contents": [{
            "parts": [
                {
                    "file_data": {
                        "mime_type": mime_type,
                        "file_uri": file_uri
                    }
                },
                { "text": prompt }
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": transcript_schema()
        }
    })
}

pub(super) fn generate_payload_with_cached_content(cached_content: &str, prompt: &str) -> Value {
    json!({
        "contents": [{
            "role": "user",
            "parts": [
                { "text": prompt }
            ]
        }],
        "cachedContent": cached_content,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": transcript_schema()
        }
    })
}

fn transcript_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "text": {
                "type": "STRING",
                "description": "The complete transcript text."
            },
            "segments": {
                "type": "ARRAY",
                "description": "Chronological transcript segments.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "start_secs": {
                            "type": "NUMBER",
                            "nullable": true,
                            "description": "Segment start time in seconds, or null when uncertain."
                        },
                        "end_secs": {
                            "type": "NUMBER",
                            "nullable": true,
                            "description": "Segment end time in seconds, or null when uncertain."
                        },
                        "speaker": {
                            "type": "STRING",
                            "nullable": true,
                            "description": "Speaker label when confidently distinguishable."
                        },
                        "language": {
                            "type": "STRING",
                            "nullable": true,
                            "description": "BCP-47 language code when confidently detected."
                        },
                        "emotion": {
                            "type": "STRING",
                            "nullable": true,
                            "description": "Dominant speaker emotion when confidently detected."
                        },
                        "text": {
                            "type": "STRING",
                            "description": "Verbatim segment transcript text."
                        }
                    },
                    "required": ["text"],
                    "propertyOrdering": ["start_secs", "end_secs", "speaker", "language", "emotion", "text"]
                }
            }
        },
        "required": ["text", "segments"],
        "propertyOrdering": ["text", "segments"]
    })
}

#[cfg(test)]
mod tests {
    use super::upload_base_url;

    #[test]
    fn derives_upload_base_url_from_api_base_url() {
        assert_eq!(
            upload_base_url("https://generativelanguage.googleapis.com/v1beta"),
            "https://generativelanguage.googleapis.com/upload/v1beta"
        );
        assert_eq!(
            upload_base_url("https://example.com/v1"),
            "https://example.com/upload/v1"
        );
    }
}
