use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::analysis::{AnalysisConfig, AnalysisResult, SummaryAnalysis, TranscriptAnalyzer};
use crate::transcriber::Transcript;

use super::GeminiApi;
use super::response::generated_text;

const SUMMARY_PROMPT: &str = r#"Summarize this interview transcript for downstream research review.

Return only JSON matching the provided schema.
Do not invent facts that are not present in the transcript.
Keep the short summary concise.
Use key_points for the most important findings or opinions.
Use topics for major discussion areas.
Use action_items for explicit next steps only; otherwise return an empty array.
Use questions for notable interviewer questions.
Use follow_ups for issues that may need human review or additional analysis.
"#;

#[async_trait]
impl TranscriptAnalyzer for GeminiApi {
    async fn analyze_transcript(
        &self,
        transcript: &Transcript,
        config: &AnalysisConfig,
    ) -> Result<AnalysisResult> {
        let result = if config.summary {
            Some(self.summarize_transcript(transcript).await?)
        } else {
            None
        };

        Ok(AnalysisResult {
            provider: "gemini".to_string(),
            model: self.model.clone(),
            schema_version: "transcribeit.analysis.v1".to_string(),
            summary: result.as_ref().map(|result| result.summary.clone()),
            provider_metadata: result.and_then(|result| result.provider_metadata),
        })
    }
}

impl GeminiApi {
    async fn summarize_transcript(&self, transcript: &Transcript) -> Result<SummaryResult> {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.api_base_url,
            urlencoding::encode(&self.model)
        );
        let payload = summary_payload(&transcript.text(), transcript.segments.len());
        let chunks = self.stream_generate_chunks(&url, &payload).await?;
        Ok(parse_summary_chunks(&chunks))
    }
}

struct SummaryResult {
    summary: SummaryAnalysis,
    provider_metadata: Option<Value>,
}

fn summary_payload(transcript_text: &str, segment_count: usize) -> Value {
    json!({
        "contents": [{
            "parts": [{
                "text": format!(
                    "{SUMMARY_PROMPT}\nTranscript segment count: {segment_count}\n\nTranscript:\n{transcript_text}"
                )
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": summary_schema()
        }
    })
}

fn summary_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "short": {
                "type": "STRING",
                "description": "A concise one-paragraph summary."
            },
            "detailed": {
                "type": "STRING",
                "description": "A more detailed summary of the transcript."
            },
            "key_points": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
            },
            "topics": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
            },
            "action_items": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
            },
            "questions": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
            },
            "follow_ups": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
            }
        },
        "required": ["short", "detailed", "key_points", "topics", "action_items", "questions", "follow_ups"],
        "propertyOrdering": ["short", "detailed", "key_points", "topics", "action_items", "questions", "follow_ups"]
    })
}

fn parse_summary_chunks(chunks: &[Value]) -> SummaryResult {
    let generated = chunks
        .iter()
        .filter_map(generated_text)
        .collect::<Vec<_>>()
        .join("");
    let parsed = serde_json::from_str::<SummaryResponse>(strip_json_fence(generated.trim())).ok();
    let generated_json_valid = parsed.is_some();
    let summary = parsed
        .map(Into::into)
        .unwrap_or_else(|| fallback_summary(&generated));
    let metadata = json!({
        "response": {
            "streaming": true,
            "chunk_count": chunks.len(),
            "generated_json_valid": generated_json_valid,
            "finish_reasons": finish_reasons(chunks),
            "usage_metadata": chunks
                .iter()
                .rev()
                .find_map(|chunk| chunk.get("usageMetadata").cloned())
                .unwrap_or(Value::Null)
        }
    });

    SummaryResult {
        summary,
        provider_metadata: Some(metadata),
    }
}

#[derive(Deserialize)]
struct SummaryResponse {
    short: String,
    detailed: String,
    #[serde(default)]
    key_points: Vec<String>,
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default)]
    action_items: Vec<String>,
    #[serde(default)]
    questions: Vec<String>,
    #[serde(default)]
    follow_ups: Vec<String>,
}

impl From<SummaryResponse> for SummaryAnalysis {
    fn from(value: SummaryResponse) -> Self {
        Self {
            short: value.short,
            detailed: value.detailed,
            key_points: value.key_points,
            topics: value.topics,
            action_items: value.action_items,
            questions: value.questions,
            follow_ups: value.follow_ups,
        }
    }
}

fn fallback_summary(text: &str) -> SummaryAnalysis {
    let text = text.trim().to_string();
    SummaryAnalysis {
        short: text.clone(),
        detailed: text,
        key_points: Vec::new(),
        topics: Vec::new(),
        action_items: Vec::new(),
        questions: Vec::new(),
        follow_ups: vec!["Summary response was not valid JSON.".to_string()],
    }
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

fn finish_reasons(chunks: &[Value]) -> Vec<Value> {
    chunks
        .iter()
        .flat_map(|chunk| {
            chunk
                .get("candidates")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|candidate| candidate.get("finishReason").cloned())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse_summary_chunks;
    use serde_json::json;

    #[test]
    fn parses_streamed_summary_json() {
        let chunks = vec![
            json!({
                "candidates": [{
                    "content": {"parts": [{"text": "{\"short\":\"Short\",\"detailed\":\"Detailed\","}]}
                }]
            }),
            json!({
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "\"key_points\":[\"A\"],\"topics\":[\"T\"],\"action_items\":[],\"questions\":[\"Q\"],\"follow_ups\":[]}"}]}
                }],
                "usageMetadata": {"totalTokenCount": 20}
            }),
        ];

        let result = parse_summary_chunks(&chunks);

        assert_eq!(result.summary.short, "Short");
        assert_eq!(result.summary.key_points, vec!["A"]);
        assert_eq!(
            result
                .provider_metadata
                .as_ref()
                .and_then(|value| value.pointer("/response/chunk_count"))
                .and_then(serde_json::Value::as_u64),
            Some(2)
        );
    }
}
