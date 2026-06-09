use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::transcriber::{Segment, Transcript, Word};

pub(super) struct TaskResult {
    pub(super) transcription_url: String,
    pub(super) metadata: serde_json::Value,
}

pub(super) fn normalize_api_base_url(base_url: &str) -> String {
    let mut trimmed = base_url.trim_end_matches('/');
    if let Some(root) = trimmed.strip_suffix("/compatible-mode/v1") {
        trimmed = root;
    } else if let Some(root) = trimmed.strip_suffix("/api/v1") {
        trimmed = root;
    }

    format!("{trimmed}/api/v1")
}

#[derive(Clone, Serialize)]
pub(super) struct SubmitRequest {
    pub(super) model: String,
    pub(super) input: FileInput,
    pub(super) parameters: RequestParameters,
}

#[derive(Clone, Serialize)]
pub(super) struct FileInput {
    pub(super) file_url: String,
}

#[derive(Clone, Serialize)]
pub(super) struct RequestParameters {
    pub(super) channel_id: Vec<u8>,
    pub(super) enable_itn: bool,
    pub(super) enable_words: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) language: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct SubmitResponse {
    pub(super) output: SubmitOutput,
}

#[derive(Deserialize)]
pub(super) struct SubmitOutput {
    pub(super) task_id: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct QueryResponse {
    request_id: Option<String>,
    pub(super) output: QueryOutput,
    usage: Option<serde_json::Value>,
}

impl QueryResponse {
    pub(super) fn into_task_result(self, transcription_url: String) -> TaskResult {
        TaskResult {
            transcription_url,
            metadata: self.output.into_metadata(self.request_id, self.usage),
        }
    }
}

#[derive(Deserialize)]
pub(super) struct QueryOutput {
    pub(super) task_status: Option<String>,
    pub(super) result: Option<QueryResult>,
    task_id: Option<String>,
    submit_time: Option<String>,
    scheduled_time: Option<String>,
    end_time: Option<String>,
}

impl QueryOutput {
    fn into_metadata(
        self,
        request_id: Option<String>,
        usage: Option<serde_json::Value>,
    ) -> serde_json::Value {
        json!({
            "request_id": request_id,
            "task_id": self.task_id,
            "task_status": self.task_status,
            "submit_time": self.submit_time,
            "scheduled_time": self.scheduled_time,
            "end_time": self.end_time,
            "usage": usage,
        })
    }
}

#[derive(Deserialize)]
pub(super) struct QueryResult {
    pub(super) transcription_url: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct ResultDocument {
    file_url: Option<String>,
    audio_info: Option<serde_json::Value>,
    transcripts: Vec<ResultTranscript>,
}

impl ResultDocument {
    pub(super) fn into_transcript(
        self,
        model: &str,
        api_base_url: &str,
        task_metadata: serde_json::Value,
    ) -> Transcript {
        let mut segments = Vec::new();
        for transcript in &self.transcripts {
            for sentence in &transcript.sentences {
                segments.push(Segment {
                    start_ms: sentence.begin_time,
                    end_ms: sentence.end_time,
                    text: sentence.text.clone(),
                    speaker: None,
                    language: sentence.language.clone(),
                    emotion: sentence.emotion.clone(),
                    words: sentence
                        .words
                        .iter()
                        .map(|word| Word {
                            start_ms: word.begin_time,
                            end_ms: word.end_time,
                            text: word.text.clone(),
                            punctuation: word.punctuation.clone(),
                        })
                        .collect(),
                });
            }
        }

        if segments.is_empty() {
            segments = self
                .transcripts
                .iter()
                .map(|transcript| Segment {
                    start_ms: 0,
                    end_ms: 0,
                    text: transcript.text.clone(),
                    speaker: None,
                    ..Default::default()
                })
                .collect();
        }

        let sentence_count = self
            .transcripts
            .iter()
            .map(|transcript| transcript.sentences.len())
            .sum::<usize>();
        let word_count = self
            .transcripts
            .iter()
            .flat_map(|transcript| transcript.sentences.iter())
            .map(|sentence| sentence.words.len())
            .sum::<usize>();

        Transcript {
            segments,
            provider_metadata: Some(json!({
                "qwen": {
                    "model": model,
                    "api_base_url": api_base_url,
                    "task": task_metadata,
                    "result": {
                        "file_url_present": self.file_url.is_some(),
                        "audio_info": self.audio_info,
                        "transcript_count": self.transcripts.len(),
                        "sentence_count": sentence_count,
                        "word_count": word_count,
                        "words_enabled": word_count > 0,
                    }
                }
            })),
        }
    }
}

#[derive(Deserialize)]
struct ResultTranscript {
    text: String,
    #[serde(default)]
    sentences: Vec<ResultSentence>,
}

#[derive(Deserialize)]
struct ResultSentence {
    begin_time: i64,
    end_time: i64,
    text: String,
    language: Option<String>,
    emotion: Option<String>,
    #[serde(default)]
    words: Vec<ResultWord>,
}

#[derive(Deserialize)]
struct ResultWord {
    begin_time: i64,
    end_time: i64,
    text: String,
    punctuation: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::normalize_api_base_url;

    #[test]
    fn normalizes_compatible_base_url_to_api_base_url() {
        assert_eq!(
            normalize_api_base_url("https://example.com/compatible-mode/v1"),
            "https://example.com/api/v1"
        );
    }

    #[test]
    fn leaves_api_base_url_unchanged() {
        assert_eq!(
            normalize_api_base_url("https://example.com/api/v1"),
            "https://example.com/api/v1"
        );
    }
}
