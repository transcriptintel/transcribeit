#[cfg(test)]
mod tests;

use std::path::Path;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

use crate::audio::wav::encode_wav;
use crate::engines::rate_limit::{self, send_with_retry};
use crate::storage::s3::{S3CleanupResult, S3Uploader};
use crate::transcriber::{Segment, Transcriber, Transcript, Word};

pub struct DeepgramApi {
    base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
    settings: rate_limit::ApiRequestSettings,
    client: Client,
    options: DeepgramOptions,
    presigned_url_uploader: Option<S3Uploader>,
    autoclean: bool,
}

pub struct DeepgramConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub language: Option<String>,
    pub settings: rate_limit::ApiRequestSettings,
    pub options: DeepgramOptions,
    pub presigned_url_uploader: Option<S3Uploader>,
    pub autoclean: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DeepgramOptions {
    pub diarize: bool,
    pub intelligence: bool,
    pub summarize: bool,
    pub topics: bool,
    pub intents: bool,
    pub detect_entities: bool,
    pub sentiment: bool,
    pub keyterms: Vec<String>,
    pub search: Vec<String>,
    pub redact: Vec<String>,
    pub replace: Vec<String>,
    pub filler_words: bool,
    pub numerals: bool,
}

impl DeepgramApi {
    pub fn new(config: DeepgramConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.settings.request_timeout)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key: config.api_key,
            model: config.model,
            language: config.language,
            settings: config.settings,
            client,
            options: config.options,
            presigned_url_uploader: config.presigned_url_uploader,
            autoclean: config.autoclean,
        })
    }

    fn listen_url(&self) -> String {
        let mut params = vec![
            format!("model={}", urlencoding::encode(&self.model)),
            "smart_format=true".to_string(),
            "utterances=true".to_string(),
        ];

        if self.options.diarize {
            params.push("diarize_model=latest".to_string());
        }
        if self.options.intelligence || self.options.summarize {
            params.push("summarize=v2".to_string());
        }
        if self.options.intelligence || self.options.topics {
            params.push("topics=true".to_string());
        }
        if self.options.intelligence || self.options.intents {
            params.push("intents=true".to_string());
        }
        if self.options.intelligence || self.options.detect_entities {
            params.push("detect_entities=true".to_string());
        }
        if self.options.intelligence || self.options.sentiment {
            params.push("sentiment=true".to_string());
        }
        if self.options.filler_words {
            params.push("filler_words=true".to_string());
        }
        if self.options.numerals {
            params.push("numerals=true".to_string());
        }
        append_list_params(&mut params, "keyterm", &self.options.keyterms);
        append_list_params(&mut params, "search", &self.options.search);
        append_list_params(&mut params, "redact", &self.options.redact);
        append_list_params(&mut params, "replace", &self.options.replace);
        if let Some(language) = self
            .language
            .as_deref()
            .filter(|language| !language.is_empty())
        {
            params.push(format!("language={}", urlencoding::encode(language)));
        }

        format!("{}/listen?{}", self.base_url, params.join("&"))
    }

    fn audio_mime(path: &Path) -> &'static str {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("mp3") => "audio/mpeg",
            Some(ext) if ext.eq_ignore_ascii_case("m4a") => "audio/mp4",
            Some(ext) if ext.eq_ignore_ascii_case("mp4") => "audio/mp4",
            Some(ext) if ext.eq_ignore_ascii_case("webm") => "audio/webm",
            Some(ext) if ext.eq_ignore_ascii_case("ogg") => "audio/ogg",
            Some(ext) if ext.eq_ignore_ascii_case("flac") => "audio/flac",
            Some(ext) if ext.eq_ignore_ascii_case("wav") => "audio/wav",
            _ => "audio/wav",
        }
    }

    async fn transcribe_bytes(&self, bytes: Vec<u8>, mime: &'static str) -> Result<Transcript> {
        let url = self.listen_url();
        let body = send_with_retry(&self.settings, "Deepgram listen", || {
            let client = self.client.clone();
            let api_key = self.api_key.clone();
            let url = url.clone();
            let bytes = bytes.clone();
            Box::pin(async move {
                client
                    .post(url)
                    .header("Authorization", format!("Token {api_key}"))
                    .header("Content-Type", mime)
                    .body(bytes)
                    .send()
                    .await
                    .context("Failed to send request to Deepgram listen")
            })
        })
        .await
        .map_err(|(status, body)| anyhow::anyhow!("Deepgram listen returned {status}: {body}"))?;

        parse_response(&body, &self.model, &self.base_url, "direct_upload")
    }

    async fn transcribe_file_url(&self, file_url: String) -> Result<Transcript> {
        let url = self.listen_url();
        let body = send_with_retry(&self.settings, "Deepgram listen", || {
            let client = self.client.clone();
            let api_key = self.api_key.clone();
            let url = url.clone();
            let file_url = file_url.clone();
            Box::pin(async move {
                client
                    .post(url)
                    .header("Authorization", format!("Token {api_key}"))
                    .json(&json!({ "url": file_url }))
                    .send()
                    .await
                    .context("Failed to send request to Deepgram listen")
            })
        })
        .await
        .map_err(|(status, body)| anyhow::anyhow!("Deepgram listen returned {status}: {body}"))?;

        parse_response(&body, &self.model, &self.base_url, "presigned_url")
    }
}

fn append_list_params(params: &mut Vec<String>, name: &str, values: &[String]) {
    params.extend(
        values
            .iter()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(|value| format!("{name}={}", urlencoding::encode(value))),
    );
}

#[async_trait]
impl Transcriber for DeepgramApi {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_path(&self, audio_path: &Path) -> Result<Transcript> {
        if let Some(uploader) = &self.presigned_url_uploader {
            let upload = uploader.upload_and_presign_object(audio_path).await?;
            let mut transcript = self.transcribe_file_url(upload.url.clone()).await?;
            let cleanup = if self.autoclean {
                uploader.cleanup_uploaded_object(&upload).await
            } else {
                S3CleanupResult::skipped(&upload)
            };
            if let Some(error) = cleanup.error.as_deref() {
                eprintln!(
                    "Failed to delete staged Deepgram object s3://{}/{}: {error}",
                    cleanup.bucket, cleanup.key
                );
            }
            add_deepgram_staging_metadata(&mut transcript, cleanup);
            return Ok(transcript);
        }

        let bytes = tokio::fs::read(audio_path)
            .await
            .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
        self.transcribe_bytes(bytes, Self::audio_mime(audio_path))
            .await
    }

    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript> {
        if self.presigned_url_uploader.is_some() {
            let tmp = tempfile::Builder::new()
                .prefix("transcribeit-deepgram-")
                .suffix(".wav")
                .tempfile()
                .context("Failed to create temporary WAV file")?;
            tokio::fs::write(tmp.path(), wav_bytes)
                .await
                .context("Failed to write temporary WAV file")?;
            return self.transcribe_path(tmp.path()).await;
        }

        self.transcribe_bytes(wav_bytes, "audio/wav").await
    }
}

fn add_deepgram_staging_metadata(transcript: &mut Transcript, cleanup: S3CleanupResult) {
    let metadata = transcript.provider_metadata.get_or_insert_with(|| {
        serde_json::json!({
            "provider": "deepgram",
            "schema_version": "deepgram.metadata.v1",
            "data": {}
        })
    });
    if let Some(data) = metadata.get_mut("data").and_then(Value::as_object_mut) {
        data.insert(
            "staging".to_string(),
            serde_json::json!({
                "provider": "s3",
                "cleanup": cleanup.to_metadata(),
            }),
        );
    }
}

fn parse_response(
    body: &[u8],
    model: &str,
    base_url: &str,
    audio_source: &str,
) -> Result<Transcript> {
    let response: Value = serde_json::from_slice(body).context("Failed to parse Deepgram JSON")?;
    let mut segments = parse_utterances(&response);
    if segments.is_empty() {
        segments = parse_alternatives(&response);
    }
    let timestamps_clamped = response
        .pointer("/metadata/duration")
        .and_then(|value| timestamp_ms(Some(value)))
        .is_some_and(|duration_ms| clamp_segments_to_duration(&mut segments, duration_ms));

    Ok(Transcript {
        segments,
        provider_metadata: Some(json!({
            "provider": "deepgram",
            "schema_version": "deepgram.metadata.v1",
            "data": {
                "model": model,
                "base_url": base_url,
                "request": {
                    "audio_source": audio_source,
                    "file_url_present": audio_source == "presigned_url",
                },
                "metadata": response.get("metadata").cloned().unwrap_or(Value::Null),
                "intelligence": intelligence_payload(&response),
                "response": {
                    "channel_count": response.pointer("/results/channels")
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len),
                    "timestamps_clamped": timestamps_clamped,
                    "utterance_count": response.pointer("/results/utterances")
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len),
                    "alternative_count": alternative_count(&response),
                    "mean_confidence": mean_confidence(&response),
                }
            }
        })),
    })
}

fn intelligence_payload(response: &Value) -> Value {
    json!({
        "summary": response.pointer("/results/summary").cloned().unwrap_or(Value::Null),
        "topics": response.pointer("/results/topics").cloned().unwrap_or(Value::Null),
        "intents": response.pointer("/results/intents").cloned().unwrap_or(Value::Null),
        "sentiments": response.pointer("/results/sentiments").cloned().unwrap_or(Value::Null),
        "entities": collect_alternative_field(response, "entities"),
        "summaries": collect_alternative_field(response, "summaries"),
        "search": collect_channel_field(response, "search"),
        "warnings": response.get("warnings").cloned().unwrap_or(Value::Null),
    })
}

fn collect_channel_field(response: &Value, field: &str) -> Value {
    let values = response
        .pointer("/results/channels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|channel| channel.get(field).cloned())
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    json!(values)
}

fn collect_alternative_field(response: &Value, field: &str) -> Value {
    let values = response
        .pointer("/results/channels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|channel| {
            channel
                .get("alternatives")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|alternative| alternative.get(field).cloned())
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    json!(values)
}

fn clamp_segments_to_duration(segments: &mut [Segment], duration_ms: i64) -> bool {
    let mut clamped = false;
    for segment in segments {
        if segment.start_ms > duration_ms {
            segment.start_ms = duration_ms;
            clamped = true;
        }
        if segment.end_ms > duration_ms {
            segment.end_ms = duration_ms;
            clamped = true;
        }
        for word in &mut segment.words {
            if word.start_ms > duration_ms {
                word.start_ms = duration_ms;
                clamped = true;
            }
            if word.end_ms > duration_ms {
                word.end_ms = duration_ms;
                clamped = true;
            }
        }
    }
    clamped
}

fn parse_utterances(response: &Value) -> Vec<Segment> {
    response
        .pointer("/results/utterances")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|utterance| {
            let text = utterance
                .get("transcript")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())?;
            let words = parse_words(utterance.get("words"));
            let speaker = speaker_label(utterance.get("speaker")).or_else(|| {
                words
                    .iter()
                    .find_map(|_| first_word_speaker(utterance.get("words")))
            });

            Some(Segment {
                start_ms: timestamp_ms(utterance.get("start"))
                    .unwrap_or_else(|| words.first().map_or(0, |word| word.start_ms)),
                end_ms: timestamp_ms(utterance.get("end"))
                    .unwrap_or_else(|| words.last().map_or(0, |word| word.end_ms)),
                text: text.to_string(),
                speaker,
                words,
                ..Default::default()
            })
        })
        .collect()
}

fn parse_alternatives(response: &Value) -> Vec<Segment> {
    response
        .pointer("/results/channels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|channel| {
            channel
                .get("alternatives")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .flat_map(parse_alternative)
        .collect()
}

fn parse_alternative(alternative: &Value) -> Vec<Segment> {
    let words = parse_words(alternative.get("words"));
    let sentence_segments = alternative
        .pointer("/paragraphs/paragraphs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|paragraph| {
            paragraph
                .get("sentences")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|sentence| sentence_segment(sentence, &words))
        .collect::<Vec<_>>();

    if !sentence_segments.is_empty() {
        return sentence_segments;
    }

    alternative
        .get("transcript")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| {
            vec![Segment {
                start_ms: words.first().map_or(0, |word| word.start_ms),
                end_ms: words.last().map_or(0, |word| word.end_ms),
                text: text.to_string(),
                speaker: first_word_speaker(alternative.get("words")),
                words,
                ..Default::default()
            }]
        })
        .unwrap_or_default()
}

fn sentence_segment(sentence: &Value, all_words: &[Word]) -> Option<Segment> {
    let text = sentence
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())?;
    let start_ms = timestamp_ms(sentence.get("start")).unwrap_or(0);
    let end_ms = timestamp_ms(sentence.get("end")).unwrap_or(start_ms);
    let words = all_words
        .iter()
        .filter(|word| word.start_ms >= start_ms && word.end_ms <= end_ms)
        .cloned()
        .collect::<Vec<_>>();

    Some(Segment {
        start_ms,
        end_ms,
        text: text.to_string(),
        speaker: None,
        words,
        ..Default::default()
    })
}

fn parse_words(value: Option<&Value>) -> Vec<Word> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|word| {
            let raw = word.get("word").and_then(Value::as_str)?;
            let punctuated = word.get("punctuated_word").and_then(Value::as_str);
            Some(Word {
                start_ms: timestamp_ms(word.get("start")).unwrap_or(0),
                end_ms: timestamp_ms(word.get("end")).unwrap_or(0),
                text: punctuated.unwrap_or(raw).to_string(),
                punctuation: punctuated
                    .filter(|punctuated| *punctuated != raw)
                    .map(ToOwned::to_owned),
            })
        })
        .collect()
}

fn timestamp_ms(value: Option<&Value>) -> Option<i64> {
    let seconds = match value? {
        Value::Number(number) => number.as_f64()?,
        Value::String(text) => text.parse().ok()?,
        _ => return None,
    };
    Some((seconds * 1000.0).round() as i64)
}

fn speaker_label(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Number(number) => number.as_i64().map(|speaker| format!("Speaker {speaker}")),
        Value::String(text) if !text.trim().is_empty() => Some(format!("Speaker {}", text.trim())),
        _ => None,
    }
}

fn first_word_speaker(words: Option<&Value>) -> Option<String> {
    words
        .and_then(Value::as_array)?
        .iter()
        .find_map(|word| speaker_label(word.get("speaker")))
}

fn alternative_count(response: &Value) -> usize {
    response
        .pointer("/results/channels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|channel| {
            channel
                .get("alternatives")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
        })
        .sum()
}

fn mean_confidence(response: &Value) -> Value {
    let confidences = response
        .pointer("/results/channels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|channel| {
            channel
                .get("alternatives")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|alternative| alternative.get("confidence").and_then(Value::as_f64))
        .collect::<Vec<_>>();

    if confidences.is_empty() {
        Value::Null
    } else {
        json!(confidences.iter().sum::<f64>() / confidences.len() as f64)
    }
}
