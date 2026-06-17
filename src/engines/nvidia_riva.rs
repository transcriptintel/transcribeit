use std::collections::BTreeSet;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use tonic::Request;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Endpoint};

use crate::audio::wav::encode_wav;
use crate::transcriber::{Segment, Transcriber, Transcript, Word};

mod proto {
    pub mod nvidia {
        pub mod riva {
            tonic::include_proto!("nvidia.riva");

            pub mod asr {
                tonic::include_proto!("nvidia.riva.asr");
            }
        }
    }
}

use proto::nvidia::riva::asr::riva_speech_recognition_client::RivaSpeechRecognitionClient;
use proto::nvidia::riva::asr::{
    RecognitionConfig, RecognizeRequest, RecognizeResponse, SpeakerDiarizationConfig,
};
use proto::nvidia::riva::{AudioEncoding, RequestId};

const DEFAULT_SERVER: &str = "grpc.nvcf.nvidia.com:443";
const MAX_GRPC_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

pub struct NvidiaRiva {
    server: String,
    api_key: String,
    function_id: String,
    model: Option<String>,
    language: String,
    request_timeout: std::time::Duration,
    speakers: Option<i32>,
}

impl NvidiaRiva {
    pub fn new(
        server: Option<String>,
        api_key: String,
        function_id: String,
        model: Option<String>,
        language: Option<String>,
        request_timeout: std::time::Duration,
        speakers: Option<i32>,
    ) -> Self {
        Self {
            server: normalize_server_url(server.as_deref().unwrap_or(DEFAULT_SERVER)),
            api_key,
            function_id,
            model,
            language: normalize_language(language.as_deref()),
            request_timeout,
            speakers,
        }
    }

    async fn client(&self) -> Result<RivaSpeechRecognitionClient<Channel>> {
        let endpoint = Endpoint::new(self.server.clone())
            .with_context(|| format!("Invalid NVIDIA Riva server URL: {}", self.server))?
            .timeout(self.request_timeout);
        let channel = endpoint
            .connect()
            .await
            .with_context(|| format!("Failed to connect to NVIDIA Riva at {}", self.server))?;
        Ok(RivaSpeechRecognitionClient::new(channel)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
            .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES))
    }

    async fn recognize(&self, audio_bytes: Vec<u8>, audio: AudioSpec) -> Result<Transcript> {
        let started = Instant::now();
        let mut client = self.client().await?;
        let request_id = uuid::Uuid::new_v4().to_string();
        let request = RecognizeRequest {
            config: Some(self.recognition_config(audio)),
            audio: audio_bytes,
            id: Some(RequestId {
                value: request_id.clone(),
            }),
        };
        let mut request = Request::new(request);
        request
            .metadata_mut()
            .insert("function-id", metadata_value(&self.function_id)?);
        request.metadata_mut().insert(
            "authorization",
            metadata_value(&format!("Bearer {}", self.api_key))?,
        );

        let response = tokio::time::timeout(self.request_timeout, client.recognize(request))
            .await
            .context("NVIDIA Riva request timed out")?
            .context("NVIDIA Riva recognize request failed")?
            .into_inner();

        Ok(parse_response(
            response,
            ResponseContext {
                server: &self.server,
                request_id: &request_id,
                model: self.model.as_deref(),
                language: &self.language,
                audio,
                elapsed: started.elapsed(),
                speakers: self.speakers,
            },
        ))
    }

    fn recognition_config(&self, audio: AudioSpec) -> RecognitionConfig {
        RecognitionConfig {
            encoding: AudioEncoding::LinearPcm as i32,
            sample_rate_hertz: audio.sample_rate_hertz,
            language_code: self.language.clone(),
            max_alternatives: 1,
            profanity_filter: false,
            speech_contexts: Vec::new(),
            audio_channel_count: audio.channels,
            enable_word_time_offsets: true,
            enable_automatic_punctuation: true,
            enable_separate_recognition_per_channel: false,
            model: self.model.clone().unwrap_or_default(),
            verbatim_transcripts: false,
            diarization_config: self
                .speakers
                .map(|max_speaker_count| SpeakerDiarizationConfig {
                    enable_speaker_diarization: true,
                    max_speaker_count,
                }),
            custom_configuration: Default::default(),
            endpointing_config: None,
        }
    }
}

#[async_trait]
impl Transcriber for NvidiaRiva {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let wav_bytes = encode_wav(&audio_samples)?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_path(&self, wav_path: &Path) -> Result<Transcript> {
        let wav_bytes = tokio::fs::read(wav_path)
            .await
            .with_context(|| format!("Failed to read audio file: {}", wav_path.display()))?;
        self.transcribe_wav(wav_bytes).await
    }

    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<Transcript> {
        let audio = AudioSpec::from_wav_bytes(&wav_bytes).unwrap_or_default();
        self.recognize(wav_bytes, audio).await
    }
}

#[derive(Clone, Copy)]
struct AudioSpec {
    sample_rate_hertz: i32,
    channels: i32,
}

impl Default for AudioSpec {
    fn default() -> Self {
        Self {
            sample_rate_hertz: 16_000,
            channels: 1,
        }
    }
}

impl AudioSpec {
    fn from_wav_bytes(bytes: &[u8]) -> Option<Self> {
        let reader = hound::WavReader::new(Cursor::new(bytes)).ok()?;
        let spec = reader.spec();
        Some(Self {
            sample_rate_hertz: spec.sample_rate as i32,
            channels: spec.channels as i32,
        })
    }
}

struct ResponseContext<'a> {
    server: &'a str,
    request_id: &'a str,
    model: Option<&'a str>,
    language: &'a str,
    audio: AudioSpec,
    elapsed: std::time::Duration,
    speakers: Option<i32>,
}

fn parse_response(response: RecognizeResponse, context: ResponseContext<'_>) -> Transcript {
    let provider_request_id = response.id.as_ref().map(|id| id.value.clone());
    let mut segments = Vec::new();
    let mut result_count = 0usize;
    let mut alternative_count = 0usize;
    let mut word_count = 0usize;
    let mut languages = BTreeSet::new();
    let mut confidences = Vec::new();

    for result in response.results {
        result_count += 1;
        let Some(alt) = result.alternatives.into_iter().next() else {
            continue;
        };
        alternative_count += 1;
        if alt.transcript.trim().is_empty() {
            continue;
        }
        if alt.confidence != 0.0 {
            confidences.push(alt.confidence);
        }
        for lang in &alt.language_code {
            if !lang.trim().is_empty() {
                languages.insert(lang.clone());
            }
        }

        let mut words = Vec::new();
        let mut speaker_tags = BTreeSet::new();
        for word in alt.words {
            if word.word.trim().is_empty() {
                continue;
            }
            if !word.language_code.trim().is_empty() {
                languages.insert(word.language_code.clone());
            }
            if word.speaker_tag > 0 {
                speaker_tags.insert(word.speaker_tag);
            }
            words.push(Word {
                start_ms: word.start_time as i64,
                end_ms: word.end_time as i64,
                text: word.word,
                punctuation: None,
            });
        }
        word_count += words.len();

        let start_ms = words.first().map(|word| word.start_ms).unwrap_or(0);
        let end_ms = words
            .last()
            .map(|word| word.end_ms)
            .unwrap_or_else(|| (result.audio_processed as f64 * 1000.0).round() as i64);
        let speaker = if speaker_tags.len() == 1 {
            speaker_tags
                .iter()
                .next()
                .map(|speaker| format!("Speaker {speaker}"))
        } else {
            None
        };
        let language = alt
            .language_code
            .iter()
            .find(|lang| !lang.trim().is_empty())
            .cloned();

        segments.push(Segment {
            start_ms,
            end_ms,
            text: alt.transcript,
            speaker,
            language,
            emotion: None,
            words,
        });
    }

    let mean_confidence = if confidences.is_empty() {
        None
    } else {
        Some(confidences.iter().sum::<f32>() / confidences.len() as f32)
    };

    Transcript {
        segments,
        provider_metadata: Some(json!({
            "provider": "nvidia-riva",
            "schema_version": "nvidia-riva.metadata.v1",
            "data": {
                "server": context.server,
                "request_id": context.request_id,
                "provider_request_id": provider_request_id,
                "model": context.model,
                "language": context.language,
                "detected_languages": languages.into_iter().collect::<Vec<_>>(),
                "audio": {
                    "sample_rate_hertz": context.audio.sample_rate_hertz,
                    "channels": context.audio.channels,
                },
                "features": {
                    "automatic_punctuation": true,
                    "word_time_offsets": true,
                    "speaker_diarization": context.speakers.is_some(),
                    "max_speaker_count": context.speakers,
                },
                "response": {
                    "result_count": result_count,
                    "alternative_count": alternative_count,
                    "word_count": word_count,
                    "mean_confidence": mean_confidence,
                    "elapsed_ms": context.elapsed.as_millis() as u64,
                }
            }
        })),
    }
}

fn normalize_server_url(server: &str) -> String {
    let server = server.trim().trim_end_matches('/');
    if let Some(rest) = server.strip_prefix("grpc://") {
        format!("https://{rest}")
    } else if server.starts_with("http://") || server.starts_with("https://") {
        server.to_string()
    } else {
        format!("https://{server}")
    }
}

fn normalize_language(language: Option<&str>) -> String {
    match language.map(str::trim).filter(|lang| !lang.is_empty()) {
        Some("en") => "en-US".to_string(),
        Some(lang) => lang.to_string(),
        None => "en-US".to_string(),
    }
}

fn metadata_value(value: &str) -> Result<MetadataValue<tonic::metadata::Ascii>> {
    value
        .parse()
        .with_context(|| "Invalid NVIDIA Riva metadata value")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_hosted_server_url() {
        assert_eq!(
            normalize_server_url("grpc.nvcf.nvidia.com:443"),
            "https://grpc.nvcf.nvidia.com:443"
        );
        assert_eq!(
            normalize_server_url("grpc://grpc.nvcf.nvidia.com:443"),
            "https://grpc.nvcf.nvidia.com:443"
        );
        assert_eq!(
            normalize_server_url("http://localhost:50051"),
            "http://localhost:50051"
        );
    }

    #[test]
    fn normalizes_short_english_language_for_riva() {
        assert_eq!(normalize_language(Some("en")), "en-US");
        assert_eq!(normalize_language(Some("de-DE")), "de-DE");
        assert_eq!(normalize_language(None), "en-US");
    }
}
