use anyhow::Result;
use serde::Serialize;
use std::io::Write;

#[derive(Serialize)]
pub struct Manifest {
    pub schema_version: &'static str,
    pub input: InputInfo,
    pub config: ProcessingConfig,
    pub capabilities: Capabilities,
    pub quality: QualityInfo,
    pub transcript: TranscriptInfo,
    pub segments: Vec<SegmentInfo>,
    pub stats: Stats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

#[derive(Serialize)]
pub struct InputInfo {
    pub file: String,
    pub duration_secs: f64,
    pub duration_ms: i64,
}

#[derive(Serialize)]
pub struct ProcessingConfig {
    pub provider: String,
    pub model: String,
    pub segmentation_enabled: bool,
    pub silence_threshold_db: f64,
    pub min_silence_duration_secs: f64,
    pub output_format: String,
    pub language: Option<String>,
    pub normalized_audio: bool,
}

#[derive(Clone, Serialize)]
pub struct SegmentInfo {
    pub id: String,
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordInfo>,
}

#[derive(Clone, Serialize)]
pub struct WordInfo {
    pub id: String,
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub punctuation: Option<String>,
}

#[derive(Serialize)]
pub struct Stats {
    pub total_duration_secs: f64,
    pub total_duration_ms: i64,
    pub total_segments: usize,
    pub total_characters: usize,
    pub processing_time_secs: f64,
    pub processing_time_ms: i64,
}

#[derive(Serialize)]
pub struct TranscriptInfo {
    pub text: String,
    pub segments: Vec<SegmentInfo>,
}

#[derive(Serialize)]
pub struct Capabilities {
    pub segments: bool,
    pub word_timestamps: bool,
    pub speaker_labels: bool,
    pub language_per_segment: bool,
    pub emotion_per_segment: bool,
    pub native_timestamps: bool,
}

#[derive(Serialize)]
pub struct QualityInfo {
    pub timing_source: String,
    pub timing_reliable: bool,
    pub timestamps_clamped: bool,
    pub speaker_source: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Serialize)]
pub struct ProviderMetadata {
    pub provider: String,
    pub schema_version: String,
    pub data: serde_json::Value,
}

pub fn write_manifest(manifest: &Manifest, writer: &mut impl Write) -> Result<()> {
    serde_json::to_writer_pretty(writer, manifest)?;
    Ok(())
}
