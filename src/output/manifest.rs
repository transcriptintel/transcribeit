use anyhow::Result;
use serde::Serialize;
use std::io::Write;

#[derive(Serialize)]
pub struct Manifest {
    pub input: InputInfo,
    pub config: ProcessingConfig,
    pub segments: Vec<SegmentInfo>,
    pub stats: Stats,
}

#[derive(Serialize)]
pub struct InputInfo {
    pub file: String,
    pub duration_secs: f64,
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

#[derive(Serialize)]
pub struct SegmentInfo {
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
}

#[derive(Serialize)]
pub struct Stats {
    pub total_duration_secs: f64,
    pub total_segments: usize,
    pub total_characters: usize,
    pub processing_time_secs: f64,
}

pub fn write_manifest(manifest: &Manifest, writer: &mut impl Write) -> Result<()> {
    serde_json::to_writer_pretty(writer, manifest)?;
    Ok(())
}
