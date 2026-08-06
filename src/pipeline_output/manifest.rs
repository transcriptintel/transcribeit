use anyhow::{Context, Result};
use serde_json::Value;

use super::cache::build_cache_info;
use super::capability::build_capabilities;
use super::quality::build_quality;
use crate::analysis::AnalysisResult;
use crate::output::manifest::{
    AnalysisFailure, InputInfo, Manifest, ProcessingConfig, ProviderMetadata, SegmentInfo, Stats,
    TranscriptInfo, WordInfo, write_manifest,
};
use crate::output::set_private_permissions;
use crate::pipeline::PipelineConfig;
use crate::transcriber::Transcript;

pub(super) fn build_manifest(
    config: &PipelineConfig,
    transcript: &Transcript,
    analysis: Option<&AnalysisResult>,
    analysis_error: Option<&str>,
    total_duration: f64,
    should_segment: bool,
    processing_time: f64,
) -> Manifest {
    Manifest {
        schema_version: "transcribeit.manifest.v2",
        input: InputInfo {
            file: config.input.display().to_string(),
            duration_secs: total_duration,
            duration_ms: secs_to_ms(total_duration),
        },
        config: ProcessingConfig {
            provider: config.provider_name.clone(),
            model: config.model_name.clone(),
            segmentation_enabled: should_segment,
            silence_threshold_db: config.silence_threshold,
            min_silence_duration_secs: config.min_silence_duration,
            output_format: format!("{:?}", config.output_format).to_lowercase(),
            language: config.language.clone(),
            normalized_audio: config.normalize_audio,
        },
        capabilities: build_capabilities(config, transcript),
        quality: build_quality(config, transcript),
        transcript: TranscriptInfo {
            text: transcript.text(),
            segments: build_segment_infos(transcript),
        },
        segments: build_segment_infos(transcript),
        stats: Stats {
            total_duration_secs: total_duration,
            total_duration_ms: secs_to_ms(total_duration),
            total_segments: transcript.segments.len(),
            total_characters: transcript.segments.iter().map(|s| s.text.len()).sum(),
            processing_time_secs: processing_time,
            processing_time_ms: secs_to_ms(processing_time),
        },
        cache: build_cache_info(config, transcript, analysis),
        analysis: analysis.cloned(),
        analysis_error: analysis_error.map(|message| AnalysisFailure {
            message: message.to_string(),
        }),
        provider_metadata: build_provider_metadata(
            &config.provider_name,
            transcript.provider_metadata.clone(),
        ),
    }
}

pub(crate) fn write_manifest_output(
    config: &PipelineConfig,
    transcript: &Transcript,
    analysis: Option<&AnalysisResult>,
    analysis_error: Option<&str>,
    total_duration: f64,
    should_segment: bool,
    processing_time: f64,
) -> Result<()> {
    let Some(ref dir) = config.output_dir else {
        return Ok(());
    };

    std::fs::create_dir_all(dir)?;
    let stem = config
        .input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let manifest_path = dir.join(format!("{stem}.manifest.json"));
    let manifest = build_manifest(
        config,
        transcript,
        analysis,
        analysis_error,
        total_duration,
        should_segment,
        processing_time,
    );

    let mut temporary = tempfile::Builder::new()
        .prefix(".transcribeit-manifest-")
        .tempfile_in(dir)
        .with_context(|| format!("Failed to create temporary manifest in {}", dir.display()))?;
    set_private_permissions(temporary.as_file())?;
    write_manifest(&manifest, &mut temporary)?;
    temporary
        .as_file_mut()
        .sync_all()
        .context("Failed to sync temporary manifest")?;
    temporary.persist(&manifest_path).map_err(|error| {
        anyhow::anyhow!(
            "Failed to atomically replace manifest {}: {}",
            manifest_path.display(),
            error.error
        )
    })?;
    eprintln!("Manifest written to {}", manifest_path.display());
    Ok(())
}

fn build_segment_infos(transcript: &Transcript) -> Vec<SegmentInfo> {
    transcript
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| SegmentInfo {
            id: format!("seg_{:06}", index + 1),
            index,
            start_secs: segment.start_ms as f64 / 1000.0,
            end_secs: segment.end_ms as f64 / 1000.0,
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            text: segment.text.trim().to_string(),
            speaker: segment.speaker.clone(),
            language: segment.language.clone(),
            emotion: segment.emotion.clone(),
            words: segment
                .words
                .iter()
                .enumerate()
                .map(|(word_index, word)| WordInfo {
                    id: format!("seg_{:06}_word_{:06}", index + 1, word_index + 1),
                    index: word_index,
                    start_secs: word.start_ms as f64 / 1000.0,
                    end_secs: word.end_ms as f64 / 1000.0,
                    start_ms: word.start_ms,
                    end_ms: word.end_ms,
                    text: word.text.clone(),
                    punctuation: word.punctuation.clone(),
                })
                .collect(),
        })
        .collect()
}

fn build_provider_metadata(provider: &str, metadata: Option<Value>) -> Option<ProviderMetadata> {
    let metadata = metadata?;
    if metadata.get("provider").and_then(Value::as_str).is_some() && metadata.get("data").is_some()
    {
        return Some(ProviderMetadata {
            provider: metadata
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or(provider)
                .to_string(),
            schema_version: metadata
                .get("schema_version")
                .and_then(Value::as_str)
                .unwrap_or("provider.metadata.v1")
                .to_string(),
            data: metadata.get("data").cloned().unwrap_or(Value::Null),
        });
    }

    let data = metadata
        .get(provider)
        .cloned()
        .or_else(|| provider_key(provider).and_then(|key| metadata.get(key).cloned()))
        .unwrap_or(metadata);
    Some(ProviderMetadata {
        provider: provider.to_string(),
        schema_version: format!("{provider}.metadata.v1"),
        data,
    })
}

fn provider_key(provider: &str) -> Option<&'static str> {
    match provider {
        "qwen-filetrans" => Some("qwen"),
        "nvidia-riva" => Some("nvidia"),
        _ => None,
    }
}

fn secs_to_ms(seconds: f64) -> i64 {
    (seconds * 1000.0).round() as i64
}
