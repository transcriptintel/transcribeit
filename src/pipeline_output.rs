use anyhow::{Context, Result};
use serde_json::Value;

use crate::output::manifest::{
    Capabilities, InputInfo, Manifest, ProcessingConfig, ProviderMetadata, QualityInfo,
    SegmentInfo, Stats, TranscriptInfo, WordInfo, write_manifest,
};
use crate::output::{srt::write_srt, vtt::write_vtt};
use crate::pipeline::{OutputFormat, PipelineConfig};
use crate::transcriber::Transcript;

pub(crate) fn write_outputs(
    config: &PipelineConfig,
    transcript: &Transcript,
    total_duration: f64,
    should_segment: bool,
    processing_time: f64,
) -> Result<()> {
    match config.output_format {
        OutputFormat::Text => write_text(config, transcript)?,
        OutputFormat::Vtt => write_vtt_output(config, transcript)?,
        OutputFormat::Srt => write_srt_output(config, transcript)?,
    }

    if let Some(ref dir) = config.output_dir {
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
            total_duration,
            should_segment,
            processing_time,
        );

        let mut file = std::fs::File::create(&manifest_path)?;
        write_manifest(&manifest, &mut file)?;
        eprintln!("Manifest written to {}", manifest_path.display());
    }

    Ok(())
}

fn write_text(config: &PipelineConfig, transcript: &Transcript) -> Result<()> {
    let text = transcript.text();
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let text_path = dir.join(format!("{stem}.txt"));
        std::fs::write(&text_path, text.as_bytes())
            .with_context(|| format!("Failed to write text output: {}", text_path.display()))?;
        eprintln!("Text written to {}", text_path.display());
    } else {
        println!("{text}");
    }
    Ok(())
}

fn write_vtt_output(config: &PipelineConfig, transcript: &Transcript) -> Result<()> {
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let vtt_path = dir.join(format!("{stem}.vtt"));
        let mut file = std::fs::File::create(&vtt_path)?;
        write_vtt(transcript, &mut file)?;
        eprintln!("VTT written to {}", vtt_path.display());
    } else {
        let mut stdout = std::io::stdout();
        write_vtt(transcript, &mut stdout)?;
    }
    Ok(())
}

fn write_srt_output(config: &PipelineConfig, transcript: &Transcript) -> Result<()> {
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let srt_path = dir.join(format!("{stem}.srt"));
        let mut file = std::fs::File::create(&srt_path)?;
        write_srt(transcript, &mut file)?;
        eprintln!("SRT written to {}", srt_path.display());
    } else {
        let mut stdout = std::io::stdout();
        write_srt(transcript, &mut stdout)?;
    }
    Ok(())
}

fn build_manifest(
    config: &PipelineConfig,
    transcript: &Transcript,
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
        provider_metadata: build_provider_metadata(
            &config.provider_name,
            transcript.provider_metadata.clone(),
        ),
    }
}

fn build_segment_infos(transcript: &Transcript) -> Vec<SegmentInfo> {
    transcript
        .segments
        .iter()
        .enumerate()
        .map(|(i, s)| SegmentInfo {
            id: format!("seg_{:06}", i + 1),
            index: i,
            start_secs: s.start_ms as f64 / 1000.0,
            end_secs: s.end_ms as f64 / 1000.0,
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            text: s.text.trim().to_string(),
            speaker: s.speaker.clone(),
            language: s.language.clone(),
            emotion: s.emotion.clone(),
            words: s
                .words
                .iter()
                .enumerate()
                .map(|(word_index, w)| WordInfo {
                    id: format!("seg_{:06}_word_{:06}", i + 1, word_index + 1),
                    index: word_index,
                    start_secs: w.start_ms as f64 / 1000.0,
                    end_secs: w.end_ms as f64 / 1000.0,
                    start_ms: w.start_ms,
                    end_ms: w.end_ms,
                    text: w.text.clone(),
                    punctuation: w.punctuation.clone(),
                })
                .collect(),
        })
        .collect()
}

fn build_capabilities(config: &PipelineConfig, transcript: &Transcript) -> Capabilities {
    Capabilities {
        segments: !transcript.segments.is_empty(),
        word_timestamps: transcript
            .segments
            .iter()
            .any(|segment| !segment.words.is_empty()),
        speaker_labels: transcript
            .segments
            .iter()
            .any(|segment| segment.speaker.is_some()),
        language_per_segment: transcript
            .segments
            .iter()
            .any(|segment| segment.language.is_some()),
        emotion_per_segment: transcript
            .segments
            .iter()
            .any(|segment| segment.emotion.is_some()),
        native_timestamps: native_timestamps(&config.provider_name),
    }
}

fn build_quality(config: &PipelineConfig, transcript: &Transcript) -> QualityInfo {
    let timing_source = timing_source(&config.provider_name);
    let timestamps_clamped = metadata_bool(
        transcript.provider_metadata.as_ref(),
        &[
            "/data/response/timestamps_clamped",
            "/gemini/response/timestamps_clamped",
            "/response/timestamps_clamped",
        ],
    );
    let mut warnings = Vec::new();
    let has_durations = transcript
        .segments
        .iter()
        .any(|segment| segment.end_ms > segment.start_ms);

    if config.provider_name == "gemini" {
        warnings.push(
            "Gemini timestamps, speakers, language, and emotion are model-generated structured output, not a dedicated ASR schema."
                .to_string(),
        );
    }
    if timestamps_clamped {
        warnings.push(
            "One or more provider timestamps exceeded the source duration and were clamped."
                .to_string(),
        );
    }
    if metadata_bool(
        transcript.provider_metadata.as_ref(),
        &[
            "/data/response/segmented_fallback",
            "/gemini/response/segmented_fallback",
            "/response/segmented_fallback",
        ],
    ) {
        warnings.push(
            "Gemini fell back to segmented transcription; speaker identity may not be stable across segments."
                .to_string(),
        );
    }
    if transcript
        .segments
        .iter()
        .any(|segment| segment.end_ms < segment.start_ms)
    {
        warnings.push("One or more segments has end_ms earlier than start_ms.".to_string());
    }
    if has_non_monotonic_timestamps(transcript) {
        warnings.push(
            "Segment timestamps are not monotonic; subtitle cue order may not match playback time."
                .to_string(),
        );
    }
    let zero_duration_segments = zero_duration_segment_count(transcript);
    if zero_duration_segments > 0 {
        warnings.push(format!(
            "{zero_duration_segments} segment(s) have zero-duration timestamps."
        ));
    }
    if !transcript.segments.is_empty() && !has_durations {
        warnings.push("No positive-duration segment timestamps were returned.".to_string());
    }

    QualityInfo {
        timing_source: timing_source.to_string(),
        timing_reliable: matches!(timing_source, "provider_native" | "model_native")
            && !timestamps_clamped
            && has_durations,
        timestamps_clamped,
        speaker_source: transcript
            .segments
            .iter()
            .any(|segment| segment.speaker.is_some())
            .then(|| speaker_source(&config.provider_name).to_string()),
        warnings,
    }
}

fn has_non_monotonic_timestamps(transcript: &Transcript) -> bool {
    transcript
        .segments
        .windows(2)
        .any(|segments| segments[1].start_ms < segments[0].start_ms)
}

fn zero_duration_segment_count(transcript: &Transcript) -> usize {
    transcript
        .segments
        .iter()
        .filter(|segment| segment.end_ms == segment.start_ms)
        .count()
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

fn metadata_bool(metadata: Option<&Value>, pointers: &[&str]) -> bool {
    metadata.is_some_and(|metadata| {
        pointers.iter().any(|pointer| {
            metadata
                .pointer(pointer)
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
    })
}

fn native_timestamps(provider: &str) -> bool {
    matches!(
        provider,
        "local" | "openai" | "azure" | "qwen-filetrans" | "nvidia-riva"
    )
}

fn timing_source(provider: &str) -> &'static str {
    match provider {
        "gemini" => "model_generated",
        "qwen-filetrans" | "openai" | "azure" | "nvidia-riva" => "provider_native",
        "local" | "sherpa-onnx" => "model_native",
        _ => "unknown",
    }
}

fn speaker_source(provider: &str) -> &'static str {
    match provider {
        "gemini" => "model_generated",
        "openai" => "provider_native",
        "local" | "sherpa-onnx" => "local_diarization",
        _ => "provider_native",
    }
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

#[cfg(test)]
mod tests {
    use super::build_quality;
    use crate::pipeline::{OutputFormat, PipelineConfig};
    use crate::transcriber::{Segment, Transcript};
    use std::path::PathBuf;

    #[test]
    fn quality_warns_on_zero_duration_and_non_monotonic_timestamps() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 1000,
                    end_ms: 2000,
                    text: "first".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 500,
                    end_ms: 500,
                    text: "second".to_string(),
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };
        let config = PipelineConfig {
            input: PathBuf::from("sample.wav"),
            output_dir: None,
            output_format: OutputFormat::Vtt,
            language: None,
            normalize_audio: false,
            segment: false,
            silence_threshold: -40.0,
            min_silence_duration: 0.8,
            max_segment_secs: 600.0,
            segment_concurrency: 1,
            auto_split_for_api: false,
            upload_as_mp3: false,
            vad_model: None,
            diarize: false,
            speakers: None,
            diarize_segmentation_model: None,
            diarize_embedding_model: None,
            provider_name: "gemini".to_string(),
            model_name: "gemini-test".to_string(),
        };

        let quality = build_quality(&config, &transcript);

        assert!(
            quality
                .warnings
                .iter()
                .any(|warning| { warning.contains("Segment timestamps are not monotonic") })
        );
        assert!(
            quality
                .warnings
                .iter()
                .any(|warning| { warning.contains("1 segment(s) have zero-duration timestamps") })
        );
    }
}
