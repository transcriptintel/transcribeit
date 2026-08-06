use serde_json::Value;

use crate::output::manifest::QualityInfo;
use crate::pipeline::PipelineConfig;
use crate::transcriber::Transcript;

pub(super) fn build_quality(config: &PipelineConfig, transcript: &Transcript) -> QualityInfo {
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
    let all_segments_have_valid_timing = !transcript.segments.is_empty()
        && transcript
            .segments
            .iter()
            .all(|segment| segment.start_ms >= 0 && segment.end_ms > segment.start_ms)
        && !has_non_monotonic_timestamps(transcript);

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
    if transcript
        .segments
        .iter()
        .any(|segment| segment.start_ms < 0)
    {
        warnings.push("One or more segments starts before zero.".to_string());
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
            && all_segments_have_valid_timing,
        timestamps_clamped,
        speaker_source: transcript
            .segments
            .iter()
            .any(|segment| segment.speaker.is_some())
            .then(|| speaker_source(&config.provider_name).to_string()),
        warnings,
    }
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

fn timing_source(provider: &str) -> &'static str {
    match provider {
        "gemini" => "model_generated",
        "qwen-filetrans" | "openai" | "azure" | "nvidia-riva" | "deepgram" => "provider_native",
        "local" => "model_native",
        _ => "unknown",
    }
}

fn speaker_source(provider: &str) -> &'static str {
    match provider {
        "gemini" => "model_generated",
        "openai" => "provider_native",
        "local" => "unknown",
        _ => "provider_native",
    }
}

#[cfg(test)]
mod tests;
