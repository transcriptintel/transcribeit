use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use crate::transcriber::{Segment, Transcript};

const MAX_SEGMENTS: usize = 200_000;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BridgePayload {
    locale: String,
    asset_install_requested: bool,
    segments: Vec<BridgeSegment>,
}

#[derive(Deserialize)]
struct BridgeSegment {
    start: f64,
    end: f64,
    text: String,
}

pub(super) fn parse(payload: &str) -> Result<Transcript> {
    let payload: BridgePayload =
        serde_json::from_str(payload).context("Apple Speech returned malformed JSON")?;
    anyhow::ensure!(
        payload.segments.len() <= MAX_SEGMENTS,
        "Apple Speech returned too many segments"
    );

    let mut segments = Vec::with_capacity(payload.segments.len());
    let mut previous_start = None;
    for (index, segment) in payload.segments.into_iter().enumerate() {
        let start_ms = timestamp_ms(segment.start, index)?;
        let end_ms = timestamp_ms(segment.end, index)?;
        anyhow::ensure!(
            end_ms >= start_ms,
            "Apple Speech segment {} has reversed timing",
            index + 1
        );
        if let Some(previous_start) = previous_start {
            anyhow::ensure!(
                start_ms >= previous_start,
                "Apple Speech segment {} has non-monotonic timing",
                index + 1
            );
        }
        previous_start = Some(start_ms);
        if segment.text.trim().is_empty() {
            continue;
        }
        segments.push(Segment {
            start_ms,
            end_ms,
            text: segment.text,
            speaker: None,
            language: Some(payload.locale.clone()),
            emotion: None,
            words: Vec::new(),
        });
    }

    Ok(Transcript {
        segments,
        provider_metadata: Some(json!({
            "response": {
                "locale": payload.locale,
                "on_device": true,
                "apple_intelligence_available": true,
                "asset_install_requested": payload.asset_install_requested,
                "asset_managed_by": "macos"
            }
        })),
    })
}

fn timestamp_ms(seconds: f64, index: usize) -> Result<i64> {
    anyhow::ensure!(
        seconds.is_finite() && seconds >= 0.0 && seconds <= i64::MAX as f64 / 1000.0,
        "Apple Speech segment {} has an invalid timestamp",
        index + 1
    );
    Ok((seconds * 1000.0).round() as i64)
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parses_timed_segments_and_sanitized_metadata() {
        let transcript = parse(
            r#"{"locale":"en-US","assetInstallRequested":false,"segments":[{"start":1.25,"end":2.5,"text":"Hello"}]}"#,
        )
        .unwrap();

        assert_eq!(transcript.segments[0].start_ms, 1_250);
        assert_eq!(transcript.segments[0].end_ms, 2_500);
        assert_eq!(transcript.segments[0].language.as_deref(), Some("en-US"));
        assert_eq!(
            transcript.provider_metadata.as_ref().unwrap()["response"]["on_device"],
            true
        );
    }

    #[test]
    fn rejects_invalid_timing_without_echoing_text() {
        let error = parse(
            r#"{"locale":"en-US","assetInstallRequested":false,"segments":[{"start":2.0,"end":1.0,"text":"private words"}]}"#,
        )
        .err()
        .expect("reversed timing must fail");

        assert!(error.to_string().contains("reversed timing"));
        assert!(!error.to_string().contains("private words"));
    }
}
