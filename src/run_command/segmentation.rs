const SHERPA_DEFAULT_MAX_SEGMENT_SECS: f64 = 30.0;
const QWEN3_ASR_MAX_SEGMENT_SECS: f64 = 20.0;

pub(super) fn model_safe_max_segment_secs(
    provider_name: &str,
    model_name: &str,
    requested: f64,
) -> f64 {
    if provider_name != "sherpa-onnx" {
        return requested;
    }

    let model_name = model_name.to_ascii_lowercase();
    let limit = if model_name.contains("qwen3-asr") {
        QWEN3_ASR_MAX_SEGMENT_SECS
    } else {
        SHERPA_DEFAULT_MAX_SEGMENT_SECS
    };
    requested.min(limit)
}

#[cfg(test)]
mod tests {
    use super::model_safe_max_segment_secs;

    #[test]
    fn caps_qwen3_asr_chunks_at_evaluated_limit() {
        assert_eq!(
            model_safe_max_segment_secs(
                "sherpa-onnx",
                "/models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
                30.0,
            ),
            20.0
        );
        assert_eq!(
            model_safe_max_segment_secs("sherpa-onnx", "qwen3-asr", 15.0),
            15.0
        );
    }

    #[test]
    fn retains_existing_sherpa_and_hosted_provider_limits() {
        assert_eq!(
            model_safe_max_segment_secs("sherpa-onnx", "whisper-small", 45.0),
            30.0
        );
        assert_eq!(
            model_safe_max_segment_secs("openai", "gpt-4o-transcribe", 45.0),
            45.0
        );
    }
}
