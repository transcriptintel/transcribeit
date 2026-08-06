mod deepgram;
mod gemini;
mod local;
mod openai;
mod qwen;
mod riva;
mod s3;

use anyhow::Result;

use crate::analysis::{AnalysisConfig, TranscriptAnalyzer};
use crate::cli::Provider;
use crate::engines::rate_limit::ApiRequestSettings;
use crate::transcriber::Transcriber;

pub(crate) struct ProviderRuntime {
    pub(crate) engine: Box<dyn Transcriber>,
    pub(crate) analyzer: Option<Box<dyn TranscriptAnalyzer>>,
    pub(crate) provider_name: String,
    pub(crate) model_name: String,
}

pub(crate) struct ProviderFactoryArgs<'a> {
    pub(crate) provider: &'a Provider,
    pub(crate) model: Option<&'a str>,
    pub(crate) base_url: Option<&'a str>,
    pub(crate) explicit_api_key: Option<&'a str>,
    pub(crate) dashscope_api_key: Option<&'a str>,
    pub(crate) gemini_api_key: Option<&'a str>,
    pub(crate) nvidia_api_key: Option<&'a str>,
    pub(crate) nvidia_riva_function_id: Option<&'a str>,
    pub(crate) nvidia_riva_server: Option<&'a str>,
    pub(crate) deepgram_api_key: Option<&'a str>,
    pub(crate) azure_api_key: Option<&'a str>,
    pub(crate) remote_model: Option<&'a str>,
    pub(crate) qwen_api_base_url: &'a str,
    pub(crate) gemini_api_base_url: &'a str,
    pub(crate) deepgram_api_base_url: &'a str,
    pub(crate) deepgram_intelligence: bool,
    pub(crate) deepgram_summarize: bool,
    pub(crate) deepgram_topics: bool,
    pub(crate) deepgram_intents: bool,
    pub(crate) deepgram_detect_entities: bool,
    pub(crate) deepgram_sentiment: bool,
    pub(crate) deepgram_keyterm: &'a [String],
    pub(crate) deepgram_search: &'a [String],
    pub(crate) deepgram_redact: &'a [String],
    pub(crate) deepgram_replace: &'a [String],
    pub(crate) deepgram_filler_words: bool,
    pub(crate) deepgram_numerals: bool,
    pub(crate) deepgram_use_presigned_url: bool,
    pub(crate) gemini_file_cache: bool,
    pub(crate) gemini_use_presigned_url: bool,
    pub(crate) gemini_file_cache_index: Option<&'a std::path::Path>,
    pub(crate) gemini_autoclean: bool,
    pub(crate) gemini_explicit_cache: bool,
    pub(crate) gemini_cache_ttl_secs: u64,
    pub(crate) language: Option<&'a str>,
    pub(crate) azure_deployment: &'a str,
    pub(crate) azure_api_version: &'a str,
    pub(crate) diarize: bool,
    pub(crate) speakers: Option<i32>,
    pub(crate) s3_bucket: Option<&'a str>,
    pub(crate) s3_region: Option<&'a str>,
    pub(crate) s3_endpoint_url: Option<&'a str>,
    pub(crate) s3_access_key_id: Option<&'a str>,
    pub(crate) s3_secret_access_key: Option<&'a str>,
    pub(crate) s3_session_token: Option<&'a str>,
    pub(crate) s3_prefix: Option<&'a str>,
    pub(crate) s3_presign_expires_secs: u64,
    pub(crate) s3_force_path_style: bool,
    pub(crate) cleanup_staged_resources: bool,
    pub(crate) settings: ApiRequestSettings,
    pub(crate) analysis: &'a AnalysisConfig,
}

pub(crate) async fn build(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    match args.provider {
        Provider::Local => local::build_local(args),
        #[cfg(feature = "sherpa-onnx")]
        Provider::SherpaOnnx => local::build_sherpa(args),
        Provider::Openai => openai::build_openai(args),
        Provider::Azure => openai::build_azure(args),
        Provider::QwenFiletrans => qwen::build(args).await,
        Provider::Gemini => gemini::build(args).await,
        Provider::NvidiaRiva => riva::build(args),
        Provider::Deepgram => deepgram::build(args).await,
    }
}

pub(crate) fn handles_diarization(provider_name: &str, model_name: &str) -> bool {
    match provider_name {
        "nvidia-riva" | "gemini" | "deepgram" => true,
        "openai" => model_name.eq_ignore_ascii_case("gpt-4o-transcribe-diarize"),
        _ => false,
    }
}

pub(super) fn owned(value: Option<&str>) -> Option<String> {
    value.map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::handles_diarization;

    #[test]
    fn provider_native_diarization_is_model_sensitive_for_openai() {
        assert!(handles_diarization("openai", "gpt-4o-transcribe-diarize"));
        assert!(!handles_diarization("openai", "gpt-transcribe"));
        assert!(handles_diarization("deepgram", "nova-3"));
        assert!(!handles_diarization("local", "ggml-base"));
    }
}
