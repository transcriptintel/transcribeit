use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

mod setup;
pub(crate) use setup::SetupComponent;

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum ModelSize {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    #[value(name = "large-v3")]
    LargeV3,
    #[value(name = "large-v3-turbo")]
    LargeV3Turbo,
}

impl ModelSize {
    pub(crate) fn file_name(&self) -> &str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::TinyEn => "ggml-tiny.en.bin",
            Self::Base => "ggml-base.bin",
            Self::BaseEn => "ggml-base.en.bin",
            Self::Small => "ggml-small.bin",
            Self::SmallEn => "ggml-small.en.bin",
            Self::Medium => "ggml-medium.bin",
            Self::MediumEn => "ggml-medium.en.bin",
            Self::LargeV3 => "ggml-large-v3.bin",
            Self::LargeV3Turbo => "ggml-large-v3-turbo.bin",
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum Provider {
    /// Local whisper.cpp engine
    Local,
    /// OpenAI-compatible API
    Openai,
    /// Azure OpenAI API
    Azure,
    /// Qwen3-ASR-Flash-Filetrans via DashScope and S3 pre-signed URLs
    #[value(name = "qwen-filetrans")]
    QwenFiletrans,
    /// Gemini multimodal transcription through Files API and streamed generateContent
    Gemini,
    /// NVIDIA-hosted Riva ASR over gRPC
    #[value(name = "nvidia-riva")]
    NvidiaRiva,
    /// Deepgram batch transcription API
    Deepgram,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum OutputFormatArg {
    /// Plain text to stdout
    Text,
    /// WebVTT subtitle format
    Vtt,
    /// SRT subtitle format
    Srt,
}

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub(crate) enum AnalysisKind {
    /// Generate a structured summary from the transcript
    Summary,
}

#[derive(Parser)]
#[command(name = "transcribeit", version, about = "Transcribe audio files")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Command {
    /// Download and install all components for full functionality
    Setup {
        /// Install only a specific component
        #[arg(short, long)]
        component: Option<SetupComponent>,

        /// Directory for downloaded components (overrides MODEL_CACHE_DIR for models)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Hugging Face token for model downloads
        #[arg(short = 't', long, env = "HF_TOKEN", hide_env_values = true)]
        hf_token: Option<String>,
    },

    /// Download a Whisper model
    DownloadModel {
        /// Model size to download
        #[arg(short = 's', long, default_value = "base")]
        model_size: ModelSize,

        /// Directory to save the model (overrides MODEL_CACHE_DIR)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Hugging Face token (optional, or set HF_TOKEN env var)
        #[arg(short = 't', long, env = "HF_TOKEN", hide_env_values = true)]
        hf_token: Option<String>,
    },

    /// List downloaded models
    ListModels {
        /// Models directory (overrides MODEL_CACHE_DIR)
        #[arg(short, long)]
        dir: Option<PathBuf>,
    },

    /// Transcribe an audio or video file
    Run {
        /// Transcription provider
        #[arg(short, long, default_value = "local")]
        provider: Provider,

        /// Input path, directory, or glob pattern
        #[arg(short, long)]
        input: String,

        /// Path to local model file or model alias (required for --provider local)
        #[arg(short, long)]
        model: Option<String>,

        /// API base URL (for --provider openai, or set AZURE_OPENAI_ENDPOINT for azure)
        #[arg(short, long)]
        base_url: Option<String>,

        /// Explicit API key override (prefer provider env vars or --api-key-file)
        #[arg(short, long)]
        api_key: Option<String>,

        /// Read an explicit API key override from a private file
        #[arg(long, conflicts_with = "api_key")]
        api_key_file: Option<PathBuf>,

        /// DashScope API key (prefer DASHSCOPE_API_KEY to avoid process arguments)
        #[arg(long, env = "DASHSCOPE_API_KEY", hide_env_values = true)]
        dashscope_api_key: Option<String>,

        /// Gemini API key (prefer GEMINI_API_KEY to avoid process arguments)
        #[arg(long, env = "GEMINI_API_KEY", hide_env_values = true)]
        gemini_api_key: Option<String>,

        /// NVIDIA API key (prefer NVIDIA_API_KEY to avoid process arguments)
        #[arg(long, env = "NVIDIA_API_KEY", hide_env_values = true)]
        nvidia_api_key: Option<String>,

        /// NVIDIA hosted Riva function id (or set NVIDIA_RIVA_FUNCTION_ID)
        #[arg(long, env = "NVIDIA_RIVA_FUNCTION_ID")]
        nvidia_riva_function_id: Option<String>,

        /// NVIDIA Riva gRPC server (hosted default: grpc.nvcf.nvidia.com:443)
        #[arg(long, env = "NVIDIA_RIVA_SERVER")]
        nvidia_riva_server: Option<String>,

        /// Deepgram API key (prefer DEEPGRAM_API_KEY to avoid process arguments)
        #[arg(long, env = "DEEPGRAM_API_KEY", hide_env_values = true)]
        deepgram_api_key: Option<String>,

        /// Azure API key (prefer AZURE_API_KEY to avoid process arguments)
        #[arg(long, env = "AZURE_API_KEY", hide_env_values = true)]
        azure_api_key: Option<String>,

        /// Remote model name (for --provider openai, qwen-filetrans, gemini, or nvidia-riva)
        #[arg(long)]
        remote_model: Option<String>,

        /// DashScope API base URL for Qwen file transcription
        #[arg(
            long,
            env = "DASHSCOPE_ASR_BASE_URL",
            default_value = "https://dashscope-intl.aliyuncs.com/api/v1"
        )]
        qwen_api_base_url: String,

        /// Gemini API base URL
        #[arg(
            long,
            env = "GEMINI_API_BASE_URL",
            default_value = "https://generativelanguage.googleapis.com/v1beta"
        )]
        gemini_api_base_url: String,

        /// Deepgram API base URL
        #[arg(
            long,
            env = "DEEPGRAM_API_BASE_URL",
            default_value = "https://api.deepgram.com/v1"
        )]
        deepgram_api_base_url: String,

        /// Enable Deepgram summarization, topics, intents, entity detection, and sentiment
        #[arg(long, env = "DEEPGRAM_INTELLIGENCE")]
        deepgram_intelligence: bool,

        /// Enable Deepgram summarization (summarize=v2)
        #[arg(long, env = "DEEPGRAM_SUMMARIZE")]
        deepgram_summarize: bool,

        /// Enable Deepgram topic detection
        #[arg(long, env = "DEEPGRAM_TOPICS")]
        deepgram_topics: bool,

        /// Enable Deepgram intent recognition
        #[arg(long, env = "DEEPGRAM_INTENTS")]
        deepgram_intents: bool,

        /// Enable Deepgram entity detection
        #[arg(long, env = "DEEPGRAM_DETECT_ENTITIES")]
        deepgram_detect_entities: bool,

        /// Enable Deepgram sentiment analysis
        #[arg(long, env = "DEEPGRAM_SENTIMENT")]
        deepgram_sentiment: bool,

        /// Deepgram Nova-3 keyterm prompt; repeat or comma-separate up to provider limits
        #[arg(long, env = "DEEPGRAM_KEYTERM", value_delimiter = ',')]
        deepgram_keyterm: Vec<String>,

        /// Deepgram search term or phrase; repeat or comma-separate
        #[arg(long, env = "DEEPGRAM_SEARCH", value_delimiter = ',')]
        deepgram_search: Vec<String>,

        /// Deepgram redaction target, such as pii, phi, pci, numbers, or an entity class
        #[arg(long, env = "DEEPGRAM_REDACT", value_delimiter = ',')]
        deepgram_redact: Vec<String>,

        /// Deepgram find/replace rule in FIND:REPLACE format; repeat or comma-separate
        #[arg(long, env = "DEEPGRAM_REPLACE", value_delimiter = ',')]
        deepgram_replace: Vec<String>,

        /// Enable Deepgram filler word transcription
        #[arg(long, env = "DEEPGRAM_FILLER_WORDS")]
        deepgram_filler_words: bool,

        /// Enable Deepgram numerals formatting
        #[arg(long, env = "DEEPGRAM_NUMERALS")]
        deepgram_numerals: bool,

        /// Stage Deepgram input in S3-compatible storage and submit a pre-signed URL
        #[arg(long, env = "DEEPGRAM_USE_PRESIGNED_URL")]
        deepgram_use_presigned_url: bool,

        /// Reuse Gemini Files API uploads keyed by SHA-256 of prepared upload bytes
        #[arg(long, env = "GEMINI_FILE_CACHE")]
        gemini_file_cache: bool,

        /// Stage Gemini input in S3-compatible storage and pass a pre-signed URL as file_uri
        #[arg(long, env = "GEMINI_USE_PRESIGNED_URL")]
        gemini_use_presigned_url: bool,

        /// Local Gemini Files API cache index path
        #[arg(long, env = "GEMINI_FILE_CACHE_INDEX")]
        gemini_file_cache_index: Option<PathBuf>,

        /// Delete cached Gemini Files API uploads after the run
        #[arg(long, env = "GEMINI_AUTOCLEAN")]
        gemini_autoclean: bool,

        /// Create and reuse Gemini explicit cachedContent objects for prepared audio
        #[arg(long, env = "GEMINI_EXPLICIT_CACHE")]
        gemini_explicit_cache: bool,

        /// TTL in seconds for Gemini explicit cachedContent objects
        #[arg(long, env = "GEMINI_CACHE_TTL_SECS", default_value = "3600")]
        gemini_cache_ttl_secs: u64,

        /// Language code (e.g. en, fr, auto). If not set, auto-detection is used.
        #[arg(long)]
        language: Option<String>,

        /// Azure deployment name (or set AZURE_DEPLOYMENT_NAME env var)
        #[arg(long, env = "AZURE_DEPLOYMENT_NAME", default_value = "whisper")]
        azure_deployment: String,

        /// Azure API version (or set AZURE_API_VERSION env var)
        #[arg(long, env = "AZURE_API_VERSION", default_value = "2024-06-01")]
        azure_api_version: String,

        /// Maximum API request retries when rate limited
        #[arg(long, env = "TRANSCRIBEIT_MAX_RETRIES", default_value = "5")]
        max_retries: u32,

        /// Timeout in seconds for each API request
        #[arg(long, env = "TRANSCRIBEIT_REQUEST_TIMEOUT_SECS", default_value = "120")]
        request_timeout_secs: u64,

        /// Base retry wait in seconds when rate limited
        #[arg(long, env = "TRANSCRIBEIT_RETRY_WAIT_BASE_SECS", default_value = "10")]
        retry_wait_base_secs: u64,

        /// Maximum retry wait in seconds when rate limited
        #[arg(long, env = "TRANSCRIBEIT_RETRY_WAIT_MAX_SECS", default_value = "120")]
        retry_wait_max_secs: u64,

        /// Output directory for VTT and manifest files
        #[arg(short = 'o', long)]
        output_dir: Option<PathBuf>,

        /// Output format
        #[arg(short = 'f', long, default_value = "vtt")]
        output_format: OutputFormatArg,

        /// Post-transcription analysis to run (comma-separated, e.g. summary)
        #[arg(long, value_delimiter = ',')]
        analysis: Vec<AnalysisKind>,

        /// Enable silence-based segmentation
        #[arg(long)]
        segment: bool,

        /// Silence detection threshold in dB (negative value)
        #[arg(long, default_value = "-40", value_parser = parse_finite_f64)]
        silence_threshold: f64,

        /// Minimum silence duration in seconds
        #[arg(long, default_value = "0.8", value_parser = parse_positive_f64)]
        min_silence_duration: f64,

        /// Maximum segment length in seconds
        #[arg(long, default_value = "600", value_parser = parse_positive_f64)]
        max_segment_secs: f64,

        /// Maximum parallel segment requests (API providers only; local remains sequential)
        #[arg(long, default_value = "2", value_parser = parse_positive_usize)]
        segment_concurrency: usize,

        /// Normalize audio with ffmpeg loudnorm before transcription
        #[arg(long)]
        normalize: bool,

        /// Deprecated compatibility flag; staged URL resources are now deleted by default
        #[arg(long, env = "TRANSCRIBEIT_AUTOCLEAN")]
        autoclean: bool,

        /// Retain staged S3/R2 URL resources after the provider request
        #[arg(
            long,
            env = "TRANSCRIBEIT_KEEP_STAGED_RESOURCES",
            conflicts_with_all = ["autoclean", "gemini_autoclean"]
        )]
        keep_staged_resources: bool,

        /// Enable speaker diarization
        #[arg(long)]
        diarize: bool,

        /// Speaker count or provider-specific maximum speaker hint for diarization
        #[arg(long, value_parser = parse_positive_i32)]
        speakers: Option<i32>,

        /// S3 bucket used to stage audio for providers that need pre-signed URLs
        #[arg(long, env = "S3_BUCKET")]
        s3_bucket: Option<String>,

        /// S3 region used to stage audio for providers that need pre-signed URLs
        #[arg(long, env = "S3_REGION")]
        s3_region: Option<String>,

        /// S3-compatible endpoint URL (optional for AWS S3)
        #[arg(long, env = "S3_ENDPOINT_URL")]
        s3_endpoint_url: Option<String>,

        /// S3 access key ID (prefer S3_ACCESS_KEY_ID to avoid process arguments)
        #[arg(long, env = "S3_ACCESS_KEY_ID", hide_env_values = true)]
        s3_access_key_id: Option<String>,

        /// S3 secret access key (prefer S3_SECRET_ACCESS_KEY to avoid process arguments)
        #[arg(long, env = "S3_SECRET_ACCESS_KEY", hide_env_values = true)]
        s3_secret_access_key: Option<String>,

        /// S3 session token (prefer S3_SESSION_TOKEN to avoid process arguments)
        #[arg(long, env = "S3_SESSION_TOKEN", hide_env_values = true)]
        s3_session_token: Option<String>,

        /// S3 object prefix for temporary remote-provider uploads
        #[arg(long, env = "S3_PREFIX")]
        s3_prefix: Option<String>,

        /// Pre-signed URL expiry in seconds
        #[arg(long, env = "S3_PRESIGN_EXPIRES_SECS", default_value = "3600")]
        s3_presign_expires_secs: u64,

        /// Force path-style S3 URLs for S3-compatible providers
        #[arg(long, env = "S3_FORCE_PATH_STYLE")]
        s3_force_path_style: bool,
    },
}

fn parse_finite_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("expected a number, got '{value}'"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err("value must be finite".to_string())
    }
}

fn parse_positive_f64(value: &str) -> Result<f64, String> {
    let parsed = parse_finite_f64(value)?;
    if parsed >= 0.001 {
        Ok(parsed)
    } else {
        Err("duration must be at least 0.001 seconds".to_string())
    }
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got '{value}'"))?;
    if parsed > 0 {
        Ok(parsed)
    } else {
        Err("value must be greater than zero".to_string())
    }
}

fn parse_positive_i32(value: &str) -> Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| format!("expected a positive integer, got '{value}'"))?;
    if parsed > 0 {
        Ok(parsed)
    } else {
        Err("value must be greater than zero".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn run_rejects_non_positive_segmentation_values() {
        for arguments in [
            vec![
                "transcribeit",
                "run",
                "-i",
                "audio.wav",
                "--max-segment-secs",
                "0",
            ],
            vec![
                "transcribeit",
                "run",
                "-i",
                "audio.wav",
                "--min-silence-duration",
                "-1",
            ],
            vec![
                "transcribeit",
                "run",
                "-i",
                "audio.wav",
                "--segment-concurrency",
                "0",
            ],
            vec!["transcribeit", "run", "-i", "audio.wav", "--speakers", "0"],
        ] {
            assert!(Cli::try_parse_from(arguments).is_err());
        }
    }

    #[test]
    fn run_rejects_non_finite_segmentation_values() {
        assert!(
            Cli::try_parse_from([
                "transcribeit",
                "run",
                "-i",
                "audio.wav",
                "--max-segment-secs",
                "NaN",
            ])
            .is_err()
        );
    }

    #[test]
    fn retired_sherpa_cli_surface_is_rejected() {
        for arguments in [
            vec![
                "transcribeit",
                "run",
                "-i",
                "audio.wav",
                "--provider",
                "sherpa-onnx",
            ],
            vec!["transcribeit", "download-model", "--format", "onnx"],
            vec![
                "transcribeit",
                "run",
                "-i",
                "audio.wav",
                "--vad-model",
                "silero_vad.onnx",
            ],
            vec!["transcribeit", "setup", "--component", "sherpa-libs"],
        ] {
            assert!(Cli::try_parse_from(arguments).is_err());
        }
    }
}
