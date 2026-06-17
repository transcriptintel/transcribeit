use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

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

    #[cfg(feature = "sherpa-onnx")]
    pub(crate) fn onnx_archive_name(&self) -> Option<&str> {
        match self {
            Self::Tiny => Some("sherpa-onnx-whisper-tiny"),
            Self::TinyEn => Some("sherpa-onnx-whisper-tiny.en"),
            Self::Base => Some("sherpa-onnx-whisper-base"),
            Self::BaseEn => Some("sherpa-onnx-whisper-base.en"),
            Self::Small => Some("sherpa-onnx-whisper-small"),
            Self::SmallEn => Some("sherpa-onnx-whisper-small.en"),
            Self::Medium => Some("sherpa-onnx-whisper-medium"),
            Self::MediumEn => Some("sherpa-onnx-whisper-medium.en"),
            Self::LargeV3 => None,
            Self::LargeV3Turbo => Some("sherpa-onnx-whisper-turbo"),
        }
    }
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub(crate) enum ModelFormat {
    /// GGML format (for whisper.cpp / --provider local)
    #[default]
    Ggml,
    /// ONNX format (for sherpa-onnx / --provider sherpa-onnx)
    Onnx,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum Provider {
    /// Local whisper.cpp engine
    Local,
    /// Local sherpa-onnx engine (Whisper ONNX models)
    #[cfg(feature = "sherpa-onnx")]
    #[value(name = "sherpa-onnx")]
    SherpaOnnx,
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

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum SetupComponent {
    /// Default STT models (GGML base)
    Models,
    /// Silero VAD model for speech-aware segmentation
    Vad,
    /// Speaker diarization models (segmentation + embedding)
    Diarize,
    /// sherpa-onnx shared libraries for the current platform
    #[value(name = "sherpa-libs")]
    SherpaLibs,
}

#[derive(Parser)]
#[command(name = "transcribeit", about = "Transcribe audio files")]
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

        /// Directory for models (overrides MODEL_CACHE_DIR)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Hugging Face token for model downloads
        #[arg(short = 't', long, env = "HF_TOKEN")]
        hf_token: Option<String>,
    },

    /// Download a Whisper model
    DownloadModel {
        /// Model size to download
        #[arg(short = 's', long, default_value = "base")]
        model_size: ModelSize,

        /// Model format: ggml (for whisper.cpp) or onnx (for sherpa-onnx)
        #[arg(short, long, default_value = "ggml")]
        format: ModelFormat,

        /// Directory to save the model (overrides MODEL_CACHE_DIR)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Hugging Face token (optional, or set HF_TOKEN env var)
        #[arg(short = 't', long, env = "HF_TOKEN")]
        hf_token: Option<String>,

        /// Also download VAD model (silero_vad.onnx)
        #[arg(long)]
        vad: bool,

        /// Also download diarization models (segmentation + embedding)
        #[arg(long)]
        diarize: bool,
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

        /// API key (or set OPENAI_API_KEY / AZURE_API_KEY env var)
        #[arg(short, long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        /// DashScope API key for Qwen providers (or set DASHSCOPE_API_KEY)
        #[arg(long, env = "DASHSCOPE_API_KEY")]
        dashscope_api_key: Option<String>,

        /// Gemini API key (or set GEMINI_API_KEY)
        #[arg(long, env = "GEMINI_API_KEY")]
        gemini_api_key: Option<String>,

        /// NVIDIA API key for hosted Riva endpoints (or set NVIDIA_API_KEY)
        #[arg(long, env = "NVIDIA_API_KEY")]
        nvidia_api_key: Option<String>,

        /// NVIDIA hosted Riva function id (or set NVIDIA_RIVA_FUNCTION_ID)
        #[arg(long, env = "NVIDIA_RIVA_FUNCTION_ID")]
        nvidia_riva_function_id: Option<String>,

        /// NVIDIA Riva gRPC server (hosted default: grpc.nvcf.nvidia.com:443)
        #[arg(long, env = "NVIDIA_RIVA_SERVER")]
        nvidia_riva_server: Option<String>,

        /// Azure API key (or set AZURE_API_KEY env var)
        #[arg(long, env = "AZURE_API_KEY")]
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
        #[arg(long, default_value = "-40")]
        silence_threshold: f64,

        /// Minimum silence duration in seconds
        #[arg(long, default_value = "0.8")]
        min_silence_duration: f64,

        /// Maximum segment length in seconds
        #[arg(long, default_value = "600")]
        max_segment_secs: f64,

        /// Maximum parallel segment requests (API providers only; local remains sequential)
        #[arg(long, default_value = "2")]
        segment_concurrency: usize,

        /// Normalize audio with ffmpeg loudnorm before transcription
        #[arg(long)]
        normalize: bool,

        /// Enable speaker diarization
        #[arg(long)]
        diarize: bool,

        /// Speaker count or provider-specific maximum speaker hint for diarization
        #[arg(long)]
        speakers: Option<i32>,

        /// Path to speaker segmentation model (pyannote ONNX)
        #[arg(long, env = "DIARIZE_SEGMENTATION_MODEL")]
        diarize_segmentation_model: Option<String>,

        /// Path to speaker embedding model (ONNX)
        #[arg(long, env = "DIARIZE_EMBEDDING_MODEL")]
        diarize_embedding_model: Option<String>,

        /// Path to Silero VAD model for speech-aware segmentation (avoids mid-word cuts)
        #[arg(long, env = "VAD_MODEL")]
        vad_model: Option<String>,

        /// S3 bucket used to stage audio for Qwen file transcription
        #[arg(long, env = "S3_BUCKET")]
        s3_bucket: Option<String>,

        /// S3 region used to stage audio for Qwen file transcription
        #[arg(long, env = "S3_REGION")]
        s3_region: Option<String>,

        /// S3-compatible endpoint URL (optional for AWS S3)
        #[arg(long, env = "S3_ENDPOINT_URL")]
        s3_endpoint_url: Option<String>,

        /// S3 access key ID
        #[arg(long, env = "S3_ACCESS_KEY_ID")]
        s3_access_key_id: Option<String>,

        /// S3 secret access key
        #[arg(long, env = "S3_SECRET_ACCESS_KEY")]
        s3_secret_access_key: Option<String>,

        /// S3 session token, when using temporary credentials
        #[arg(long, env = "S3_SESSION_TOKEN")]
        s3_session_token: Option<String>,

        /// S3 object prefix for temporary Qwen uploads
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
