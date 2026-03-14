mod audio;
mod engines;
mod output;
mod pipeline;
mod transcriber;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

use crate::audio::extract::check_ffmpeg;
use crate::engines::azure_openai::AzureOpenAi;
use crate::engines::model_cache::ModelCache;
use crate::engines::openai_api::OpenAiApi;
use crate::engines::rate_limit;
use crate::engines::whisper_local::WhisperLocal;
use crate::pipeline::{OutputFormat, PipelineConfig, run_pipeline};
use crate::transcriber::Transcriber;

const HF_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| ".cache".to_string()))
}

fn resolve_cached_model_path(model: &str) -> Result<String> {
    let model = model.trim();
    if model.is_empty() {
        anyhow::bail!("Model name cannot be empty");
    }

    let direct_path = Path::new(model);
    if direct_path.exists() {
        return Ok(direct_path.to_string_lossy().into_owned());
    }

    let file_name = match model {
        "tiny" => Some("ggml-tiny.bin"),
        "tiny.en" => Some("ggml-tiny.en.bin"),
        "base" => Some("ggml-base.bin"),
        "base.en" => Some("ggml-base.en.bin"),
        "small" => Some("ggml-small.bin"),
        "small.en" => Some("ggml-small.en.bin"),
        "medium" => Some("ggml-medium.bin"),
        "medium.en" => Some("ggml-medium.en.bin"),
        "large-v3" => Some("ggml-large-v3.bin"),
        "large-v3-turbo" => Some("ggml-large-v3-turbo.bin"),
        _ => None,
    };

    if let Some(file_name) = file_name {
        let cache_path = models_dir().join(file_name);
        if cache_path.exists() {
            return Ok(cache_path.to_string_lossy().into_owned());
        }
        anyhow::bail!(
            "Model '{model}' not found in cache directory '{}'. Download it with: transcribeit download-model -s {model}",
            models_dir().display()
        );
    }

    if !model.contains(std::path::MAIN_SEPARATOR) && model.ends_with(".bin") {
        let cache_path = models_dir().join(model);
        if cache_path.exists() {
            return Ok(cache_path.to_string_lossy().into_owned());
        }
        anyhow::bail!(
            "Model file '{model}' not found in cache directory '{}'. Set --model to an existing path or download it first.",
            models_dir().display()
        );
    }

    anyhow::bail!(
        "Model '{model}' is not a recognized alias. Use a GGML model path or one of: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v3, large-v3-turbo."
    );
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelSize {
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
    fn file_name(&self) -> &str {
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
enum Provider {
    /// Local whisper.cpp engine
    Local,
    /// OpenAI-compatible API
    Openai,
    /// Azure OpenAI API
    Azure,
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormatArg {
    /// Plain text to stdout
    Text,
    /// WebVTT subtitle format
    Vtt,
    /// SRT subtitle format
    Srt,
}

#[derive(Parser)]
#[command(name = "transcribeit", about = "Transcribe audio files")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Download a Whisper GGML model from Hugging Face
    DownloadModel {
        /// Model size to download
        #[arg(short = 's', long, default_value = "base")]
        model_size: ModelSize,

        /// Directory to save the model (overrides MODEL_CACHE_DIR)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Hugging Face token (optional, or set HF_TOKEN env var)
        #[arg(short = 't', long, env = "HF_TOKEN")]
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

        /// API key (or set OPENAI_API_KEY / AZURE_API_KEY env var)
        #[arg(short, long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        /// Azure API key (or set AZURE_API_KEY env var)
        #[arg(long, env = "AZURE_API_KEY")]
        azure_api_key: Option<String>,

        /// Remote model name (for --provider openai)
        #[arg(long, default_value = "whisper-1")]
        remote_model: String,

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
        #[arg(long, default_value = "text")]
        output_format: OutputFormatArg,

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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Command::DownloadModel {
            model_size,
            output_dir,
            hf_token,
        } => {
            download_model(&model_size, output_dir, hf_token.as_deref()).await?;
        }

        Command::ListModels { dir } => {
            list_models(dir)?;
        }

        Command::Run {
            provider,
            input,
            model,
            base_url,
            api_key,
            azure_api_key,
            remote_model,
            language,
            azure_deployment,
            azure_api_version,
            output_dir,
            output_format,
            segment,
            silence_threshold,
            min_silence_duration,
            max_segment_secs,
            segment_concurrency,
            normalize,
            max_retries,
            request_timeout_secs,
            retry_wait_base_secs,
            retry_wait_max_secs,
        } => {
            check_ffmpeg()?;

            let input_paths = resolve_input_paths(&input)?;
            if input_paths.len() > 1 && output_dir.is_none() {
                eprintln!(
                    "Batch mode detected ({} files). Output will be printed per file unless --output-dir is set.",
                    input_paths.len()
                );
            }

            let api_settings = rate_limit::ApiRequestSettings::new(
                Duration::from_secs(request_timeout_secs.max(1)),
                max_retries,
                Duration::from_secs(retry_wait_base_secs.max(1)),
                Duration::from_secs(retry_wait_max_secs.max(1)),
            );

            let upload_as_mp3 = matches!(provider, Provider::Openai | Provider::Azure);
            let auto_split = upload_as_mp3;
            let segment_concurrency = if upload_as_mp3 {
                segment_concurrency.max(1)
            } else {
                1
            };

            let (engine, provider_name, model_name): (Box<dyn Transcriber>, String, String) =
                match provider {
                    Provider::Local => {
                        let model_path = resolve_cached_model_path(
                            &model.context("--model is required for --provider local")?,
                        )?;
                        let cache = Arc::new(ModelCache::new());
                        let name = model_path.clone();
                        (
                            Box::new(WhisperLocal::new(model_path, cache, language.clone())),
                            "local".into(),
                            name,
                        )
                    }
                    Provider::Openai => {
                        let key = api_key.context(
                            "--api-key or OPENAI_API_KEY is required for --provider openai",
                        )?;
                        let url = base_url.unwrap_or_else(|| "https://api.openai.com".into());
                        (
                            Box::new(OpenAiApi::new(
                                url,
                                key,
                                remote_model.clone(),
                                language.clone(),
                                api_settings,
                            )?),
                            "openai".into(),
                            remote_model,
                        )
                    }
                    Provider::Azure => {
                        let key = azure_api_key
                            .or(api_key)
                            .context(
                                "--azure-api-key or --api-key/AZURE_API_KEY is required for --provider azure",
                            )?;
                        let endpoint = base_url
                            .or_else(|| std::env::var("AZURE_OPENAI_ENDPOINT").ok())
                            .context(
                                "--base-url or AZURE_OPENAI_ENDPOINT is required for --provider azure",
                            )?;
                        (
                            Box::new(AzureOpenAi::new(
                                endpoint,
                                azure_deployment.clone(),
                                azure_api_version,
                                key,
                                language.clone(),
                                api_settings,
                            )?),
                            "azure".into(),
                            azure_deployment,
                        )
                    }
                };

            for (index, input_path) in input_paths.iter().enumerate() {
                if input_paths.len() > 1 {
                    eprintln!(
                        "[{} / {}] Processing {}",
                        index + 1,
                        input_paths.len(),
                        input_path.display()
                    );
                }

                let config = PipelineConfig {
                    input: input_path.clone(),
                    output_dir: output_dir.clone(),
                    output_format: match output_format {
                        OutputFormatArg::Text => OutputFormat::Text,
                        OutputFormatArg::Vtt => OutputFormat::Vtt,
                        OutputFormatArg::Srt => OutputFormat::Srt,
                    },
                    language: language.clone(),
                    segment,
                    silence_threshold,
                    min_silence_duration,
                    max_segment_secs,
                    provider_name: provider_name.clone(),
                    model_name: model_name.clone(),
                    auto_split_for_api: auto_split,
                    upload_as_mp3,
                    segment_concurrency,
                    normalize_audio: normalize,
                };

                run_pipeline(engine.as_ref(), config).await?;
            }
        }
    }

    Ok(())
}

fn resolve_input_paths(input: &str) -> Result<Vec<PathBuf>> {
    let input_path = Path::new(input);

    if input_path.exists() {
        if input_path.is_file() {
            return Ok(vec![input_path.to_path_buf()]);
        }

        if input_path.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(input_path)
                .with_context(|| format!("Failed to read directory: {}", input_path.display()))?
            {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    files.push(entry.path());
                }
            }

            if files.is_empty() {
                anyhow::bail!("No files found in directory: {}", input_path.display());
            }

            files.sort_unstable();
            return Ok(files);
        }

        anyhow::bail!(
            "Input exists but is not a file or directory: {}",
            input_path.display()
        );
    }

    let mut matches = Vec::new();
    for entry in glob::glob(input).with_context(|| format!("Invalid input pattern: {input}"))? {
        let path =
            entry.with_context(|| format!("Invalid input glob match for pattern: {input}"))?;
        if path.is_file() {
            matches.push(path);
        }
    }

    if matches.is_empty() {
        anyhow::bail!("No files matched input pattern: {input}");
    }

    matches.sort_unstable();
    Ok(matches)
}

async fn download_model(
    model_size: &ModelSize,
    output_dir: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<()> {
    let dir = output_dir.unwrap_or_else(models_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    let file_name = model_size.file_name();
    let dest = dir.join(file_name);

    if dest.exists() {
        println!("Model already exists: {}", dest.display());
        return Ok(());
    }

    let url = format!("{HF_BASE_URL}/{file_name}");
    println!("Downloading {file_name} ...");
    println!("  from: {url}");
    println!("  to:   {}", dest.display());

    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if let Some(token) = hf_token {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.context("Failed to start download")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed with status: {}", resp.status());
    }

    let total_size = resp.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")?
            .progress_chars("##-"),
    );

    let tmp_dest = dest.with_extension("bin.part");
    let mut file = tokio::fs::File::create(&tmp_dest)
        .await
        .context("Failed to create temp file")?;

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading download stream")?;
        file.write_all(&chunk).await.context("Failed to write")?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);

    tokio::fs::rename(&tmp_dest, &dest)
        .await
        .context("Failed to finalize download")?;

    pb.finish_and_clear();
    println!("Done: {}", dest.display());
    Ok(())
}

fn list_models(dir: Option<PathBuf>) -> Result<()> {
    let dir = dir.unwrap_or_else(models_dir);

    if !dir.exists() {
        println!("No models found. Run `transcribeit download-model` first.");
        return Ok(());
    }

    let mut found = false;
    for entry in std::fs::read_dir(&dir).context("Failed to read models directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("bin") {
            let size = entry.metadata()?.len();
            let size_mb = size as f64 / (1024.0 * 1024.0);
            println!(
                "  {} ({:.0} MB)",
                path.file_name().unwrap().to_string_lossy(),
                size_mb
            );
            found = true;
        }
    }

    if !found {
        println!("No models found in {}", dir.display());
    }

    Ok(())
}
