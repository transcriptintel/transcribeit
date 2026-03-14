mod audio;
mod engines;
mod output;
mod pipeline;
mod transcriber;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

use crate::audio::extract::check_ffmpeg;
use crate::engines::azure_openai::AzureOpenAi;
use crate::engines::model_cache::ModelCache;
use crate::engines::openai_api::OpenAiApi;
use crate::engines::whisper_local::WhisperLocal;
use crate::pipeline::{OutputFormat, PipelineConfig, run_pipeline};
use crate::transcriber::Transcriber;

const HF_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| ".cache".to_string()))
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
}

#[derive(Parser)]
#[command(name = "transcribeit", about = "Transcribe audio files")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a Whisper GGML model from Hugging Face
    DownloadModel {
        /// Model size to download
        #[arg(short = 's', long, default_value = "base")]
        model_size: ModelSize,

        /// Directory to save the model (overrides MODEL_CACHE_DIR)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Hugging Face token (or set HF_TOKEN env var)
        #[arg(short = 't', long, env = "HF_TOKEN")]
        hf_token: String,
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

        /// Path to input audio/video file
        #[arg(short, long)]
        input: PathBuf,

        /// Path to local model file (required for --provider local)
        #[arg(short, long)]
        model: Option<String>,

        /// API base URL (for --provider openai)
        #[arg(short, long, default_value = "https://api.openai.com")]
        base_url: String,

        /// API key (or set OPENAI_API_KEY / AZURE_API_KEY env var)
        #[arg(short, long, env = "OPENAI_API_KEY")]
        api_key: Option<String>,

        /// Remote model name (for --provider openai)
        #[arg(long, default_value = "whisper-1")]
        remote_model: String,

        /// Azure deployment name (for --provider azure)
        #[arg(long, default_value = "whisper")]
        azure_deployment: String,

        /// Azure API version (for --provider azure)
        #[arg(long, default_value = "2024-06-01")]
        azure_api_version: String,

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
            download_model(&model_size, output_dir, &hf_token).await?;
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
            remote_model,
            azure_deployment,
            azure_api_version,
            output_dir,
            output_format,
            segment,
            silence_threshold,
            min_silence_duration,
            max_segment_secs,
        } => {
            check_ffmpeg()?;

            let auto_split = matches!(provider, Provider::Openai | Provider::Azure);

            let (engine, provider_name, model_name): (Box<dyn Transcriber>, String, String) =
                match provider {
                    Provider::Local => {
                        let model_path =
                            model.context("--model is required for --provider local")?;
                        let cache = Arc::new(ModelCache::new());
                        let name = model_path.clone();
                        (
                            Box::new(WhisperLocal::new(model_path, cache)),
                            "local".into(),
                            name,
                        )
                    }
                    Provider::Openai => {
                        let key = api_key.context(
                            "--api-key or OPENAI_API_KEY is required for --provider openai",
                        )?;
                        (
                            Box::new(OpenAiApi::new(base_url, key, remote_model.clone())),
                            "openai".into(),
                            remote_model,
                        )
                    }
                    Provider::Azure => {
                        let key = api_key.context(
                            "--api-key or AZURE_API_KEY is required for --provider azure",
                        )?;
                        (
                            Box::new(AzureOpenAi::new(
                                base_url,
                                azure_deployment.clone(),
                                azure_api_version,
                                key,
                            )),
                            "azure".into(),
                            azure_deployment,
                        )
                    }
                };

            let config = PipelineConfig {
                input,
                output_dir,
                output_format: match output_format {
                    OutputFormatArg::Text => OutputFormat::Text,
                    OutputFormatArg::Vtt => OutputFormat::Vtt,
                },
                segment,
                silence_threshold,
                min_silence_duration,
                max_segment_secs,
                provider_name,
                model_name,
                auto_split_for_api: auto_split,
            };

            run_pipeline(engine, config).await?;
        }
    }

    Ok(())
}

async fn download_model(
    model_size: &ModelSize,
    output_dir: Option<PathBuf>,
    hf_token: &str,
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
    let resp = client
        .get(&url)
        .bearer_auth(hf_token)
        .send()
        .await
        .context("Failed to start download")?;

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
