mod audio;
mod cli;
#[cfg(feature = "sherpa-onnx")]
mod diarize;
mod engines;
mod input;
mod models;
mod output;
mod pipeline;
mod pipeline_output;
mod setup;
mod transcriber;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use crate::audio::extract::check_ffmpeg;
use crate::cli::{Cli, Command, ModelFormat, OutputFormatArg, Provider, SetupComponent};
use crate::engines::azure_openai::AzureOpenAi;
use crate::engines::model_cache::ModelCache;
use crate::engines::openai_api::OpenAiApi;
use crate::engines::rate_limit;
#[cfg(feature = "sherpa-onnx")]
use crate::engines::sherpa_onnx::SherpaOnnxEngine;
use crate::engines::whisper_local::WhisperLocal;
use crate::input::resolve_input_paths;
use crate::models::{download_model, list_models, resolve_cached_model_path};
#[cfg(feature = "sherpa-onnx")]
use crate::models::{download_onnx_model, resolve_onnx_model_dir};
use crate::pipeline::{OutputFormat, PipelineConfig, run_pipeline};
use crate::setup::{
    print_setup_summary, setup_diarize, setup_models, setup_sherpa_libs, setup_vad,
};
use crate::transcriber::Transcriber;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Command::Setup {
            component,
            output_dir,
            hf_token,
        } => {
            let components = match component {
                Some(c) => vec![c],
                None => vec![
                    SetupComponent::Models,
                    SetupComponent::Vad,
                    SetupComponent::Diarize,
                    SetupComponent::SherpaLibs,
                ],
            };

            let mut summary: Vec<(&str, String)> = Vec::new();

            for comp in &components {
                match comp {
                    SetupComponent::Models => {
                        let status = setup_models(output_dir.clone(), hf_token.as_deref()).await?;
                        summary.push(("models", status));
                    }
                    SetupComponent::Vad => {
                        let status = setup_vad(output_dir.clone()).await?;
                        summary.push(("vad", status));
                    }
                    SetupComponent::Diarize => {
                        let status = setup_diarize(output_dir.clone()).await?;
                        summary.push(("diarize", status));
                    }
                    SetupComponent::SherpaLibs => {
                        let status = setup_sherpa_libs().await?;
                        summary.push(("sherpa-libs", status));
                    }
                }
            }

            print_setup_summary(&summary);
        }

        Command::DownloadModel {
            model_size,
            format,
            output_dir,
            hf_token,
            vad,
            diarize,
        } => {
            match format {
                ModelFormat::Ggml => {
                    download_model(&model_size, output_dir.clone(), hf_token.as_deref()).await?;
                }
                ModelFormat::Onnx => {
                    #[cfg(feature = "sherpa-onnx")]
                    download_onnx_model(&model_size, output_dir.clone()).await?;
                    #[cfg(not(feature = "sherpa-onnx"))]
                    anyhow::bail!(
                        "ONNX model download requires the 'sherpa-onnx' feature. Build with: cargo build --features sherpa-onnx"
                    );
                }
            }
            if vad {
                setup_vad(output_dir.clone()).await?;
            }
            if diarize {
                setup_diarize(output_dir).await?;
            }
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
            speakers,
            diarize_segmentation_model,
            diarize_embedding_model,
            vad_model,
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
            #[cfg(feature = "sherpa-onnx")]
            let is_sherpa = matches!(provider, Provider::SherpaOnnx);
            #[cfg(not(feature = "sherpa-onnx"))]
            let is_sherpa = false;
            let auto_split = upload_as_mp3 || is_sherpa;
            let max_segment_secs = if is_sherpa {
                // sherpa-onnx Whisper only supports ≤30s per call
                max_segment_secs.min(30.0)
            } else {
                max_segment_secs
            };
            let segment = segment || is_sherpa;
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
                    #[cfg(feature = "sherpa-onnx")]
                    Provider::SherpaOnnx => {
                        let model_arg =
                            model.context("--model is required for --provider sherpa-onnx")?;
                        let model_dir = resolve_onnx_model_dir(&model_arg)?;
                        let name = model_dir.display().to_string();
                        (
                            Box::new(SherpaOnnxEngine::new(model_dir, language.clone())?),
                            "sherpa-onnx".into(),
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
                    speakers,
                    diarize_segmentation_model: diarize_segmentation_model.clone(),
                    diarize_embedding_model: diarize_embedding_model.clone(),
                    vad_model: vad_model.clone(),
                };

                run_pipeline(engine.as_ref(), config).await?;
            }
        }
    }

    Ok(())
}
