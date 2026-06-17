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
mod storage;
mod transcriber;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use crate::audio::extract::check_ffmpeg;
use crate::cli::{Cli, Command, ModelFormat, OutputFormatArg, Provider, SetupComponent};
use crate::engines::azure_openai::AzureOpenAi;
use crate::engines::gemini::GeminiApi;
use crate::engines::model_cache::ModelCache;
use crate::engines::nvidia_riva::NvidiaRiva;
use crate::engines::openai_api::OpenAiApi;
use crate::engines::qwen_filetrans::QwenFileTrans;
use crate::engines::qwen_filetrans::limits::{
    is_filetrans_model as is_qwen_filetrans_model,
    validate_model_for_path as validate_qwen_model_for_path,
};
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
use crate::storage::s3::{S3ConfigInput, S3Uploader, s3_config_from_input};
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
            dashscope_api_key,
            gemini_api_key,
            nvidia_api_key,
            nvidia_riva_function_id,
            nvidia_riva_server,
            azure_api_key,
            remote_model,
            qwen_api_base_url,
            gemini_api_base_url,
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
            diarize,
            speakers,
            diarize_segmentation_model,
            diarize_embedding_model,
            vad_model,
            s3_bucket,
            s3_region,
            s3_endpoint_url,
            s3_access_key_id,
            s3_secret_access_key,
            s3_session_token,
            s3_prefix,
            s3_presign_expires_secs,
            s3_force_path_style,
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

            let qwen_filetrans_model = remote_model
                .as_deref()
                .unwrap_or("qwen3-asr-flash-filetrans");
            let qwen_needs_mp3_staging = matches!(provider, Provider::QwenFiletrans)
                && is_qwen_filetrans_model(qwen_filetrans_model);
            let openai_style_upload = matches!(provider, Provider::Openai | Provider::Azure);
            let nvidia_riva_upload = matches!(provider, Provider::NvidiaRiva);
            let gemini_needs_mp3_upload = matches!(provider, Provider::Gemini);
            let upload_as_mp3 =
                openai_style_upload || qwen_needs_mp3_staging || gemini_needs_mp3_upload;
            #[cfg(feature = "sherpa-onnx")]
            let is_sherpa = matches!(provider, Provider::SherpaOnnx);
            #[cfg(not(feature = "sherpa-onnx"))]
            let is_sherpa = false;
            let auto_split =
                openai_style_upload || qwen_needs_mp3_staging || nvidia_riva_upload || is_sherpa;
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
                        let model_name = remote_model.unwrap_or_else(|| {
                            if diarize || speakers.is_some() {
                                "gpt-4o-transcribe-diarize".to_string()
                            } else {
                                "whisper-1".to_string()
                            }
                        });
                        (
                            Box::new(OpenAiApi::new(
                                url,
                                key,
                                model_name.clone(),
                                language.clone(),
                                api_settings,
                            )?),
                            "openai".into(),
                            model_name,
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
                    Provider::QwenFiletrans => {
                        let key = dashscope_api_key.or(api_key).context(
                            "--dashscope-api-key, --api-key, DASHSCOPE_API_KEY, or OPENAI_API_KEY is required for --provider qwen-filetrans",
                        )?;
                        let s3_config = s3_config_from_input(S3ConfigInput {
                            bucket: s3_bucket.context(
                                "--s3-bucket or S3_BUCKET is required for --provider qwen-filetrans",
                            )?,
                            region: s3_region.or_else(|| std::env::var("AWS_REGION").ok()).context(
                                "--s3-region, S3_REGION, or AWS_REGION is required for --provider qwen-filetrans",
                            )?,
                            endpoint_url: s3_endpoint_url,
                            access_key_id: s3_access_key_id
                                .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
                                .context(
                                "--s3-access-key-id, S3_ACCESS_KEY_ID, or AWS_ACCESS_KEY_ID is required for --provider qwen-filetrans",
                            )?,
                            secret_access_key: s3_secret_access_key
                                .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
                                .context(
                                "--s3-secret-access-key, S3_SECRET_ACCESS_KEY, or AWS_SECRET_ACCESS_KEY is required for --provider qwen-filetrans",
                            )?,
                            session_token: s3_session_token.or_else(|| std::env::var("AWS_SESSION_TOKEN").ok()),
                            prefix: s3_prefix,
                            presign_expires_secs: s3_presign_expires_secs,
                            force_path_style: s3_force_path_style,
                        })?;
                        let uploader = S3Uploader::new(s3_config).await?;
                        let model_name =
                            remote_model.unwrap_or_else(|| "qwen3-asr-flash-filetrans".into());
                        (
                            Box::new(QwenFileTrans::new(
                                qwen_api_base_url,
                                key,
                                model_name.clone(),
                                language.clone(),
                                api_settings,
                                uploader,
                            )?),
                            "qwen-filetrans".into(),
                            model_name,
                        )
                    }
                    Provider::Gemini => {
                        let key = gemini_api_key
                            .or(api_key)
                            .context(
                                "--gemini-api-key, GEMINI_API_KEY, --api-key, or OPENAI_API_KEY is required for --provider gemini",
                            )?;
                        let model_name = remote_model.unwrap_or_else(|| "gemini-3.5-flash".into());
                        (
                            Box::new(GeminiApi::new(
                                gemini_api_base_url,
                                key,
                                model_name.clone(),
                                language.clone(),
                                api_settings,
                            )?),
                            "gemini".into(),
                            model_name,
                        )
                    }
                    Provider::NvidiaRiva => {
                        let key = nvidia_api_key.or(api_key).context(
                            "--nvidia-api-key, NVIDIA_API_KEY, --api-key, or OPENAI_API_KEY is required for --provider nvidia-riva",
                        )?;
                        let function_id = nvidia_riva_function_id.context(
                            "--nvidia-riva-function-id or NVIDIA_RIVA_FUNCTION_ID is required for --provider nvidia-riva",
                        )?;
                        let riva_speakers = if diarize || speakers.is_some() {
                            Some(speakers.unwrap_or(4))
                        } else {
                            None
                        };
                        let riva_model = remote_model.clone();
                        let model_name = remote_model.unwrap_or_else(|| {
                            let prefix = function_id.chars().take(8).collect::<String>();
                            format!("function:{prefix}")
                        });
                        (
                            Box::new(NvidiaRiva::new(
                                nvidia_riva_server,
                                key,
                                function_id,
                                riva_model,
                                language.clone(),
                                api_settings.request_timeout,
                                riva_speakers,
                            )),
                            "nvidia-riva".into(),
                            model_name,
                        )
                    }
                };
            let requested_diarization = diarize || speakers.is_some();
            let provider_native_diarization =
                provider_handles_diarization(&provider_name, &model_name);
            let local_diarize = requested_diarization && !provider_native_diarization;
            if local_diarize && !cfg!(feature = "sherpa-onnx") {
                anyhow::bail!(
                    "--diarize for provider '{provider_name}' requires local Sherpa diarization. Build with --features sherpa-onnx and pass --speakers plus diarization models, or use provider-native diarization with nvidia-riva or openai --remote-model gpt-4o-transcribe-diarize."
                );
            }

            for (index, input_path) in input_paths.iter().enumerate() {
                if input_paths.len() > 1 {
                    eprintln!(
                        "[{} / {}] Processing {}",
                        index + 1,
                        input_paths.len(),
                        input_path.display()
                    );
                }

                if provider_name == "qwen-filetrans" && !is_qwen_filetrans_model(&model_name) {
                    validate_qwen_model_for_path(&model_name, input_path).await?;
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
                    diarize: local_diarize,
                    speakers: local_diarize.then_some(speakers).flatten(),
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

fn provider_handles_diarization(provider_name: &str, model_name: &str) -> bool {
    match provider_name {
        "nvidia-riva" | "gemini" => true,
        "openai" => model_name.eq_ignore_ascii_case("gpt-4o-transcribe-diarize"),
        _ => false,
    }
}
