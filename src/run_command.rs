use std::time::Duration;

use anyhow::Result;

use crate::analysis::AnalysisConfig;
use crate::audio::extract::check_ffmpeg;
use crate::batch::validate_batch_output_plan;
use crate::cli::{AnalysisKind, Command, OutputFormatArg, Provider};
use crate::credentials::resolve_explicit_key;
use crate::engines::qwen_filetrans::limits::{
    is_filetrans_model as is_qwen_filetrans_model,
    validate_model_for_path as validate_qwen_model_for_path,
};
use crate::engines::rate_limit::{self, ApiRequestSettings};
use crate::input::resolve_input_paths;
use crate::pipeline::{API_UPLOAD_MAX_BYTES, OutputFormat, PipelineConfig, run_pipeline};
use crate::provider_factory::{self, ProviderFactoryArgs};

mod segmentation;

pub(crate) async fn execute(command: Command) -> Result<()> {
    let Command::Run {
        provider,
        input,
        model,
        base_url,
        api_key,
        api_key_file,
        dashscope_api_key,
        gemini_api_key,
        nvidia_api_key,
        nvidia_riva_function_id,
        nvidia_riva_server,
        deepgram_api_key,
        azure_api_key,
        remote_model,
        qwen_api_base_url,
        gemini_api_base_url,
        deepgram_api_base_url,
        deepgram_intelligence,
        deepgram_summarize,
        deepgram_topics,
        deepgram_intents,
        deepgram_detect_entities,
        deepgram_sentiment,
        deepgram_keyterm,
        deepgram_search,
        deepgram_redact,
        deepgram_replace,
        deepgram_filler_words,
        deepgram_numerals,
        deepgram_use_presigned_url,
        gemini_file_cache,
        gemini_use_presigned_url,
        gemini_file_cache_index,
        gemini_autoclean,
        gemini_explicit_cache,
        gemini_cache_ttl_secs,
        language,
        azure_deployment,
        azure_api_version,
        output_dir,
        output_format,
        analysis,
        segment,
        silence_threshold,
        min_silence_duration,
        max_segment_secs,
        segment_concurrency,
        normalize,
        autoclean,
        keep_staged_resources,
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
    } = command
    else {
        unreachable!("run command dispatcher received a non-run command");
    };

    let input_paths = resolve_input_paths(&input)?;
    validate_batch_output_plan(&input_paths, output_dir.as_deref())?;
    check_ffmpeg()?;
    if autoclean {
        eprintln!("--autoclean is deprecated because staged URL resources are deleted by default.");
    }

    let cleanup_staged_resources = !keep_staged_resources;
    let explicit_api_key = resolve_explicit_key(api_key, api_key_file.as_deref())?;
    let settings = request_settings(
        request_timeout_secs,
        max_retries,
        retry_wait_base_secs,
        retry_wait_max_secs,
    );
    let qwen_model = remote_model
        .as_deref()
        .unwrap_or("qwen3-asr-flash-filetrans");
    let qwen_needs_mp3_staging =
        matches!(&provider, Provider::QwenFiletrans) && is_qwen_filetrans_model(qwen_model);
    let openai_style_upload = matches!(&provider, Provider::Openai | Provider::Azure);
    let nvidia_riva_upload = matches!(&provider, Provider::NvidiaRiva);
    let upload_as_mp3 =
        openai_style_upload || qwen_needs_mp3_staging || matches!(&provider, Provider::Gemini);
    #[cfg(feature = "sherpa-onnx")]
    let is_sherpa = matches!(&provider, Provider::SherpaOnnx);
    #[cfg(not(feature = "sherpa-onnx"))]
    let is_sherpa = false;
    let auto_split = openai_style_upload || nvidia_riva_upload;
    let segment = segment || is_sherpa;
    let segment_concurrency = if upload_as_mp3 {
        segment_concurrency.max(1)
    } else {
        1
    };
    let analysis_config = AnalysisConfig {
        summary: analysis.contains(&AnalysisKind::Summary),
    };
    validate_analysis_request(&analysis_config, &provider, output_dir.as_deref())?;

    let factory_args = ProviderFactoryArgs {
        provider: &provider,
        model: model.as_deref(),
        base_url: base_url.as_deref(),
        explicit_api_key: explicit_api_key.as_deref(),
        dashscope_api_key: dashscope_api_key.as_deref(),
        gemini_api_key: gemini_api_key.as_deref(),
        nvidia_api_key: nvidia_api_key.as_deref(),
        nvidia_riva_function_id: nvidia_riva_function_id.as_deref(),
        nvidia_riva_server: nvidia_riva_server.as_deref(),
        deepgram_api_key: deepgram_api_key.as_deref(),
        azure_api_key: azure_api_key.as_deref(),
        remote_model: remote_model.as_deref(),
        qwen_api_base_url: &qwen_api_base_url,
        gemini_api_base_url: &gemini_api_base_url,
        deepgram_api_base_url: &deepgram_api_base_url,
        deepgram_intelligence,
        deepgram_summarize,
        deepgram_topics,
        deepgram_intents,
        deepgram_detect_entities,
        deepgram_sentiment,
        deepgram_keyterm: &deepgram_keyterm,
        deepgram_search: &deepgram_search,
        deepgram_redact: &deepgram_redact,
        deepgram_replace: &deepgram_replace,
        deepgram_filler_words,
        deepgram_numerals,
        deepgram_use_presigned_url,
        gemini_file_cache,
        gemini_use_presigned_url,
        gemini_file_cache_index: gemini_file_cache_index.as_deref(),
        gemini_autoclean: autoclean || gemini_autoclean,
        gemini_explicit_cache,
        gemini_cache_ttl_secs,
        language: language.as_deref(),
        azure_deployment: &azure_deployment,
        azure_api_version: &azure_api_version,
        diarize,
        speakers,
        s3_bucket: s3_bucket.as_deref(),
        s3_region: s3_region.as_deref(),
        s3_endpoint_url: s3_endpoint_url.as_deref(),
        s3_access_key_id: s3_access_key_id.as_deref(),
        s3_secret_access_key: s3_secret_access_key.as_deref(),
        s3_session_token: s3_session_token.as_deref(),
        s3_prefix: s3_prefix.as_deref(),
        s3_presign_expires_secs,
        s3_force_path_style,
        cleanup_staged_resources,
        settings,
        analysis: &analysis_config,
    };
    let runtime = provider_factory::build(&factory_args).await?;
    let max_segment_secs = segmentation::model_safe_max_segment_secs(
        &runtime.provider_name,
        &runtime.model_name,
        max_segment_secs,
    );
    let requested_diarization = diarize || speakers.is_some();
    let provider_native_diarization =
        provider_factory::handles_diarization(&runtime.provider_name, &runtime.model_name);
    let local_diarize = requested_diarization && !provider_native_diarization;
    if local_diarize && !cfg!(feature = "sherpa-onnx") {
        anyhow::bail!(
            "--diarize for provider '{}' requires local Sherpa diarization. Build with --features sherpa-onnx and pass --speakers plus diarization models, or use provider-native diarization with nvidia-riva or openai --remote-model gpt-4o-transcribe-diarize.",
            runtime.provider_name
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
        if runtime.provider_name == "qwen-filetrans"
            && !is_qwen_filetrans_model(&runtime.model_name)
        {
            validate_qwen_model_for_path(&runtime.model_name, input_path).await?;
        }

        let config = PipelineConfig {
            input: input_path.clone(),
            output_dir: output_dir.clone(),
            output_format: output_format_value(&output_format),
            language: language.clone(),
            segment,
            silence_threshold,
            min_silence_duration,
            max_segment_secs,
            provider_name: runtime.provider_name.clone(),
            model_name: runtime.model_name.clone(),
            auto_split_max_bytes: auto_split.then_some(API_UPLOAD_MAX_BYTES),
            upload_as_mp3,
            segment_concurrency,
            normalize_audio: normalize,
            diarize: local_diarize,
            speakers: local_diarize.then_some(speakers).flatten(),
            diarize_segmentation_model: diarize_segmentation_model.clone(),
            diarize_embedding_model: diarize_embedding_model.clone(),
            vad_model: vad_model.clone(),
            analysis: analysis_config.clone(),
        };
        run_pipeline(runtime.engine.as_ref(), runtime.analyzer.as_deref(), config).await?;
    }
    Ok(())
}

fn request_settings(
    request_timeout_secs: u64,
    max_retries: u32,
    retry_wait_base_secs: u64,
    retry_wait_max_secs: u64,
) -> ApiRequestSettings {
    rate_limit::ApiRequestSettings::new(
        Duration::from_secs(request_timeout_secs.max(1)),
        max_retries,
        Duration::from_secs(retry_wait_base_secs.max(1)),
        Duration::from_secs(retry_wait_max_secs.max(1)),
    )
}

fn validate_analysis_request(
    analysis: &AnalysisConfig,
    provider: &Provider,
    output_dir: Option<&std::path::Path>,
) -> Result<()> {
    if analysis.is_enabled() && output_dir.is_none() {
        anyhow::bail!("--analysis requires --output-dir so results can be written to the manifest");
    }
    if analysis.is_enabled() && !matches!(provider, Provider::Gemini) {
        anyhow::bail!("--analysis is currently supported only with --provider gemini");
    }
    Ok(())
}

fn output_format_value(output_format: &OutputFormatArg) -> OutputFormat {
    match output_format {
        OutputFormatArg::Text => OutputFormat::Text,
        OutputFormatArg::Vtt => OutputFormat::Vtt,
        OutputFormatArg::Srt => OutputFormat::Srt,
    }
}
