use anyhow::{Context, Result};

use super::{ProviderFactoryArgs, ProviderRuntime, owned};
use crate::credentials::{resolve_openai_key, resolve_provider_key};
use crate::engines::azure_openai::AzureOpenAi;
use crate::engines::openai_api::OpenAiApi;

pub(super) fn build_openai(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let key = resolve_openai_key(owned(args.explicit_api_key))?;
    let model_name = args.remote_model.map_or_else(
        || {
            if args.diarize || args.speakers.is_some() {
                "gpt-4o-transcribe-diarize".to_string()
            } else {
                "whisper-1".to_string()
            }
        },
        str::to_owned,
    );
    Ok(ProviderRuntime {
        engine: Box::new(OpenAiApi::new(
            args.base_url
                .unwrap_or("https://api.openai.com")
                .to_string(),
            key,
            model_name.clone(),
            owned(args.language),
            args.settings,
        )?),
        analyzer: None,
        provider_name: "openai".into(),
        model_name,
    })
}

pub(super) fn build_azure(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let key = resolve_provider_key(
        owned(args.azure_api_key),
        owned(args.explicit_api_key),
        "azure",
        "--azure-api-key",
        "AZURE_API_KEY",
    )?;
    let endpoint = args
        .base_url
        .map(str::to_owned)
        .or_else(|| std::env::var("AZURE_OPENAI_ENDPOINT").ok())
        .context("--base-url or AZURE_OPENAI_ENDPOINT is required for --provider azure")?;
    Ok(ProviderRuntime {
        engine: Box::new(AzureOpenAi::new(
            endpoint,
            args.azure_deployment.to_string(),
            args.azure_api_version.to_string(),
            key,
            owned(args.language),
            args.settings,
        )?),
        analyzer: None,
        provider_name: "azure".into(),
        model_name: args.azure_deployment.to_string(),
    })
}
