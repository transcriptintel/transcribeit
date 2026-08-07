use anyhow::{Context, Result};

use super::{ProviderFactoryArgs, ProviderRuntime, owned};
use crate::credentials::resolve_provider_key;
use crate::engines::nvidia_riva::NvidiaRiva;

pub(super) fn build(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let key = resolve_provider_key(
        owned(args.nvidia_api_key),
        owned(args.explicit_api_key),
        "nvidia-riva",
        "--nvidia-api-key",
        "NVIDIA_API_KEY",
    )?;
    let function_id = args
        .nvidia_riva_function_id
        .context(
            "--nvidia-riva-function-id or NVIDIA_RIVA_FUNCTION_ID is required for --provider nvidia-riva",
        )?
        .to_string();
    let riva_speakers = if args.diarize || args.speakers.is_some() {
        Some(args.speakers.unwrap_or(4))
    } else {
        None
    };
    let model_name = args.remote_model.map_or_else(
        || {
            let prefix = function_id.chars().take(8).collect::<String>();
            format!("function:{prefix}")
        },
        str::to_owned,
    );
    Ok(ProviderRuntime {
        engine: Box::new(NvidiaRiva::new(
            owned(args.nvidia_riva_server),
            key,
            function_id,
            owned(args.remote_model),
            owned(args.language),
            args.settings.request_timeout,
            riva_speakers,
        )),
        analyzer: None,
        provider_name: "nvidia-riva".into(),
        model_name,
    })
}
