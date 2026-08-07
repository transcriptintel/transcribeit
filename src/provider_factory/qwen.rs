use anyhow::Result;

use super::s3::build_uploader;
use super::{ProviderFactoryArgs, ProviderRuntime, owned};
use crate::credentials::resolve_provider_key;
use crate::engines::qwen_filetrans::QwenFileTrans;

pub(super) async fn build(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let key = resolve_provider_key(
        owned(args.dashscope_api_key),
        owned(args.explicit_api_key),
        "qwen-filetrans",
        "--dashscope-api-key",
        "DASHSCOPE_API_KEY",
    )?;
    let uploader = build_uploader(
        args,
        "transcribeit/qwen-filetrans",
        "--provider qwen-filetrans",
    )
    .await?;
    let model_name = args
        .remote_model
        .unwrap_or("qwen3-asr-flash-filetrans")
        .to_string();
    Ok(ProviderRuntime {
        engine: Box::new(QwenFileTrans::new(
            args.qwen_api_base_url.to_string(),
            key,
            model_name.clone(),
            owned(args.language),
            args.settings,
            uploader,
            args.cleanup_staged_resources,
        )?),
        analyzer: None,
        provider_name: "qwen-filetrans".into(),
        model_name,
    })
}
