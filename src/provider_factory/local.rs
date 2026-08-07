use std::sync::Arc;

use anyhow::{Context, Result};

use super::{ProviderFactoryArgs, ProviderRuntime};
use crate::engines::model_cache::ModelCache;
use crate::engines::whisper_local::WhisperLocal;
use crate::models::resolve_cached_model_path;

pub(super) fn build_local(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let model = args
        .model
        .context("--model is required for --provider local")?;
    let model_path = resolve_cached_model_path(model)?;
    let model_name = model_path.clone();
    Ok(ProviderRuntime {
        engine: Box::new(WhisperLocal::new(
            model_path,
            Arc::new(ModelCache::new()),
            args.language.map(str::to_owned),
        )),
        analyzer: None,
        provider_name: "local".into(),
        model_name,
    })
}
