use std::sync::Arc;

use anyhow::{Context, Result};

use super::{ProviderFactoryArgs, ProviderRuntime};
use crate::engines::model_cache::ModelCache;
#[cfg(feature = "sherpa-onnx")]
use crate::engines::sherpa_onnx::SherpaOnnxEngine;
use crate::engines::whisper_local::WhisperLocal;
use crate::models::resolve_cached_model_path;
#[cfg(feature = "sherpa-onnx")]
use crate::models::resolve_onnx_model_dir;

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

#[cfg(feature = "sherpa-onnx")]
pub(super) fn build_sherpa(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let model = args
        .model
        .context("--model is required for --provider sherpa-onnx")?;
    let model_dir = resolve_onnx_model_dir(model)?;
    let model_name = model_dir.display().to_string();
    Ok(ProviderRuntime {
        engine: Box::new(SherpaOnnxEngine::new(
            model_dir,
            args.language.map(str::to_owned),
        )?),
        analyzer: None,
        provider_name: "sherpa-onnx".into(),
        model_name,
    })
}
