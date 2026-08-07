use anyhow::Result;

use super::s3::build_uploader;
use super::{ProviderFactoryArgs, ProviderRuntime, owned};
use crate::analysis::TranscriptAnalyzer;
use crate::credentials::resolve_provider_key;
use crate::engines::gemini::{GeminiApi, GeminiConfig, GeminiFileCacheConfig};

const DEFAULT_GEMINI_MODEL: &str = "gemini-3.6-flash";

pub(super) async fn build(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let key = resolve_provider_key(
        owned(args.gemini_api_key),
        owned(args.explicit_api_key),
        "gemini",
        "--gemini-api-key",
        "GEMINI_API_KEY",
    )?;
    let model_name = args
        .remote_model
        .unwrap_or(DEFAULT_GEMINI_MODEL)
        .to_string();
    let autoclean = if args.gemini_use_presigned_url {
        args.cleanup_staged_resources
    } else {
        args.gemini_autoclean
    };
    if args.gemini_use_presigned_url && (args.gemini_file_cache || args.gemini_explicit_cache) {
        anyhow::bail!(
            "--gemini-use-presigned-url cannot be combined with --gemini-file-cache or --gemini-explicit-cache because signed URLs do not create reusable Gemini Files API handles"
        );
    }
    if args.gemini_use_presigned_url && is_gemini_2_0_model(&model_name) {
        anyhow::bail!(
            "--gemini-use-presigned-url is not supported for Gemini 2.0 family models; use Gemini Files API mode instead"
        );
    }

    let signed_url_uploader = if args.gemini_use_presigned_url {
        Some(
            build_uploader(
                args,
                "transcribeit/gemini",
                "--provider gemini --gemini-use-presigned-url",
            )
            .await?,
        )
    } else {
        None
    };
    let file_cache = if args.gemini_file_cache || args.gemini_explicit_cache {
        Some(GeminiFileCacheConfig {
            index_path: args.gemini_file_cache_index.map(ToOwned::to_owned),
            autoclean,
            explicit_cache: args.gemini_explicit_cache,
            explicit_cache_ttl_secs: args.gemini_cache_ttl_secs.max(60),
        })
    } else {
        None
    };
    let analyzer = args
        .analysis
        .is_enabled()
        .then(|| {
            GeminiApi::new(GeminiConfig {
                api_base_url: args.gemini_api_base_url.to_string(),
                api_key: key.clone(),
                model: model_name.clone(),
                language: owned(args.language),
                settings: args.settings,
                file_cache: None,
                signed_url_uploader: None,
                autoclean: false,
            })
            .map(|api| Box::new(api) as Box<dyn TranscriptAnalyzer>)
        })
        .transpose()?;

    Ok(ProviderRuntime {
        engine: Box::new(GeminiApi::new(GeminiConfig {
            api_base_url: args.gemini_api_base_url.to_string(),
            api_key: key,
            model: model_name.clone(),
            language: owned(args.language),
            settings: args.settings,
            file_cache,
            signed_url_uploader,
            autoclean,
        })?),
        analyzer,
        provider_name: "gemini".into(),
        model_name,
    })
}

fn is_gemini_2_0_model(model: &str) -> bool {
    model
        .strip_prefix("models/")
        .unwrap_or(model)
        .to_ascii_lowercase()
        .starts_with("gemini-2.0")
}

#[cfg(test)]
mod tests {
    use super::is_gemini_2_0_model;

    #[test]
    fn detects_prefixed_and_unprefixed_gemini_2_0_models() {
        assert!(is_gemini_2_0_model("gemini-2.0-flash"));
        assert!(is_gemini_2_0_model("models/GEMINI-2.0-PRO"));
        assert!(!is_gemini_2_0_model("gemini-2.5-flash"));
    }
}
