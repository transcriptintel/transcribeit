use serde_json::Value;

use crate::analysis::AnalysisResult;
use crate::output::manifest::{CacheEntry, CacheInfo};
use crate::pipeline::PipelineConfig;
use crate::transcriber::Transcript;

pub(super) fn build_cache_info(
    config: &PipelineConfig,
    transcript: &Transcript,
    analysis: Option<&AnalysisResult>,
) -> CacheInfo {
    CacheInfo {
        transcription: cache_entry_for_provider(
            &config.provider_name,
            transcript.provider_metadata.as_ref(),
            "transcription",
        ),
        analysis: analysis.map(|analysis| {
            cache_entry_for_provider(
                &analysis.provider,
                analysis.provider_metadata.as_ref(),
                "analysis",
            )
        }),
    }
}

fn cache_entry_for_provider(provider: &str, metadata: Option<&Value>, source: &str) -> CacheEntry {
    match provider {
        "gemini" => gemini_cache_entry(provider, metadata, source),
        "openai" | "azure" => openai_cache_entry(provider, metadata, source),
        "qwen-filetrans" | "nvidia-riva" | "deepgram" | "local" | "sherpa-onnx" => CacheEntry {
            provider: provider.to_string(),
            mode: "none".to_string(),
            hit: false,
            input_tokens: None,
            cached_tokens: None,
            cached_fraction: None,
            token_details: None,
            source: Some(source.to_string()),
        },
        _ => CacheEntry {
            provider: provider.to_string(),
            mode: "unknown".to_string(),
            hit: false,
            input_tokens: None,
            cached_tokens: None,
            cached_fraction: None,
            token_details: None,
            source: Some(source.to_string()),
        },
    }
}

fn gemini_cache_entry(provider: &str, metadata: Option<&Value>, source: &str) -> CacheEntry {
    let explicit_cache = metadata.is_some_and(|metadata| {
        metadata
            .pointer("/data/cached_content/enabled")
            .or_else(|| metadata.pointer("/gemini/cached_content/enabled"))
            .or_else(|| metadata.pointer("/cached_content/enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });
    let usage = metadata.and_then(|metadata| {
        metadata
            .pointer("/data/response/usage_metadata")
            .or_else(|| metadata.pointer("/gemini/response/usage_metadata"))
            .or_else(|| metadata.pointer("/response/usage_metadata"))
    });
    let input_tokens = usage
        .and_then(|usage| usage.get("promptTokenCount"))
        .and_then(Value::as_u64);
    let cached_tokens = usage
        .and_then(|usage| usage.get("cachedContentTokenCount"))
        .and_then(Value::as_u64);
    let token_details = usage
        .and_then(|usage| usage.get("cacheTokensDetails").cloned())
        .filter(|details| !details.is_null());

    CacheEntry {
        provider: provider.to_string(),
        mode: if explicit_cache {
            "explicit"
        } else {
            "implicit"
        }
        .to_string(),
        hit: cached_tokens.is_some_and(|tokens| tokens > 0),
        input_tokens,
        cached_tokens,
        cached_fraction: cache_fraction(cached_tokens, input_tokens),
        token_details,
        source: Some(source.to_string()),
    }
}

fn openai_cache_entry(provider: &str, metadata: Option<&Value>, source: &str) -> CacheEntry {
    let usage = metadata.and_then(|metadata| {
        metadata
            .pointer("/data/response/usage")
            .or_else(|| metadata.pointer("/openai/response/usage"))
            .or_else(|| metadata.pointer("/azure/response/usage"))
            .or_else(|| metadata.pointer("/response/usage"))
            .or_else(|| metadata.pointer("/usage"))
    });
    let input_tokens = usage
        .and_then(|usage| {
            usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
        })
        .and_then(Value::as_u64);
    let cached_tokens = usage
        .and_then(|usage| {
            usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .or_else(|| usage.pointer("/input_tokens_details/cached_tokens"))
        })
        .and_then(Value::as_u64);
    let token_details = usage.and_then(|usage| {
        usage
            .get("prompt_tokens_details")
            .or_else(|| usage.get("input_tokens_details"))
            .cloned()
    });

    CacheEntry {
        provider: provider.to_string(),
        mode: "implicit".to_string(),
        hit: cached_tokens.is_some_and(|tokens| tokens > 0),
        input_tokens,
        cached_tokens,
        cached_fraction: cache_fraction(cached_tokens, input_tokens),
        token_details,
        source: Some(source.to_string()),
    }
}

fn cache_fraction(cached_tokens: Option<u64>, input_tokens: Option<u64>) -> Option<f64> {
    let cached_tokens = cached_tokens?;
    let input_tokens = input_tokens?;
    if input_tokens == 0 || cached_tokens > input_tokens {
        return None;
    }
    Some(cached_tokens as f64 / input_tokens as f64)
}

#[cfg(test)]
mod tests;
