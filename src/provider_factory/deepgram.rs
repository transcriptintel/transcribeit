use anyhow::Result;

use super::s3::build_uploader;
use super::{ProviderFactoryArgs, ProviderRuntime, owned};
use crate::credentials::resolve_provider_key;
use crate::engines::deepgram::{DeepgramApi, DeepgramConfig, DeepgramOptions};

pub(super) async fn build(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    let key = resolve_provider_key(
        owned(args.deepgram_api_key),
        owned(args.explicit_api_key),
        "deepgram",
        "--deepgram-api-key",
        "DEEPGRAM_API_KEY",
    )?;
    let model_name = args.remote_model.unwrap_or("nova-3").to_string();
    if args.speakers.is_some() && !args.diarize {
        eprintln!(
            "--speakers was provided for Deepgram, so provider-native diarization will be enabled with diarize_model=latest."
        );
    }
    if args.speakers.is_some() {
        eprintln!(
            "Deepgram does not accept an exact speaker-count hint for batch diarization; --speakers is treated as a request to enable diarization."
        );
    }
    let presigned_url_uploader = if args.deepgram_use_presigned_url {
        Some(
            build_uploader(
                args,
                "transcribeit/deepgram",
                "--provider deepgram --deepgram-use-presigned-url",
            )
            .await?,
        )
    } else {
        None
    };

    Ok(ProviderRuntime {
        engine: Box::new(DeepgramApi::new(DeepgramConfig {
            base_url: args.deepgram_api_base_url.to_string(),
            api_key: key,
            model: model_name.clone(),
            language: owned(args.language),
            settings: args.settings,
            options: DeepgramOptions {
                diarize: args.diarize || args.speakers.is_some(),
                intelligence: args.deepgram_intelligence,
                summarize: args.deepgram_summarize,
                topics: args.deepgram_topics,
                intents: args.deepgram_intents,
                detect_entities: args.deepgram_detect_entities,
                sentiment: args.deepgram_sentiment,
                keyterms: args.deepgram_keyterm.to_vec(),
                search: args.deepgram_search.to_vec(),
                redact: args.deepgram_redact.to_vec(),
                replace: args.deepgram_replace.to_vec(),
                filler_words: args.deepgram_filler_words,
                numerals: args.deepgram_numerals,
            },
            presigned_url_uploader,
            autoclean: args.cleanup_staged_resources,
        })?),
        analyzer: None,
        provider_name: "deepgram".into(),
        model_name,
    })
}
