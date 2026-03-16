use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use whisper_rs::{FullParams, SamplingStrategy};

use super::model_cache::ModelCache;
use crate::transcriber::{Segment, Transcriber, Transcript};

pub struct WhisperLocal {
    model_path: String,
    cache: Arc<ModelCache>,
    language: Option<String>,
}

impl WhisperLocal {
    pub fn new(model_path: String, cache: Arc<ModelCache>, language: Option<String>) -> Self {
        Self {
            model_path,
            cache,
            language,
        }
    }
}

#[async_trait]
impl Transcriber for WhisperLocal {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let model_path = self.model_path.clone();
        let cache = Arc::clone(&self.cache);
        let language = self.language.clone();

        tokio::task::spawn_blocking(move || {
            let ctx = cache.get_or_load(&model_path)?;

            let mut state = ctx
                .create_state()
                .context("Failed to create whisper state")?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_print_special(false);
            params.set_debug_mode(false);
            if let Some(language) = language.as_deref() {
                params.set_language(Some(language));
            }

            state
                .full(params, &audio_samples)
                .context("Whisper inference failed")?;

            let segments: Vec<Segment> = state
                .as_iter()
                .map(|seg| Segment {
                    start_ms: seg.start_timestamp() * 10,
                    end_ms: seg.end_timestamp() * 10,
                    text: seg.to_string(),
                    speaker: None,
                })
                .collect();

            Ok(Transcript { segments })
        })
        .await?
    }
}
