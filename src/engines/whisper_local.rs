use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use whisper_rs::{FullParams, SamplingStrategy};

use super::model_cache::ModelCache;
use crate::transcriber::{Segment, Transcriber, Transcript};

pub struct WhisperLocal {
    model_path: String,
    cache: Arc<ModelCache>,
}

impl WhisperLocal {
    pub fn new(model_path: String, cache: Arc<ModelCache>) -> Self {
        Self { model_path, cache }
    }
}

#[async_trait]
impl Transcriber for WhisperLocal {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let model_path = self.model_path.clone();
        let cache = Arc::clone(&self.cache);

        // whisper-rs is synchronous and CPU-heavy; run on a blocking thread
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

            state
                .full(params, &audio_samples)
                .context("Whisper inference failed")?;

            let num_segments = state
                .full_n_segments()
                .context("Failed to get segment count")?;

            let mut segments = Vec::new();
            for i in 0..num_segments {
                let text = state
                    .full_get_segment_text(i)
                    .context("Failed to get segment text")?;
                let start = state
                    .full_get_segment_t0(i)
                    .context("Failed to get segment start")?;
                let end = state
                    .full_get_segment_t1(i)
                    .context("Failed to get segment end")?;

                segments.push(Segment {
                    start_ms: start * 10,
                    end_ms: end * 10,
                    text,
                });
            }

            Ok(Transcript { segments })
        })
        .await?
    }
}
