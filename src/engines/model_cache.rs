use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use whisper_rs::{WhisperContext, WhisperContextParameters};

pub struct ModelCache {
    cache: Mutex<HashMap<String, Arc<WhisperContext>>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_load(&self, model_path: &str) -> Result<Arc<WhisperContext>> {
        let mut map = self
            .cache
            .lock()
            .map_err(|e| anyhow::anyhow!("Model cache lock poisoned: {e}"))?;
        if let Some(ctx) = map.get(model_path) {
            return Ok(Arc::clone(ctx));
        }
        let ctx = Arc::new(
            WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
                .context("Failed to load whisper model")?,
        );
        map.insert(model_path.to_string(), Arc::clone(&ctx));
        Ok(ctx)
    }
}
