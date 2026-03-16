use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use whisper_rs::{WhisperContext, WhisperContextParameters};

pub struct ModelCache {
    cache: Mutex<HashMap<String, Arc<WhisperContext>>>,
}

impl ModelCache {
    fn silence_whisper_logs() {
        unsafe extern "C" fn noop(
            _level: std::os::raw::c_uint,
            _text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
        }
        unsafe {
            whisper_rs::set_log_callback(Some(noop), std::ptr::null_mut());
        }
    }

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
        Self::silence_whisper_logs();
        let ctx = Arc::new(
            WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
                .context("Failed to load whisper model")?,
        );
        map.insert(model_path.to_string(), Arc::clone(&ctx));
        Ok(ctx)
    }
}
