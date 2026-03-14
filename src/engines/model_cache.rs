use std::collections::HashMap;
use std::os::raw::{c_char, c_void};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use whisper_rs::{WhisperContext, WhisperContextParameters};

pub struct ModelCache {
    cache: Mutex<HashMap<String, Arc<WhisperContext>>>,
}

impl ModelCache {
    #[inline]
    unsafe extern "C" fn whisper_log_silencer(
        _level: u32,
        _text: *const c_char,
        _user_data: *mut c_void,
    ) {
    }

    fn silence_whisper_logs() {
        unsafe {
            whisper_rs::set_log_callback(Some(Self::whisper_log_silencer), std::ptr::null_mut());
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
