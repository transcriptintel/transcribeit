use std::ffi::{CString, c_char, c_void};
use std::fmt;
use std::path::Path;

use anyhow::{Context, Result};

const MAX_BRIDGE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const INVALID_BRIDGE_RESPONSE: i32 = 0;
const AUDIO_READ_FAILED: i32 = 6;

#[derive(Debug)]
struct BridgeError {
    code: i32,
    message: String,
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for BridgeError {}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn transcribeit_apple_speech_transcribe_file(
        path: *const c_char,
        locale: *const c_char,
        user_data: *mut c_void,
        on_done: extern "C" fn(*const u8, usize, *mut c_void),
        on_error: extern "C" fn(i32, *const u8, usize, *mut c_void),
    );
}

#[cfg(target_os = "macos")]
pub(super) fn transcribe(path: &Path, locale: Option<&str>) -> Result<String> {
    struct CallbackState {
        result: Option<Result<String, BridgeError>>,
    }

    extern "C" fn done_callback(value: *const u8, length: usize, user_data: *mut c_void) {
        let state = unsafe { &mut *user_data.cast::<CallbackState>() };
        state.result = Some(copy_bounded(value, length).map_err(|error| BridgeError {
            code: INVALID_BRIDGE_RESPONSE,
            message: error.to_string(),
        }));
    }

    extern "C" fn error_callback(
        code: i32,
        value: *const u8,
        length: usize,
        user_data: *mut c_void,
    ) {
        let state = unsafe { &mut *user_data.cast::<CallbackState>() };
        let message = copy_bounded(value, length)
            .unwrap_or_else(|_| "Apple Speech bridge returned an invalid error".to_string());
        state.result = Some(Err(BridgeError { code, message }));
    }

    let path = path
        .to_str()
        .context("Apple Speech requires a UTF-8 input path")?;
    let path = CString::new(path).context("Apple Speech input path contains a null byte")?;
    let locale = locale
        .map(CString::new)
        .transpose()
        .context("Apple Speech locale contains a null byte")?;
    let mut state = CallbackState { result: None };

    unsafe {
        transcribeit_apple_speech_transcribe_file(
            path.as_ptr(),
            locale
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            std::ptr::addr_of_mut!(state).cast(),
            done_callback,
            error_callback,
        );
    }

    match state.result {
        Some(Ok(value)) => Ok(value),
        Some(Err(error)) => Err(error.into()),
        None => anyhow::bail!("Apple Speech bridge returned without a result"),
    }
}

pub(super) fn is_audio_read_failure(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<BridgeError>()
        .is_some_and(|bridge_error| bridge_error.code == AUDIO_READ_FAILED)
}

fn copy_bounded(value: *const u8, length: usize) -> Result<String> {
    anyhow::ensure!(
        length <= MAX_BRIDGE_RESPONSE_BYTES,
        "Apple Speech bridge response exceeded 64 MiB"
    );
    anyhow::ensure!(
        !value.is_null() || length == 0,
        "Apple Speech bridge returned null data"
    );
    let bytes = if length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(value, length) }
    };
    String::from_utf8(bytes.to_vec()).context("Apple Speech bridge returned invalid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::{AUDIO_READ_FAILED, BridgeError, is_audio_read_failure};

    #[test]
    fn only_audio_read_failures_request_conversion_fallback() {
        let read_error = anyhow::Error::new(BridgeError {
            code: AUDIO_READ_FAILED,
            message: "audio read failed".to_string(),
        });
        let unrelated_error = anyhow::Error::new(BridgeError {
            code: 7,
            message: "transcription failed".to_string(),
        });

        assert!(is_audio_read_failure(&read_error));
        assert!(!is_audio_read_failure(&unrelated_error));
        assert!(!is_audio_read_failure(&anyhow::anyhow!("untyped error")));
    }
}
