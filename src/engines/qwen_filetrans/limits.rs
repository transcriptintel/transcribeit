use std::path::Path;

use anyhow::{Context, Result};

use crate::audio::segment::get_duration;

const SHORT_FLASH_MAX_BYTES: u64 = 10 * 1024 * 1024;
const SHORT_FLASH_MAX_SECONDS: f64 = 300.0;

pub(crate) fn is_filetrans_model(model: &str) -> bool {
    model.contains("qwen3-asr-flash-filetrans")
}

pub(crate) async fn validate_model_for_path(model: &str, audio_path: &Path) -> Result<()> {
    if is_filetrans_model(model) {
        return Ok(());
    }

    if is_short_flash_model(model) {
        let metadata = tokio::fs::metadata(audio_path)
            .await
            .with_context(|| format!("Failed to read audio metadata: {}", audio_path.display()))?;
        let duration_secs = get_duration(audio_path).await?;
        let mut violations = Vec::new();

        if metadata.len() > SHORT_FLASH_MAX_BYTES {
            violations.push(format!(
                "size is {:.2} MB, limit is 10 MB",
                metadata.len() as f64 / (1024.0 * 1024.0)
            ));
        }
        if duration_secs > SHORT_FLASH_MAX_SECONDS {
            violations.push(format!(
                "duration is {:.3}s, limit is 300.000s",
                duration_secs
            ));
        }

        let suffix = if violations.is_empty() {
            String::new()
        } else {
            format!(
                " Current file is invalid for that model: {}.",
                violations.join("; ")
            )
        };

        anyhow::bail!(
            "{model} is a short-audio Qwen3-ASR-Flash model and is not supported by --provider qwen-filetrans.{suffix} Use qwen3-asr-flash-filetrans for whole-file async transcription."
        );
    }

    anyhow::bail!(
        "{model} is not supported by --provider qwen-filetrans; use qwen3-asr-flash-filetrans."
    );
}

fn is_short_flash_model(model: &str) -> bool {
    model.contains("qwen3-asr-flash") && !model.contains("filetrans") && !model.contains("realtime")
}
