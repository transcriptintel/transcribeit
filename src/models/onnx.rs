use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::artifacts::verify_installed_directory;
use crate::setup::QWEN3_ASR_ARCHIVE;

use super::models_dir;

pub(crate) fn resolve_onnx_model_dir(model: &str) -> Result<PathBuf> {
    let model = model.trim();
    let direct = PathBuf::from(model);
    if direct.is_dir() && is_supported_model_dir(&direct) {
        return Ok(direct);
    }

    let normalized = match model {
        "large-v3-turbo" => "turbo",
        "qwen3-asr" | "qwen3-asr-0.6b" => QWEN3_ASR_ARCHIVE,
        other => other,
    };
    let candidates = [
        format!("sherpa-onnx-whisper-{normalized}"),
        format!("sherpa-onnx-whisper-{model}"),
        normalized.to_string(),
        model.to_string(),
    ];
    for name in &candidates {
        let cache_path = models_dir().join(name);
        if cache_path.is_dir() && is_supported_model_dir(&cache_path) {
            verify_installed_directory(&cache_path)?;
            return Ok(cache_path);
        }
    }

    let cache_dir = models_dir();
    if cache_dir.is_dir() {
        let pattern = format!("{}/*{}*", cache_dir.display(), normalized);
        if let Ok(paths) = glob::glob(&pattern) {
            for entry in paths.flatten() {
                if entry.is_dir() && is_supported_model_dir(&entry) {
                    verify_installed_directory(&entry)?;
                    return Ok(entry);
                }
            }
        }
    }

    anyhow::bail!(
        "ONNX model not found for '{model}'. Expected a supported Whisper, Moonshine, SenseVoice, or Qwen3-ASR model directory.\n\
         Download Whisper with `transcribeit download-model -f onnx -s <size>` or Qwen3-ASR with `transcribeit setup --component qwen3-asr`."
    )
}

pub(crate) fn is_supported_model_dir(dir: &Path) -> bool {
    has_tokens_file(dir) || is_qwen3_asr_model_dir(dir)
}

fn is_qwen3_asr_model_dir(dir: &Path) -> bool {
    dir.join("conv_frontend.onnx").is_file()
        && (dir.join("encoder.int8.onnx").is_file() || dir.join("encoder.onnx").is_file())
        && (dir.join("decoder.int8.onnx").is_file() || dir.join("decoder.onnx").is_file())
        && dir.join("tokenizer").is_dir()
}

fn has_tokens_file(dir: &Path) -> bool {
    if dir.join("tokens.txt").is_file() {
        return true;
    }
    glob::glob(&format!("{}/*-tokens.txt", dir.display()))
        .ok()
        .and_then(|mut paths| paths.next())
        .is_some_and(|path| path.is_ok())
}

#[cfg(test)]
mod tests {
    use super::is_supported_model_dir;

    #[test]
    fn qwen_directory_does_not_require_whisper_tokens() {
        let root = tempfile::tempdir().unwrap();
        for name in [
            "conv_frontend.onnx",
            "encoder.int8.onnx",
            "decoder.int8.onnx",
        ] {
            std::fs::write(root.path().join(name), []).unwrap();
        }
        std::fs::create_dir(root.path().join("tokenizer")).unwrap();
        assert!(is_supported_model_dir(root.path()));
    }
}
